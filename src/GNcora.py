import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import optuna

import sys

if "s" in sys.argv:
    base_path = '/home/eliasof/pdeGraphs/data/'
    import graphOps as GO
    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA


elif "e" in sys.argv:
    base_path = '/home/cluster/users/erant_group/pfold/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA


else:
    base_path = '../../../data/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA

# Setup the network and its parameters
dataset = 'Cora'

if dataset == 'Cora':
    nNin = 1433
elif dataset == 'CiteSeer':
    nNin = 3703
elif dataset == 'PubMed':
    nNin = 500
nEin = 1
nopen = 64
nhid = 64
nNclose = 64
nlayer = 8
h = 1.5  # 16 / nlayer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties('cuda:0'))


dropout = 0.6
# h = 20 / nlayer
print("dataset:", dataset)
print("n channels:", nopen)
print("n layers:", nlayer)
print("h step:", h)
print("dropout:", dropout)
batchSize = 32

if "s" in sys.argv:
    path = '/home/eliasof/GraphNetworks/data/' + dataset
else:
    path = '/home/cluster/users/erant_group/moshe/' + dataset
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
realVarlet = False
model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=dataset.num_classes, dropOut=dropout, gated=False,
                                  realVarlet=realVarlet, mixDyamics=True)
model.reset_parameters()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
if not realVarlet:
    optimizer = torch.optim.Adam([
        dict(params=model.KN1, lr=0.00001, weight_decay=0),
        dict(params=model.KN2, lr=0.00001, weight_decay=0),
        dict(params=model.K1Nopen, weight_decay=5e-4),
        dict(params=model.KNclose, weight_decay=5e-4),
        dict(params=model.alpha, lr=0.01, weight_decay=0)
    ], lr=0.01)
else:
    optimizer = torch.optim.Adam([
        dict(params=model.KN1, weight_decay=0.01),
        dict(params=model.KN2, weight_decay=0.01),
        dict(params=model.KE1, weight_decay=0.01),
        dict(params=model.K1Nopen, weight_decay=5e-4),
        dict(params=model.K2Nopen, weight_decay=5e-4),
        dict(params=model.KNclose, weight_decay=5e-4)

    ], lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)


# optimizer = torch.optim.Adam([
#     dict(params=model.convs.parameters(), weight_decay=0.01),
#     dict(params=model.K1Nopen, weight_decay=5e-4),
#     dict(params=model.KNclose, weight_decay=5e-4)
# ], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

betas = []
def train():
    model.train()
    optimizer.zero_grad()
    I = data.edge_index[0, :]
    J = data.edge_index[1, :]
    N = data.y.shape[0]

    features = data.x.squeeze().t()
    D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                   torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                   2 * features.t() @ features)

    D = D / D.std()
    D = torch.exp(-2 * D)

    w = D[I, J]
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)
    xn = data.x.t().unsqueeze(0)
    xe = torch.ones(1, 1, I.shape[0]).to(device)

    # out = model(xn, xe, G)
    [out, G, beta] = model(xn, G)
    betas.append(beta.item())
    g = G.nodeGrad(out.t().unsqueeze(0))
    eps = 1e-4
    absg = torch.sum(g ** 2, dim=1)
    tvreg = absg.mean()
    # tvreg = torch.norm(G.nodeGrad(out.t().unsqueeze(0)), p=1) / I.shape[0]
    # out = out.squeeze()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # + 0.1*tvreg
    loss.backward()
    optimizer.step()
    scheduler.step()

    #gKN2 = model.KN2.grad.norm().item()
    gKN2 = 0
    #gKN1 = model.KN1.grad.norm().item()
    gKN1 = 0
    gKo = model.K1Nopen.grad.norm().item()
    gKc = model.KNclose.grad.norm().item()
    print("gKo gKN1  gKN2    gKc")
    print("%10.3E   %10.3E   %10.3E   %10.3E" %
          (gKo, gKN1, gKN2, gKc), flush=True)
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    I = data.edge_index[0, :]
    J = data.edge_index[1, :]
    N = data.y.shape[0]
    features = data.x.squeeze().t()
    D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                   torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                   2 * features.t() @ features)

    D = D / D.std()
    D = torch.exp(-2 * D)
    w = D[I, J]
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)
    xn = data.x.t().unsqueeze(0)
    xe = torch.ones(1, 1, I.shape[0]).to(device)
    # out = model(xn, xe, G)
    [out, G, _] = model(xn, G)
    pred, accs = out.argmax(dim=-1), []
    # pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
acc_hist = []
for epoch in range(1, 500):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    acc_hist.append(tmp_test_acc)
    if tmp_test_acc > test_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}', flush=True)


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch #')
ax1.set_ylabel('Wave portion', color=color)
ax1.plot(betas, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Test acc (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(acc_hist, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.savefig("cora_heat_graph.png")

with open('betas_cora_heat.txt', 'w') as filehandle:
    for listitem in betas:
        filehandle.write('%s\n' % listitem)

with open('acc_hist_cora_heat.txt', 'w') as filehandle:
    for listitem in acc_hist:
        filehandle.write('%s\n' % listitem)


print("bye")
