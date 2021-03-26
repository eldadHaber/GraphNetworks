import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import sys

if "s" in sys.argv:
    base_path = '/home/eliasof/pFold/data/'
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
nNin = 1433
nEin = 1
nopen = 64
nhid = 64
nNclose = 64
nlayer = 32
h = 1 / nlayer

batchSize = 32

dataset = 'Cora'
path = '/home/cluster/users/erant_group/moshe/cora'
transform = T.Compose([T.NormalizeFeatures()])
# transform = T.Compose([T.NormalizeFeatures(), T.()])

dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


# data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=0.01)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
model = GN.graphNetwork_try(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=True, wave=True,
                            diffOrder=1, num_output=dataset.num_classes, dropOut=True)

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=dataset.num_classes, dropOut=True)
model.reset_parameters()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
optimizer = torch.optim.Adam([
    dict(params=model.KN1, weight_decay=0.01),
    dict(params=model.KN2, weight_decay=0.01),
    dict(params=model.K1Nopen, weight_decay=5e-4),
    dict(params=model.KNclose, weight_decay=5e-4)
], lr=0.01)


# optimizer = torch.optim.Adam([
#     dict(params=model.convs.parameters(), weight_decay=0.01),
#     dict(params=model.K1Nopen, weight_decay=5e-4),
#     dict(params=model.KNclose, weight_decay=5e-4)
# ], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
def train():
    model.train()
    optimizer.zero_grad()
    print("data:", data)
    print("dataset.num_classes:", dataset.num_classes)
    print("dataset.num_features:", dataset.num_features)
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
    [out, G] = model(xn, G)
    print("out shape:", out.shape)
    [valmax, argmax] = torch.max(out, dim=1)
    tvreg = torch.norm(G.nodeGrad(out.t().unsqueeze(0)), p=1) / I.shape[0]
    # print("tvreg:", tvreg)
    # out = out.squeeze()
    loss = 0.1 * tvreg + F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
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
    [out, G] = model(xn, G)
    pred, accs = out.argmax(dim=-1), []
    # pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}')
