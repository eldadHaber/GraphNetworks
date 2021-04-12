import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv, GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


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
nopen = 256
nhid = 256
nNclose = 256
nlayer = 64
h = 40 / nlayer
dropout = 0.3
# h = 20 / nlayer
print("dataset:", dataset)
print("n channels:", nopen)
print("n layers:", nlayer)
print("h step:", h)
print("dropout:", dropout)
batchSize = 32

h = 1


class Net(torch.nn.Module):
    def __init__(self, out_channels, nIn, k=10, aggr='max'):
        super().__init__()
        self.numlayers = 100
        self.lin0 = MLP([nIn, 64])
        self.conv1 = EdgeConv(MLP([2 * 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 64]), aggr)
        self.conv3 = EdgeConv(MLP([2 * 64, 64]), aggr)
        self.conv4 = EdgeConv(MLP([2 * 64, 64]), aggr)
        self.conv5 = EdgeConv(MLP([2 * 64, 64]), aggr)

        # self.Layers = torch.nn.ModuleList()
        # for i in torch.arange(0, self.numlayers):
        #    self.Layers.append(EdgeConv(in_channels=3, out_channels=3))

        self.lin1 = MLP([64, 64])

        self.mlp = Seq(
            MLP([64, 64]), MLP([64, 64]),
            Lin(64, out_channels))

    def forward(self, data):
        xn = data.x
        xn = F.dropout(xn, p=0.6, training=self.training)
        xn = self.lin0(xn)
        xn = F.dropout(xn, p=0.6, training=self.training)

        out = self.conv1(xn, data.edge_index)
        xn = xn - (h * out)
        xn = F.dropout(xn, p=0.6, training=self.training)

        out = self.conv2(xn, data.edge_index)
        xn = out #xn - (h * out)
        xn = F.dropout(xn, p=0.6, training=self.training)

        out = self.conv3(xn, data.edge_index)
        xn = xn - (h * out)
        xn = F.dropout(xn, p=0.6, training=self.training)

        out = self.conv4(xn, data.edge_index)
        xn = xn - (h * out)
        xn = F.dropout(xn, p=0.6, training=self.training)

        out = self.lin1(torch.cat([xn], dim=1))
        # out = global_max_pool(out, batch)
        # out = self.mlp(out)
        return F.log_softmax(out, dim=1)


if "s" in sys.argv:
    path = '/home/eliasof/GraphNetworks/data/' + dataset
else:
    path = '/home/cluster/users/erant_group/moshe/' + dataset
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
# model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
#                                  diffOrder=1, num_output=dataset.num_classes, dropOut=dropout)

model = Net(out_channels=dataset.num_classes, nIn=nNin)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)


def train():
    model.train()
    optimizer.zero_grad()
    out = model.forward(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # + 0.1*tvreg
    loss.backward()
    optimizer.step()
    # scheduler.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()

    out = model.forward(data)
    pred, accs = out.argmax(dim=-1), []
    # pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if tmp_test_acc > test_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}', flush=True)
