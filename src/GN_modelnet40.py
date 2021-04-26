import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
#from src.gcn2conv import GCN2Conv
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch_cluster import knn
from torch_geometric.typing import PairTensor

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from src import pnetArch as PNA

path = '/home/cluster/users/erant_group/ModelNet10'
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=6)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(3, hidden_channels))
        self.lins.append(Linear(hidden_channels, hidden_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout
        self.mlp = Seq(
            MLP([hidden_channels, hidden_channels]), MLP([hidden_channels, hidden_channels]),
            Lin(hidden_channels, 10))
    def forward(self, data):
        xtmp: PairTensor = (data.pos, data.pos)
        b = (data.batch, data.batch)
        k = 10
        edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                         num_workers=3)
        x = data.pos
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        edge_weights = torch.ones(edge_index.shape[1]).cuda()
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index, edge_weights)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        out = global_max_pool(x, data.batch)
        x = self.mlp(out)
        return x.log_softmax(dim=-1)



nEin = 1
nopen = 64
nNin = 3
nhid = 64
nNclose = 64
nlayer = 6
h = 1 / nlayer
dropout = 0.0

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=nopen, dropOut=dropout, modelnet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        xtmp: PairTensor = (data.pos, data.pos)
        b = (data.batch, data.batch)
        k = 10
        edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                         num_workers=3)
        edge_index = edge_index.to(device)
        I = edge_index[0, :]
        J = edge_index[1, :]
        N = data.pos.shape[0]
        G = GO.graph(I, J, N, pos=None, faces=None)
        G = G.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        xn = data.pos.t().unsqueeze(0).cuda()
        out = model(xn, G, data=data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        xtmp: PairTensor = (data.pos, data.pos)
        b = (data.batch, data.batch)
        k = 10
        edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                         num_workers=3)
        edge_index = edge_index.to(device)
        I = edge_index[0, :]
        J = edge_index[1, :]
        N = data.pos.shape[0]
        G = GO.graph(I, J, N, pos=None, faces=None)
        G = G.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        xn = data.pos.t().unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(xn, G, data=data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()