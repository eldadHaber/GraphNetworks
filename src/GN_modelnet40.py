import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
# from src.gcn2conv import GCN2Conv
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch_cluster import knn
from torch_geometric.typing import PairTensor


import sys
if "s" in sys.argv:
    import graphOps as GO
    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')

else:
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA

    path = '/home/cluster/users/erant_group/moshe/ModelNet10_fixed'


def transferWeights(smallmodel, largemodel, interp=True):
    # Assuming the interpolation is always doubling the number of layers
    if interp:
        K1 = smallmodel.KN1.unsqueeze(0).unsqueeze(0).clone()
        K2 = smallmodel.KN2.unsqueeze(0).unsqueeze(0).clone()
        Kopen = smallmodel.K1Nopen.clone()
        Kclose = smallmodel.KNclose.clone()
        mlp = smallmodel.mlp#.clone().detach()


        largemodel.mlp = mlp #torch.nn.Parameter(mlp)

        largemodel.KNclose = torch.nn.Parameter(Kclose)
        largemodel.K1Nopen = torch.nn.Parameter(Kopen)

        K1 = torch.nn.functional.interpolate(K1, size=[2*K1.shape[2], K1.shape[3], K1.shape[4]], mode='trilinear', align_corners=True).squeeze()
        K2 = torch.nn.functional.interpolate(K2, size=[2*K2.shape[2], K2.shape[3], K2.shape[4]], mode='trilinear', align_corners=True).squeeze()

        largemodel.KN1 = torch.nn.Parameter(K1)
        largemodel.KN2 = torch.nn.Parameter(K2)
    else:
        K1 = smallmodel.KN1.clone().detach()
        K2 = smallmodel.KN2.clone().detach()
        Kopen = smallmodel.K1Nopen.clone().detach()
        Kclose = smallmodel.KNclose.clone().detach()
        mlp = smallmodel.mlp #.clone().detach()

        largemodel.mlp = mlp #torch.nn.Parameter(mlp)
        largemodel.KNclose = torch.nn.Parameter(Kclose)
        largemodel.K1Nopen = torch.nn.Parameter(Kopen)

        new_KN1 = largemodel.KN1.clone().detach()
        new_KN2 = largemodel.KN2.clone().detach()
        new_KN1[0:K1.shape[0]] = K1
        new_KN2[0:K2.shape[0]] = K2

        largemodel.KN1 = torch.nn.Parameter(new_KN1)
        largemodel.KN2 = torch.nn.Parameter(new_KN1)

#path = '/home/cluster/users/erant_group/ModelNet10'
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=6)


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
                         num_workers=6)
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


printFiles = False
if printFiles:
    print("**********************************************************************************")
    file2Open = "src/optuna_GNcora.py"
    print("DRIVER CODE:")
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("NETWORKS CODE:")
    file2Open = "src/graphNet.py"
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("**********************************************************************************")

nEin = 1
nopen = 128
nNin = 3
nhid = 128
nNclose = 128
nlayer = 4
h = 0.1  # / nlayer
dropout = 0.0
wave = False
import datetime

now = datetime.datetime.now()
filename = 'nopen_' + str(nopen) + 'nhid_' + str(nhid) + 'nlayer_' + str(nlayer) + 'h_' + str(h) + 'dropout_' + str(
    dropout) + 'wave_' + str(wave) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(
    now.minute) + "_" + str(
    now.second) + '.pth'
model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=wave,
                                  diffOrder=1, num_output=nlayer*nopen, dropOut=dropout, modelnet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

optimizer = torch.optim.Adam([
    dict(params=model.KN1, lr=0.001, weight_decay=0),
    dict(params=model.KN2, lr=0.001, weight_decay=0),
    dict(params=model.convs1x1, weight_decay=0),
    dict(params=model.K1Nopen, weight_decay=0),
    dict(params=model.KNclose, weight_decay=0),
    dict(params=model.mlp.parameters(), weight_decay=0)
    # dict(params=model.alpha, lr=0.1, weight_decay=0),
], lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        xtmp: PairTensor = (data.pos, data.pos)
        b = (data.batch, data.batch)
        k = 10
        edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                         num_workers=6)
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

save_path = '/home/cluster/users/erant_group/moshe/pdegcnCheckpoints/' + filename
accs = []
best_test_acc = 0
for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    accs.append(test_acc)
    if (nlayer < 8) and (epoch % 100 == 99):
        nlayer = nlayer * 2
        #h = h / 2
        model_new = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=wave,
                                          diffOrder=1, num_output=nopen, dropOut=dropout, modelnet=True)
        model_new.to(device)
        transferWeights(model, model_new, interp=True)
        model = model_new
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        #torch.save(model.state_dict(), save_path)

    scheduler.step()


if 1 == 1:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(accs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.savefig("modelnet_heat_continuation.png")

    with open('acc_hist_modelnet_heat_continuation.txt', 'w') as filehandle:
        for listitem in accs:
            filehandle.write('%s\n' % listitem)

print("bye")
