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
import torchvision.transforms as transforms
import torchvision

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from src import pnetArch as PNA

data_path = '/home/cluser/users/erant_group_moshe/cifar_10'
nImg = 32
nClasses = 10
transform_train = transforms.Compose([
    transforms.RandomCrop(nImg, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=6)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=6)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


nEin = 1
nopen = 64
nNin = 3
nhid = 64
nNclose = 64
nlayer = 6
h = 1 / nlayer
dropout = 0.0

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=nopen, dropOut=dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

xs = torch.arange(0, nImg)
ys = torch.arange(0, nImg)
pos = torch.meshgrid([nImg, nImg])
pos = torch.stack(pos).view(2, -1)
xtmp: PairTensor = (pos, pos)
batch = torch.zeros(pos.shape[1])
b = (batch, batch)
k = 9
edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                 num_workers=6)
edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                 num_workers=6)
edge_index = edge_index.to(device)
I = edge_index[0, :]
J = edge_index[1, :]
N = pos.shape[1]
img_graph = GO.graph(I, J, N, pos=None, faces=None)
img_graph = img_graph.to(device)

def train():
    model.train()

    total_loss = 0
    for (data, target) in trainloader:
        data = data.to(device)
        xn = data.view[-1, 3].unsqueeze(0).cuda()
        optimizer.zero_grad()
        out = model(xn, img_graph, data=data)
        loss = F.nll_loss(out, target)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(trainset)


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
    test_acc = test(testloader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()