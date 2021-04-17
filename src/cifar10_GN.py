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
from torch_geometric.data import Data

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from src import pnetArch as PNA

data_path = '/home/cluster/users/erant_group/moshe/cifar_10'
nImg = 32
nClasses = 10
transform_train = transforms.Compose([
    transforms.RandomCrop(nImg, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

batch_size = 1
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=6)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                         num_workers=6)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


nEin = 1
nopen = 128
nNin = 3
nhid = 128
nNclose = 128
nlayer = 6
h = 1  # 10 / nlayer
dropout = 0.0

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=nopen, dropOut=dropout, modelnet=True, gated=False,
                                  realVarlet=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
import numpy as np

xs = np.arange(0, nImg)
ys = np.arange(0, nImg)
pos = np.meshgrid(xs, ys, indexing='ij')
pos = np.stack(pos)
pos = torch.from_numpy(pos).view(2, -1).t()

xtmp: PairTensor = (pos, pos)
batch = torch.zeros(pos.shape[0], dtype=torch.int64)
b = (batch, batch)
k = 8
edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                 num_workers=6)
edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                 num_workers=6)
edge_index = edge_index.to(device)
edge_index_batch = edge_index.clone()
if batch_size > 1:
    for i in range(1, batch_size):
        edge_index_new = edge_index + i * pos.shape[0]

        edge_index_batch = torch.cat([edge_index_batch, edge_index_new], dim=1)
        batch = torch.cat([batch, i * torch.ones(pos.shape[0], dtype=torch.int64)], dim=0)
batch = batch.to(device)
print("Edge index:", edge_index, "shape:", edge_index.shape)
edge_index = edge_index_batch
print("Edge index batch:", edge_index, "shape:", edge_index.shape)

I = edge_index[0, :]
J = edge_index[1, :]
N = pos.shape[0] * batch_size
img_graph = GO.graph(I, J, N, pos=None, faces=None)
img_graph = img_graph.to(device)

import matplotlib.pyplot as plt


def train():
    model.train()

    total_loss = 0
    tmp_loss = 0
    for i, (data, target) in enumerate(trainloader):
        data = data.to(device)
        features = data.clone()
        # plt.figure()
        # print("pos shape:", pos.shape)
        # print("features shape:", features.shape)
        # plt.scatter(x=pos[:, 0], y=pos[:, 1],
        #             s=features.clone().detach().cpu().numpy().squeeze()[0, :, :].flatten().squeeze())
        #
        # plt.figure()
        # plt.imshow(data.clone().squeeze().permute(1, 2, 0).cpu().numpy()[:, :, 0])
        # plt.savefig('/users/others/eliasof/GraphNetworks/plots/input.jpg')
        # # plt.close()
        # print("features:", features.squeeze()[0, 0, :])
        # print("features shape:", features.shape)


        data.batch = batch
        target = target.to(device)
        xn = features.view(3, -1).unsqueeze(0).cuda()

        optimizer.zero_grad()
        out = model(xn, img_graph, data=data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tmp_loss += loss.item()
        if i % 100 == 99:
            print("Train loss:", tmp_loss / 100)
            tmp_loss = 0
    return total_loss / len(trainset)


def test(loader):
    model.eval()

    correct = 0
    for (data, target) in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        xn = data.view(-1, 3).t().unsqueeze(0).cuda()
        data.batch = batch
        with torch.no_grad():
            pred = model(xn, img_graph, data=data).max(dim=1)[1]
        correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    test_acc = test(testloader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()
