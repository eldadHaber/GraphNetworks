import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# from torch_geometric.utils import grid
from torch_geometric.datasets import ModelNet, FAUST
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

caspver = "casp11"  # Change this to choose casp version

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
nNin = 1
nEin = 3
nopen = 64
nhid = 64
nNclose = 64
nlayer = 10

batchSize = 32


modelnet_path = '/home/cluster/users/erant_group/ModelNet10'
faust_path = '/home/cluster/users/erant_group/faust'
transforms = T.FaceToEdge(remove_faces=False)
#train_dataset = ModelNet(modelnet_path, '10', train=True, transform=transforms)


pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])
train_dataset = FAUST(faust_path, True, T.Cartesian(), pre_transform)
test_dataset = FAUST(faust_path, False, T.Cartesian(), pre_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


#train_dataset = FAUST(faust_path, train=True, transform=transforms)
d = train_dataset[0]

model = GN.graphNetwork_try(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.5, dense=False, varlet=True, wave=False,
                 diffOrder=1, num_nodes=d.num_nodes)

model.to(device)

target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#train_loader = DataLoader(
#    train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

#test_dataset = FAUST(faust_path, train=False, transform=transforms)
#test_loader = DataLoader(
#    train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

def train(epoch):
    model.train()

    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.pos.shape[0]
        G = GO.graph(I, J, N, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)
        xnOut = model(xn, xe, G)
        loss = F.nll_loss(xnOut, target)
        total_loss += loss.item()
        loss.backward
        optimizer.step()

        if i % 10 == 9:
            print("train loss:", total_loss / 10)
            total_loss = 0


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.pos.shape[0]
        G = GO.graph(I, J, N, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)
        xnOut = model(xn, xe, G)

        pred = xnOut.max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes)


for epoch in range(1, 101):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))









for i, data in enumerate(train_loader):
    print(data.pos)
    print("pos:", data.pos.shape)
    print(data.edge_index)
    print("edge index:", data.edge_index.shape)

    I = data.edge_index[0, :]
    print("I shape:", I.shape)
    J = data.edge_index[1, :]
    N = data.pos.shape[0]
    print("data:", data)
    G = GO.graph(I, J, N, pos=data.pos, faces=data.face.t())

    xn = torch.randn(1, 1, N).float()
    xn = torch.zeros(1, 1, N).float()
    xn[:, :, 1:100] = 1.0
    xn[:, :, 1000:1700] = 1.0
    xe = torch.ones(1, 1, data.edge_index.shape[1])

    xnOut, xeOut = model(xn, xe, G)
    exit()