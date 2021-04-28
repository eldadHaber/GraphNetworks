# import os, sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import math
#
# # from torch_geometric.utils import grid
# from torch_geometric.datasets import ModelNet, FAUST
# from torch_geometric.data import DataLoader
# import torch_geometric.transforms as T
#
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# caspver = "casp11"  # Change this to choose casp version
#
# if "s" in sys.argv:
#     base_path = '/home/eliasof/pFold/data/'
#     import graphOps as GO
#     import processContacts as prc
#     import utils
#     import graphNet as GN
#     import pnetArch as PNA
#
#
# elif "e" in sys.argv:
#     base_path = '/home/cluster/users/erant_group/pfold/'
#     from src import graphOps as GO
#     from src import processContacts as prc
#     from src import utils
#     from src import graphNet as GN
#     from src import pnetArch as PNA
#
#
# else:
#     base_path = '../../../data/'
#     from src import graphOps as GO
#     from src import processContacts as prc
#     from src import utils
#     from src import graphNet as GN
#     from src import pnetArch as PNA
#
# # Setup the network and its parameters
# nNin = 1
# nEin = 3
# nopen = 256
# nhid = 256
# nNclose = 256
# nlayer = 4
#
# batchSize = 32
#
# modelnet_path = '/home/cluster/users/erant_group/ModelNet10'
# faust_path = '/home/cluster/users/erant_group/faust'
# transforms = T.FaceToEdge(remove_faces=False)
# # train_dataset = ModelNet(modelnet_path, '10', train=True, transform=transforms)
#
#
# pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])
# train_dataset = FAUST(faust_path, True, T.Cartesian(), pre_transform)
# test_dataset = FAUST(faust_path, False, T.Cartesian(), pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1)
#
# # train_dataset = FAUST(faust_path, train=True, transform=transforms)
# d = train_dataset[0]
#
# model = GN.graphNetwork_try(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=True, wave=True,
#                             diffOrder=1, num_output=d.num_nodes)
# # dropout = 0.0
# # wave = True
# # model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=0.01, dense=False, varlet=True, wave=wave,
# #                                   diffOrder=1, num_output=d.num_nodes, dropOut=dropout, faust=True,
# #                                   gated=False,
# #                                   realVarlet=False, mixDyamics=False)
#
#
# model.to(device)
#
# target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#
# # train_loader = DataLoader(
# #    train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
#
# # test_dataset = FAUST(faust_path, train=False, transform=transforms)
# # test_loader = DataLoader(
# #    train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
#
# def train(epoch):
#     model.train()
#
#     if epoch == 20:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.001
#
#     total_loss = 0
#     for i, data in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#
#         I = data.edge_index[0, :]
#         J = data.edge_index[1, :]
#         N = data.pos.shape[0]
#         W = torch.ones(N).to(device)
#         G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
#         G = G.to(device)
#         xn = data.x.t().unsqueeze(0)
#         xe = data.edge_attr.t().unsqueeze(0)
#
#         # print("I shape:", I.shape)
#         # print("edge index shape:", data.edge_index.shape)
#         # print("xn shape:", xn.shape)
#         # print("xe shape:", xe.shape)
#         xnOut = model(xn, xe, G)
#         # xnOut = model(xn, G)
#         # print(xnOut.shape)
#         # print(target.shape)
#         loss = F.nll_loss(xnOut, target)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#
#         if i % 10 == 9:
#             print("train loss:", total_loss / 10)
#             total_loss = 0
#
#
# def test():
#     model.eval()
#     correct = 0
#
#     for data in test_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#
#         I = data.edge_index[0, :]
#         J = data.edge_index[1, :]
#         N = data.pos.shape[0]
#         W = torch.ones(N).to(device)
#         G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
#         G = G.to(device)
#         xn = data.x.t().unsqueeze(0)
#         xe = data.edge_attr.t().unsqueeze(0)
#         xnOut = model(xn, xe, G)
#         # xnOut = model(xn, G)
#         pred = xnOut.max(1)[1]
#         correct += pred.eq(target).sum().item()
#     return correct / (len(test_dataset) * d.num_nodes)
#
#
#
#
# for epoch in range(1, 101):
#     train(epoch)
#     test_acc = test()
#     print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
#
# for i, data in enumerate(train_loader):
#     print(data.pos)
#     print("pos:", data.pos.shape)
#     print(data.edge_index)
#     print("edge index:", data.edge_index.shape)
#
#     I = data.edge_index[0, :]
#     print("I shape:", I.shape)
#     J = data.edge_index[1, :]
#     N = data.pos.shape[0]
#     print("data:", data)
#     G = GO.graph(I, J, N, pos=data.pos, faces=data.face.t())
#
#     xn = torch.randn(1, 1, N).float()
#     xn = torch.zeros(1, 1, N).float()
#     xn[:, :, 1:100] = 1.0
#     xn[:, :, 1000:1700] = 1.0
#     xe = torch.ones(1, 1, data.edge_index.shape[1])
#
#     xnOut, xeOut = model(xn, xe, G)
#     exit()


##--------------------new code is up---------##
##below is old code###

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
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
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
nNin = 6  # 6  # 6
nEin = 3  # 3
nopen = 3  # 64
nhid = 3  # 64
nNclose = 3  # 64
nlayer = 50  # 8#16

batchSize = 32
h = 0.02
lr = 0.01
lrGCN = 0.01
wdGCN = 0
wd = 5e-4

print("nchannels:", nopen)
print("nlayers:", nlayer)
print("h:", h)
print("lr:", lr)
print("lr gcn:", lrGCN)
print("wdgcn:", wdGCN)
print("wd:", wd)

modelnet_path = '/home/cluster/users/erant_group/ModelNet10'
faust_path = '/home/cluster/users/erant_group/faust'
transforms = T.FaceToEdge(remove_faces=False)
# train_dataset = ModelNet(modelnet_path, '10', train=True, transform=transforms)


pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])
train_dataset = FAUST(faust_path, True, T.Cartesian(), pre_transform)
test_dataset = FAUST(faust_path, False, T.Cartesian(), pre_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1)

# train_dataset = FAUST(faust_path, train=True, transform=transforms)
d = train_dataset[0]

model = GN.graphNetwork_faust(nNin, nEin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=True,
                              diffOrder=1, num_nodes=d.num_nodes, mixDynamics=False)

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=False,
                                  diffOrder=1, num_output=d.num_nodes, dropOut=0.0, faust=True,
                                  gated=False,
                                   realVarlet=False, mixDyamics=False)


model.to(device)

target = torch.arange(d.num_nodes, dtype=torch.long, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam([
#     dict(params=model.KN1, lr=lrGCN, weight_decay=0),
#     dict(params=model.KN2, lr=lrGCN, weight_decay=0),
#     dict(params=model.K1Eopen, lr=lr),
#     dict(params=model.K2Eopen, lr=lr),
#     dict(params=model.KE1, lr=lrGCN),
#     dict(params=model.KE2, lr=lrGCN),
#     dict(params=model.K1Nopen, weight_decay=0),
#     dict(params=model.KNclose, weight_decay=0),
#     dict(params=model.lin1.parameters(), weight_decay=0),
#     dict(params=model.lin2.parameters(), weight_decay=0),
#     dict(params=model.alpha, lr=0.01, weight_decay=0),
# ], lr=lr)


## bring this opt:
# optimizer = torch.optim.Adam([
#     dict(params=model.KN1, lr=lrGCN, weight_decay=wdGCN),
#     dict(params=model.KN2, lr=lrGCN, weight_decay=wdGCN),
#     dict(params=model.K1Nopen, weight_decay=0),
#     dict(params=model.KNclose, weight_decay=0),
#     dict(params=model.lin1.parameters(), weight_decay=0),
#     dict(params=model.lin2.parameters(), weight_decay=0),
#     dict(params=model.alpha, lr=0.1, weight_decay=0),
# ], lr=lr)

print_files = False
if print_files:
    file2Open = "src/faust_test.py"
    print("------------------------------------ Driver file: ------------------------------------")

    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("------------------------------------ Graph Networks file: ------------------------------------")
    file2Open = "src/graphNet.py"
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

# train_loader = DataLoader(
#    train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

# test_dataset = FAUST(faust_path, train=False, transform=transforms)
# test_loader = DataLoader(
#    train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
betas = []


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
        W = torch.ones(N).to(device)
        G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xn = data.pos.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)

        # print("I shape:", I.shape)
        # print("edge index shape:", data.edge_index.shape)
        # print("xn shape:", xn.shape)
        # print("xe shape:", xe.shape)
        [xnOut, beta] = model(xn, G, xe=xe)
        # betas.append(beta)
        print("beta:", beta)
        loss = F.nll_loss(xnOut, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print("train loss:", total_loss / 10)
            total_loss = 0


acc_hist = []


def test():
    model.eval()
    correct = 0

    for idx, data in enumerate(test_loader):
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.pos.shape[0]
        W = torch.ones(N).to(device)
        G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xn = data.pos.t().unsqueeze(0)
        xe = data.edge_attr.t().unsqueeze(0)
        [xnOut, beta] = model(xn, G, xe=xe)
        if idx == 0:
            betas.append(beta)
        pred = xnOut.max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes)


debug = True
if debug:
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
        N = data.pos.shape[0]
        W = torch.ones(N).to(device)
        G = GO.graph(I, J, N, W=W, pos=data.pos, faces=data.face.t())

        # xn = torch.randn(1, 1, N).float()
        # xn = torch.zeros(1, 1, N).float()
        # xn[:, :, 1:100] = 1.0
        # xn[:, :, 1000:1700] = 1.0
        pos, batch = data.pos, data.batch
        xn = pos.t().unsqueeze(0)  # torch.zeros(pos.shape[0], 3).float()
        xe = data.edge_attr.t().unsqueeze(0)  # torch.ones(1, 1, data.edge_index.shape[1])

        xnOut, xeOut = model(xn,G, xe=xe)
        exit()

for epoch in range(1, 101):
    train(epoch)
    test_acc = test()
    acc_hist.append(test_acc)
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))

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
plt.savefig("faust_wave_graph.png")

with open('betas_faust_wave.txt', 'w') as filehandle:
    for listitem in betas:
        filehandle.write('%s\n' % listitem)

with open('acc_hist_faust_wave.txt', 'w') as filehandle:
    for listitem in acc_hist:
        filehandle.write('%s\n' % listitem)

print("bye")

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
