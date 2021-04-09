import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN


def getIterData_MD17(Coords, device='cpu'):

    D = torch.relu(torch.sum(Coords ** 2, dim=0, keepdim=True) + \
                   torch.sum(Coords ** 2, dim=0, keepdim=True).t() - \
                   2 * Coords.t() @ Coords)
    D = D / D.std()
    D = torch.exp(-2 * D)

    nsparse = Coords.shape[-1]
    vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)

    nEdges = I.shape[0]
    xe = torch.zeros(1, 1, nEdges, device=device)
    for i in range(nEdges):
        if I[i] + 1 == J[i]:
            xe[:, :, i] = 1
        if I[i] - 1 == J[i]:
            xe[:, :, i] = 1

    Coords = Coords.to(device=device, non_blocking=True)
    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)
    xe = xe.to(device=device, non_blocking=True)
    D = D.to(device=device, non_blocking=True)

    return Coords.unsqueeze(0), I, J, xe, D

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
# load training data
data = np.load('../../../data/MD/MD17/aspirin_dft.npz')
E = data['E']
Force = data['F']
R = data['R']
z = data['z']
# All geometries in Å, energy labels in kcal mol-1 and force labels in kcal mol-1 Å-1.
# According to http://quantum-machine.org/gdml/#datasets, we need to reach a test error of 0.35 kcal mol-1 Å-1 on force prediction to be compatible with the best



ndata = E.shape[0]
natoms = z.shape[0]

print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

# Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
n_train = 1000
n_val = 1000

ndata_rand = np.arange(ndata)
np.random.shuffle(ndata_rand)

E_train = torch.from_numpy(E[:n_train])
F_train = torch.from_numpy(Force[:n_train])
R_train = torch.from_numpy(R[:n_train])

E_val = torch.from_numpy(E[n_train:n_train+n_val])
F_val = torch.from_numpy(Force[n_train:n_train+n_val])
R_val = torch.from_numpy(R[n_train:n_train+n_val])

E_test = torch.from_numpy(E[n_train+n_val:])
F_test = torch.from_numpy(Force[n_train+n_val:])
R_test = torch.from_numpy(R[n_train+n_val:])

# Setup the network and its parameters
nNin = 1
nEin = 1
nNopen = 128
nEopen = 64
nEhid = 64
nNclose = 3
nEclose = 1
nlayer = 6

model = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.1, dense=False)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####
lrO = 1e-3
lrC = 1e-3
lrN = 1e-3
lrE1 = 1e-3
lrE2 = 1e-3

optimizer = optim.Adam([{'params': model.K1Nopen, 'lr': lrO},
                        {'params': model.K2Nopen, 'lr': lrC},
                        {'params': model.K1Eopen, 'lr': lrO},
                        {'params': model.K2Eopen, 'lr': lrC},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.KN1, 'lr': lrE1},
                        {'params': model.KN2, 'lr': lrE2},
                        {'params': model.KNclose, 'lr': lrE2}])


alossBest = 1e6
epochs = 1000

bestModel = model
hist = torch.zeros(epochs)

for epoch in range(epochs):
    aloss = 0.0
    MAE = 0.0
    for i in range(n_train):
        # Get the data
        Ri = R_train[i,:]
        Fi = F_train[i,:].to(dtype=torch.float32)
        Ei = E_train[i,:]

        Coords, I, J, xe, Ds = getIterData_MD17(Ri.t())

        nNodes = Ds.shape[0]
        w = Ds[I, J].to(dtype=torch.float32)
        G = GO.graph(I, J, nNodes, w)
        xn = torch.from_numpy(z).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        xe = w.unsqueeze(0).unsqueeze(0)

        optimizer.zero_grad()

        xnOut, xeOut = model(xn, xe, G)
        F_true = Fi.t()[None,:,:]
        loss = F.mse_loss(xnOut, F_true)
        MAEi = torch.mean(torch.abs(xnOut - F_true)).detach()
        MAE += MAEi
        loss.backward()

        aloss += loss.detach()
        gN = model.KNclose.grad.norm().item()
        gE1 = model.KE1.grad.norm().item()
        gE2 = model.KE2.grad.norm().item()
        gO = model.KN1.grad.norm().item()
        gC = model.KN2.grad.norm().item()

        optimizer.step()
        nprnt = 10
        if (i + 1) % nprnt == 0:
            aloss /= nprnt
            MAE /= nprnt
            # print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
            #       (j, i, aloss, gO, gN, gE1, gE2, gC), flush=True)

            print(f'{epoch:2d}.{i:4d}  Loss: {aloss:.2f}  MAE: {MAE:.2f} ')
            aloss = 0.0
            MAE = 0.0

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

