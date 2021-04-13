import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import torch.autograd.profiler as profiler
import imageio

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader

from src.MD17_utils import getIterData_MD17, print_distogram, print_3d_structure, Dataset_MD17, getBatchData_MD17

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print_distograms = False
    print_3d_structures = False
    # load training data
    data = np.load('../../../data/MD/MD17/aspirin_dft.npz')
    E = data['E']
    Force = data['F']
    R = data['R']
    z = torch.from_numpy(data['z']).to(dtype=torch.float32, device=device)
    # All geometries in Å, energy labels in kcal mol-1 and force labels in kcal mol-1 Å-1.
    # According to http://quantum-machine.org/gdml/#datasets, we need to reach a test error of 0.35 kcal mol-1 Å-1 on force prediction to be compatible with the best



    ndata = E.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
    n_train = 10
    n_val = 1000
    batch_size = 5

    ndata_rand = 0 + np.arange(ndata)
    # np.random.shuffle(ndata_rand)


    E_train = torch.from_numpy(E[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)
    F_train = torch.from_numpy(Force[ndata_rand[:n_train]]).transpose(1,2).to(dtype=torch.float32, device=device)
    R_train = torch.from_numpy(R[ndata_rand[:n_train]]).transpose(1,2).to(dtype=torch.float32, device=device)

    E_val = torch.from_numpy(E[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)
    F_val = torch.from_numpy(Force[ndata_rand[n_train:n_train+n_val]]).transpose(1,2).to(dtype=torch.float32, device=device)
    R_val = torch.from_numpy(R[ndata_rand[n_train:n_train+n_val]]).transpose(1,2).to(dtype=torch.float32, device=device)

    E_test = torch.from_numpy(E[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)
    F_test = torch.from_numpy(Force[ndata_rand[n_train+n_val:]]).transpose(1,2).to(dtype=torch.float32, device=device)
    R_test = torch.from_numpy(R[ndata_rand[n_train+n_val:]]).transpose(1,2).to(dtype=torch.float32, device=device)


    dataset_train = Dataset_MD17(R_train, F_train, E_train, z)
    dataset_val = Dataset_MD17(R_val, F_val, E_val, z)
    dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False)

    # Setup the network and its parameters
    nNin = 1
    nEin = 1
    nNopen = 128
    nEopen = 64
    nEhid = 64
    nNclose = 1
    nEclose = 1
    nlayer = 6

    model = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nEclose, nlayer, h=.1, dense=False)
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
    epochs = 100000

    bestModel = model
    hist = torch.zeros(epochs)
    eps = 1e-10
    nprnt = 1
    nprnt2 = min(nprnt, n_train)
    t0 = time.time()

    fig = plt.figure(num=1,figsize=[10,10])
    for epoch in range(epochs):
        aloss = 0.0
        aloss_E = 0.0
        aloss_F = 0.0
        MAE = 0.0
        Fps = 0.0
        Fts = 0.0
        for i, (Ri, Fi, Ei, zi) in enumerate(dataloader_train):
            nb = Ri.shape[0]
            # Get the data
            Ri.requires_grad_(True)

            I, J, xe, D, iD, nnodes, w = getBatchData_MD17(Ri, device='cpu')
            if print_distograms:
                print_distogram(Ds,Ei,iDs,i)
                continue
            elif print_3d_structures:
                print_3d_structure(fig,z,Ri,Fi)
                continue

            N = torch.sum(torch.tensor(nnodes))
            G = GO.graph(I, J, N, w)
            xn = zi.view(-1).unsqueeze(0).unsqueeze(0)
            xe = w.unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()

            xnOut, xeOut = model(xn, xe, G)

            E_pred = torch.sum(xnOut)
            F_pred = -grad(E_pred, Ri, create_graph=True)[0].requires_grad_(True)
            loss_F = F.mse_loss(F_pred, Fi)
            # loss_E = F.mse_loss(E_pred, Ei)
            loss = loss_F
            Fps += torch.mean(torch.sqrt(torch.sum(F_pred.detach()**2,dim=1)))
            Fts += torch.mean(torch.sqrt(torch.sum(Fi**2,dim=1)))
            MAEi = torch.mean(torch.abs(F_pred - Fi)).detach()
            MAE += MAEi
            loss.backward()

            aloss += loss.detach()
            # aloss_E += loss_E.detach()
            aloss_F += loss_F.detach()
            gNclose = model.KNclose.grad.norm().item()
            gE1 = model.KE1.grad.norm().item()
            gE2 = model.KE2.grad.norm().item()
            gN1 = model.KN1.grad.norm().item()
            gN2 = model.KN2.grad.norm().item()

            Nclose = model.KNclose.norm().item()
            E1 = model.KE1.norm().item()
            E2 = model.KE2.norm().item()
            N1 = model.KN1.norm().item()
            N2 = model.KN2.norm().item()

            optimizer.step()
            if (i + 1) % nprnt == 0 or (epoch + 1) % nprnt ==0:
                aloss /= nprnt2
                aloss_E /= nprnt2
                aloss_F /= nprnt2
                MAE /= nprnt2
                Fps /= nprnt2
                Fts /= nprnt2
                print(f'{epoch:2d}.{i:4d}  Loss: {aloss:.2f}  Loss_E: {aloss_E:.2f}  Loss_F: {aloss_F:.2f}  MAE: {MAE:.2f}  |F_pred|: {Fps:.4f}  |F_true|: {Fts:.4f}  Nclose: {gNclose/Nclose:2.4f}  E1: {gE1/E1:2.4f}  E2: {gE2/E2:2.4f}  N1: {gN1/N1:2.4f}  N2: {gN2/N2:2.4f}')
                # print(f'{epoch:2d}.{i:4d}  Loss: {aloss:.2f}  Loss_E: {aloss_E:.2f} Loss_F: {aloss_F:.2f}  MAE: {MAE:.2f}  |F_pred|: {Fps:.4f}  |F_true|: {Fts:.4f} Time taken: {time.time()-t0:.2f}s')
                aloss = 0.0
                aloss_E = 0.0
                aloss_F = 0.0
                Fps = 0.0
                Fts = 0.0
                MAE = 0.0

        if aloss < alossBest:
            alossBest = aloss
            bestModel = model

