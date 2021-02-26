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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load training data
Aind = torch.load('../data/casp11/AminoAcidIdx.pt')
Yobs = torch.load('../data/casp11/RCalpha.pt')
MSK  = torch.load('../data/casp11/Masks.pt')
S     = torch.load('../data/casp11/PSSM.pt')
# load validation data
AindVal = torch.load('../data/casp11/AminoAcidIdxVal.pt')
YobsVal = torch.load('../data/casp11/RCalphaVal.pt')
MSKVal  = torch.load('../data/casp11/MasksVal.pt')
SVal     = torch.load('../data/casp11/PSSMVal.pt')

# load Testing data
AindTesting = torch.load('../data/casp11/AminoAcidIdxTesting.pt')
YobsTesting = torch.load('../data/casp11/RCalphaTesting.pt')
MSKTesting  = torch.load('../data/casp11/MasksTesting.pt')
STesting     = torch.load('../data/casp11/PSSMTesting.pt')



print('Number of data: ', len(S))
n_data_total = len(S)



# Setup the network and its parameters
nNin = 40
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

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)
        if nodeProperties.shape[2] > 700:
            continue
        nNodes = Ds.shape[0]
        # G = GO.dense_graph(nNodes, Ds)
        w = Ds[I, J]
        G = GO.graph(I, J, nNodes, w)
        # Organize the node data
        xn = nodeProperties
        # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xe = w.unsqueeze(0).unsqueeze(0)

        M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        ## Profiler:
        # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        #     with profiler.record_function("model_inference"):
        #         xnOut, xeOut = model(xn, xe, G)
        # print(prof.key_averages())

        xnOut, xeOut = model(xn, xe, G)
        # xnOut = utils.distConstraint(xnOut, dc=3.79)
        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)

        loss = F.mse_loss(M * Dout, M * Dtrue)
        loss.backward()

        aloss += loss.detach()
        alossAQ += (torch.norm(M * Dout - M * Dtrue) / torch.sqrt(torch.sum(M)).detach())
        gN = model.KNclose.grad.norm().item()
        gE1 = model.KE1.grad.norm().item()
        gE2 = model.KE2.grad.norm().item()
        gO = model.KN1.grad.norm().item()
        gC = model.KN2.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 100
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC), flush=True)
            aloss = 0.0
            alossAQ = 0.0
        # Validation
        nextval = 300
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                nVal = len(STesting)
                for jj in range(nVal):
                    nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(STesting, AindTesting, YobsTesting,
                                                                                        MSKTesting, jj, device=device)

                    nNodes = Ds.shape[0]
                    # G = GO.dense_graph(nNodes, Ds)
                    w = Ds[I, J]
                    G = GO.graph(I, J, nNodes, w)
                    # Organize the node data
                    xn = nodeProperties
                    # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xe = w.unsqueeze(0).unsqueeze(0)

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G)

                    Dout = utils.getDistMat(xnOut)
                    Dtrue = utils.getDistMat(Coords)
                    loss = F.mse_loss(M * Dout, M * Dtrue)

                    AQdis += (torch.norm(M * Dout - M * Dtrue) / torch.sqrt(torch.sum(M))).detach()
                    misVal += loss.detach()

                print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model
