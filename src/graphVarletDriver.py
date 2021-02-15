import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load training data
Aind = torch.load('../../../data/casp11/AminoAcidIdx.pt')
Yobs = torch.load('../../../data/casp11/RCalpha.pt')
MSK  = torch.load('../../../data/casp11/Masks.pt')
S     = torch.load('../../../data/casp11/PSSM.pt')
# load validation data
AindVal = torch.load('../../../data/casp11/AminoAcidIdxVal.pt')
YobsVal = torch.load('../../../data/casp11/RCalphaVal.pt')
MSKVal  = torch.load('../../../data/casp11/MasksVal.pt')
SVal     = torch.load('../../../data/casp11/PSSMVal.pt')

# load Testing data
AindTesting = torch.load('../../../data/casp11/AminoAcidIdxTesting.pt')
YobsTesting = torch.load('../../../data/casp11/RCalphaTesting.pt')
MSKTesting  = torch.load('../../../data/casp11/MasksTesting.pt')
STesting     = torch.load('../../../data/casp11/PSSMTesting.pt')


print('Number of data: ', len(S))
n_data_total = len(S)

nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs, MSK, 0, device=device)

# Setup the network and its parameters
nNin    = 40
nEin    = 1
nopen  = 128
nhid   = 192
nNout = 3
nEout = 1
nlayer = 6

model = GN.verletNetworks(nNin, nEin, nopen, nhid, nNout, nEout, nlayer, h=1)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)


#### Start Training ####

lrO = 1e-4
lrC = 1e-4
lrN = 1e-4
lrE1 = 1e-4
lrE2 = 1e-4


optimizer = optim.Adam([{'params': model.KNopen, 'lr': lrO},
                        {'params': model.KNclose, 'lr': lrC},
                        {'params': model.KEopen, 'lr': lrO},
                        {'params': model.KEclose, 'lr': lrC},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.KN1, 'lr': lrE2},
                        {'params': model.KN2, 'lr': lrN}])


alossBest = 1e6
epochs    = 100

ndata = 100 #n_data_total
bestModel = model
hist = torch.zeros(epochs)
batchSize = 32
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)

        nNodes = Ds.shape[0]
        if nNodes < 500:
            w = Ds[IJ[:, 0], IJ[:, 1]]
            G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)

            xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
            xn = nodeProperties

            M = torch.ger(M.squeeze(), M.squeeze())
            if i % batchSize == 0:
                optimizer.zero_grad()

            xnOut, xeOut = model(xn, xe, G)
            loss = utils.dRMSD(xnOut, Coords, M)
            loss.backward()

            aloss   += loss.detach()
            alossAQ += torch.sqrt(loss)
            if i % batchSize == 0:
                gN  = model.KN1.grad.norm().item()
                gE1 = model.KE1.grad.norm().item()
                gE2 = model.KE2.grad.norm().item()
                gO = model.KNopen.grad.norm().item()
                gC = model.KNclose.grad.norm().item()

                optimizer.step()
            # scheduler.step()
            nprnt = batchSize
            if i % nprnt == 0:
                aloss = aloss / nprnt
                alossAQ = alossAQ/nprnt
                print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                      (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC))
                aloss = 0.0
                alossAQ = 0.0
        # Validation
        nextval = 1e9
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal  = 0
                AQdis   = 0
                nVal = len(STesting)
                for jj in range(nVal):
                    nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                                        MSK, 0, device=device)

                    nNodes = Ds.shape[0]
                    w = Ds[IJ[:, 0], IJ[:, 1]]
                    G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
                    xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xn = nodeProperties

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G)

                    loss = utils.dRMSD(xnOut, Coords, M)
                    AQdis  += torch.sqrt(loss)
                    misVal += loss.detach()



                print("%2d       %10.3E   %10.3E" % (j, misVal/nVal, AQdis/nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

