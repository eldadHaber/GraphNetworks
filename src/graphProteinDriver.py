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
nopen   = 5
nhid    = 10
nNclose = 3
nlayer  = 6

dense = True
model = GN.graphNetwork(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=dense)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)


#### Start Training ####

lrO = 1e-1
lrC = 1e-1
lrN = 1e-1
lrE1 = 1e-1
lrE2 = 1e-1


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
epochs    = 100

ndata = 100 #n_data_total
bestModel = model
hist = torch.zeros(epochs)
batchSize = 16

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
            if dense:
                G = GO.dense_graph(nNodes, Ds)
                xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
            else:
                w = Ds[IJ[:, 0], IJ[:, 1]]
                G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
                xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties

            xn = nodeProperties

            M = torch.ger(M.squeeze(), M.squeeze())
            optimizer.zero_grad()

            xnOut, xeOut = model(xn, xe, G)
            loss = utils.dRMSD(xnOut, Coords, M)

            loss.backward()
            #gN = model.KN1.grad.norm().item()
            #print('norm of the gradient', gN)
            optimizer.step()

            aloss   += loss.detach()
            alossAQ += torch.sqrt(loss)


            # scheduler.step()
            nprnt = 10
            if i%nprnt == 0:
                aloss = aloss / nprnt
                alossAQ = alossAQ/nprnt
                print("%2d.%1d   %10.3E   %10.3E" % (j, i, aloss, alossAQ))
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
                    if dense:
                        G = GO.dense_graph(nNodes, Ds)
                        xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    else:
                        w = Ds[IJ[:, 0], IJ[:, 1]]
                        G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
                        xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xn = nodeProperties

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G, Ds)

                    loss = utils.dRMSD(xnOut, Coords, M)
                    AQdis  += torch.sqrt(loss)
                    misVal += loss.detach()



                print("%2d       %10.3E   %10.3E" % (j, misVal/nVal, AQdis/nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

