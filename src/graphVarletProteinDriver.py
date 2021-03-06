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

nNodes = Ds.shape[0]
W = torch.diag(torch.ones(nNodes - 1), 1) + torch.diag(torch.ones(nNodes - 1), -1) + \
    torch.diag(torch.ones(nNodes - 2), 2) + torch.diag(torch.ones(nNodes - 2), -2) + \
    torch.diag(torch.ones(nNodes - 3), 3) + torch.diag(torch.ones(nNodes - 3), -3)

W = W.to(device)
G = GO.dense_graph(nNodes, W)


# Organize the node data
xn = nodeProperties.to(device)
xe = Ds.unsqueeze(0).unsqueeze(0).to(device) #edgeProperties

# Setup the network and its parameters
nNin    = 40
nEin    = 1
nopenN  = 512
nopenE  = 5
ncloseN = 3
ncloseE = 1
nlayer  = 18

model = GN.verletNetworks(nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.1)
model.to(device)



total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

xnOut, xeOut = model(xn,xe, G)
#xnOut = utils.distConstraint(xnOut,dc=3.79)

Dout  = utils.getDistMat(xnOut)
Dtrue = utils.getDistMat(Coords)

M   = torch.ger(M.squeeze(),M.squeeze())
err = F.mse_loss(M*Dout, M*Dtrue)
err0 = F.mse_loss(M*Dout, M*Dtrue)/F.mse_loss(M*Dout*0, M*Dtrue)
print('Initial Error pretraining = ',err.item(),err0.item())


#### Start Training ####

lrO = 1e-4
lrC = 1e-4
lrN = 1e-4
lrE = 1e-4

optimizer = optim.Adam([{'params': model.KNopen, 'lr': lrO},
                        {'params': model.KNclose, 'lr': lrC},
                        {'params': model.KEopen, 'lr': lrO},
                        {'params': model.KEclose, 'lr': lrC},
                        {'params': model.KE, 'lr': lrN},
                        {'params': model.KN, 'lr': lrE}])


alossBest = 1e6
epochs    = 500

ndata = 2 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)

        nNodes = Ds.shape[0]
        G = GO.dense_graph(nNodes, Ds)
        # Organize the node data
        xn = nodeProperties
        xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties

        M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        xnOut, xeOut = model(xn, xe, G)
        # xnOut = utils.distConstraint(xnOut, dc=3.79)
        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)

        loss = F.mse_loss(M*Dout, M*Dtrue)
        loss.backward()

        aloss   += loss.detach()
        alossAQ += (torch.norm(M*Dout - M*Dtrue)/torch.sum(M>0)).detach()
        gN = model.KN.grad.norm().item()
        gE = model.KE.grad.norm().item()
        gO = model.KNopen.grad.norm().item()
        gC = model.KNclose.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 100
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, alossAQ, gO, gN, gE, gC))
            aloss = 0.0

        # Validation
        if True:
            with torch.no_grad():
                misVal  = 0
                AQdis   = 0
                nVal = len(STesting)
                for jj in range(nVal):
                    nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                                        MSK, 0, device=device)

                    nNodes = Ds.shape[0]
                    G = GO.dense_graph(nNodes, Ds)
                    # Organize the node data
                    xn = nodeProperties
                    xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G)

                    Dout = utils.getDistMat(xnOut)
                    Dtrue = utils.getDistMat(Coords)
                    loss = F.mse_loss(M * Dout, M * Dtrue)

                    AQdis  += (torch.norm(M * Dout - M * Dtrue) / torch.sqrt(torch.sum(M>0))).detach()
                    misVal += loss.detach()



                print("%2d       %10.3E   %10.3E" % (j, misVal/nVal, AQdis/nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

