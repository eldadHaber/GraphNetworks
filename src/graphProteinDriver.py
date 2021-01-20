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

nNodes = Ds.shape[0]
I = IJ[:,0]
J = IJ[:,1]
G = GO.graph(I, J, nNodes)

# Organize the node data
xn = nodeProperties
xe = edgeProperties

# Setup the network and its parameters
nNin = 40
nEin = 1
nopenN = 256
nopenE = 256
ncloseN = 3
ncloseE = 1
nlayer = 50

model = GN.verletNetworks(nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.1)
#model = GN.diffusionNetworks(nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.05)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

xnOut, xeOut, XX, XE = model(xn,xe, G)
#xnOut, XX = model(xn)
xnOut = utils.distConstraint(xnOut,dc=0.379)

Dout  = utils.getDistMat(xnOut)
Dtrue = utils.getDistMat(Coords)
#plt.figure(1)
#plt.imshow(Ds)
#plt.figure(2)
#plt.imshow(Dout.detach())
#plt.figure(3)
#plt.imshow(Dtrue.detach())

M = torch.ger(M.squeeze(),M.squeeze())
err1 = torch.norm(M*(Dout.detach()-Dtrue))/torch.norm(M*Dtrue)
err2 = (M*(Dout.detach()-Dtrue)).abs().mean().item()/M.sum()

sig = 0.3
dm = Dtrue.max()
Dout = torch.exp(-Dout ** 2 / (dm * sig) ** 2)
Dtrue = torch.exp(-Dtrue ** 2 / (dm * sig) ** 2)

#plt.figure(4)
#plt.imshow(Dout.detach())
#plt.figure(5)
#plt.imshow(Dtrue.detach())

err3 = torch.norm(M*(Dout.detach()-Dtrue))/torch.norm(M*Dtrue)

print('Initial Error pretraining = ',err1, err2, err3 )

#### Start Training ####

lrO = 1e-4
lrC = 1e-4
lrN = 1e-3
lrE = 1e-3
lrD = 1e-3

optimizer = optim.Adam([{'params': model.KNopen, 'lr': lrO},
                        {'params': model.KNclose, 'lr': lrC},
                        {'params': model.KEopen, 'lr': lrO},
                        {'params': model.KEclose, 'lr': lrC},
                        {'params': model.KE, 'lr': lrN},
                        {'params': model.KN, 'lr': lrE},
                        {'params': model.KD, 'lr': lrD}])


alossBest = 1e6
epochs = 30
sig   = 0.3
ndata = 10 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)

        nNodes = Ds.shape[0]
        I = IJ[:, 0]
        J = IJ[:, 1]
        G = GO.graph(I, J, nNodes)
        # Organize the node data
        xn = nodeProperties
        xe = edgeProperties
        M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        xnOut, xeOut, _, _ = model(xn, xe, G)
        xnOut = utils.distConstraint(xnOut, dc=0.379)

        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)
        dm = Dtrue.max()
        Dout  = torch.exp(-Dout**2 / (dm * sig)**2)
        Dtrue = torch.exp(-Dtrue**2 / (dm * sig)**2)

        loss = torch.norm(M*(Dout-Dtrue))**2/torch.norm(M*Dtrue)**2
        loss.backward()

        aloss += loss.detach()
        gN = model.KN.grad.norm().item()
        gO = model.KNopen.grad.norm().item()
        gC = model.KNclose.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, gO, gN, gC))
            aloss = 0.0
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        misVal  = 0
        AQdis   = 0
        nVal = len(STesting)
        for jj in range(nVal):

            # Get the data
            nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(STesting, AindTesting, YobsTesting,
                                                                                MSKTesting, jj, device=device)
            # Organize the node data
            nNodes = Ds.shape[0]
            I = IJ[:, 0]
            J = IJ[:, 1]
            G = GO.graph(I, J, nNodes)

            xn = nodeProperties
            xe = edgeProperties
            M = torch.ger(M.squeeze(), M.squeeze())

            xnOut, xeOut, _, _ = model(xn, xe, G)
            xnOut = utils.distConstraint(xnOut, dc=0.379)

            Dout = utils.getDistMat(xnOut)
            Dtrue = utils.getDistMat(Coords)
            AQdis += torch.norm(M * (Dout - Dtrue)) / torch.sum(M > 0)

            dm = Dtrue.max()
            Dout = torch.exp(-Dout ** 2 / (dm * sig) ** 2)
            Dtrue = torch.exp(-Dtrue ** 2 / (dm * sig) ** 2)

            loss = torch.norm(M * (Dout.detach() - Dtrue)) ** 2 / torch.norm(M * Dtrue) ** 2

            misVal += loss.detach()



        print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
        print('===============================================')


