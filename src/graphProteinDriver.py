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

PSSM, Coords, M, IJ, Ds = prc.getIterData(S, Aind, Yobs, MSK, 11, device=device)

plt.figure(1)
plt.imshow(Ds)

nNodes = Ds.shape[0]
I = IJ[:,0]
J = IJ[:,1]
G = GO.graph(I, J, nNodes)

# Organize the edge data
nEdges = I.shape[0]
xe = torch.zeros(1,1,nEdges)
for i in range(nEdges):
    if I[i]+1 == J[i]:
        xe[:,:,i] = 1
    if I[i]-1 == J[i]:
        xe[:,:,i] = 1

# Organize the node data
xn = PSSM.unsqueeze(0)

# Setup the network and its parameters
nNin = 40
nEin = 1
nopenN = 256
nopenE = 256
ncloseN = 3
ncloseE = 1
nlayer = 18

#model = GN.verletNetworks(G, nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.1)
model = GN.diffusionNetworks(G, nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.05)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#xnOut, xeOut, XX, XE = model(xn,xe)
xnOut, XX = model(xn)

xnOut = utils.distConstraint(xnOut,dc=0.379)

Dout  = utils.getDistMat(xnOut)
Dtrue = utils.getDistMat(Coords.unsqueeze(0))
plt.figure(2)
plt.imshow(Dout.detach())
plt.figure(3)
plt.imshow(Dtrue.detach())

err1 = torch.norm(M*(Dout.detach()-Dtrue))/torch.norm(M*Dtrue)
err2 = (M*(Dout.detach()-Dtrue)).abs().mean().item()/M.sum()
print('Initial Error = ',err1, err2 )