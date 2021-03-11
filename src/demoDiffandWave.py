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

# Setup the network and its parameters
nNin = 1
nEin = 1
nNopen = 4
nEhid = 4
nNclose = 1
nlayer = 24

modelV = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.05, dense=False,varlet=True)
modelD = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.05, dense=False,varlet=False)

modelD.to(device)
modelV.to(device)

total_params = sum(p.numel() for p in modelD.parameters())
print('Number of parameters ', total_params)

# Get a graph
n    = 32
X, Y = torch.meshgrid(torch.arange(n),torch.arange(n))
XX = torch.cat([X.reshape(n**2,1),Y.reshape(n**2,1)],dim=1)
D  = torch.sum(XX**2,dim=1,keepdim=True) + torch.sum(XX**2,dim=1,keepdim=True).t() - 2*XX@XX.t()
D[D>1] = 0
D = D + torch.eye(n**2)
IJ = torch.nonzero(D)
ne = IJ.shape[0]
# Get the graph
G = GO.graph(IJ[:,0], IJ[:,1], n**2)

xn = torch.zeros(n,n)
xn[16,16] = 1.0
xn = xn.reshape(1,1,n**2)
xe = torch.zeros(1,1,ne)

xnoutD, _ = modelD(xn, xe, G)
xnoutD = xnoutD.reshape(32,32)

xnoutV, _ = modelV(xn, xe, G)
xnoutV = xnoutV.reshape(32,32)

plt.subplot(1,2,1)
plt.imshow(torch.tanh(xnoutD).detach())
plt.colorbar()
plt.title('Diffusion')
plt.subplot(1,2,2)
plt.imshow(torch.tanh(xnoutV).detach())
plt.colorbar()
plt.title('Varlet')


