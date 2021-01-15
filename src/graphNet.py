import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim

from src import graphOps as GO

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X,dim=1, keepdim=True)
    X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True) + eps)
    return X



class varletNetworks(nn.Module):
    def __init__(self, G,nNIn, nEin,nopenN, nopenE, ncloseN, ncloseE,nlayer,h=0.1):
        super(varletNetworks, self).__init__()
        self.Graph  = G
        KNopen, KEopen, KNclose, KEclose, KN, KE = \
            self.init_weights(nNIn,nEin,nopenN,nopenE,ncloseN, ncloseE,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.KE      = KE
        self.h       = h

    def init_weights(self,nNIn,nEIn,nopenN,nopenE,ncloseN, ncloseE,nlayer):

        KNopen = nn.Parameter(torch.rand(nopenN, nNIn)*1e-1)
        KEopen = nn.Parameter(torch.rand(nopenE, nEIn)*1e-1)
        KNclose = nn.Parameter(torch.rand(ncloseN,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.rand(ncloseE,nopenE)*1e-1)

        KN = nn.Parameter(torch.randn(nlayer,nopenE, nopenN)*1e-1)
        KE = nn.Parameter(torch.randn(nlayer,nopenN, nopenE)*1e-1)

        return KNopen, KEopen, KNclose, KEclose, KN, KE

    def forward(self, xn, xe):
        # xn - node feature
        # xe - edge features

        nL = self.KN.shape[0]

        xn = self.KNopen@xn
        xe = self.KEopen@xe
        for i in range(nL):
            Kni = self.KN[i]
            Kei = self.KE[i]

            #Coords = self.KNclose@xe
            Ai = Kni@self.Graph.nodeGrad(xn)
            Ai = tv_norm(Ai)
            xe = xe + self.h*torch.relu(Ai)
            Bi = Kei@self.Graph.edgeDiv(xe)
            Bi = tv_norm(Bi)
            xn = xn - self.h*torch.relu(Bi)

        xn = self.KNclose @ xn
        xe = self.KEclose @ xe
        return xn, xe



test = True
if test:

    nNodes = 100
    nEdges = 200
    I = torch.randint(0, nNodes, (nEdges,))
    J = torch.randint(0, nNodes, (nEdges,))
    G = GO.graph(I,J,100)

    nNin  = 20
    nEin  = 30
    nopenN = 256
    nopenE = 512
    ncloseN = 3
    ncloseE = 1
    nlayer  = 18

    model = varletNetworks(G,nNin, nEin,nopenN, nopenE, ncloseN,ncloseE,nlayer,h=1)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ',total_params)

    xn = torch.zeros(1,20,nNodes)
    xe = torch.zeros(1,30,nEdges)
    xn[0,1,50] = 1
    xnOut, xeOut = model(xn,xe)

    plt.plot(xnOut[0, 0, :].detach())
    plt.plot(xnOut[0, 1, :].detach())
    plt.plot(xnOut[0, 2, :].detach())