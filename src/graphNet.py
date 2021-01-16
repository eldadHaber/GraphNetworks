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

        KNopen = nn.Parameter(torch.randn(nopenN, nNIn)*1e-1)
        KEopen = nn.Parameter(torch.randn(nopenE, nEIn)*1e-1)
        KNclose = nn.Parameter(torch.randn(ncloseN,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.randn(ncloseE,nopenE)*1e-1)

        KN = nn.Parameter(torch.randn(nlayer,nopenE, nopenN)*1e-1)
        KE = nn.Parameter(torch.randn(nlayer,nopenN, nopenE)*1e-1)

        return KNopen, KEopen, KNclose, KEclose, KN, KE

    def forward(self, xn, xe):
        # xn - node feature
        # xe - edge features
        XX = []
        nL = self.KN.shape[0]

        xn = self.KNopen@xn
        xe = self.KEopen@xe

        XX.append(self.KNclose@xn)

        for i in range(nL):
            Kni = self.KN[i]
            Kei = self.KE[i]

            Coords = self.KNclose@xn
            R      = self.Graph.nodeGrad(Coords)
            dsq    = torch.sum(R**2,dim=1, keepdim=True)
            #sigma  = 10*dsq.mean()**2
            #print(sigma)
            w      = torch.exp(-dsq/0.1) #1/(dsq+1e-3) #
            Ai = Kni@self.Graph.nodeGrad(xn,w)
            Ai = F.instance_norm(Ai) #tv_norm(Ai)
            xe = xe + self.h*torch.relu(Ai)
            Bi = Kei@self.Graph.edgeDiv(xe, w)
            #Bi = Kei @ self.Graph.edgeAve(xe, w)

            Bi = F.instance_norm(Bi)
            xn = xn + self.h*torch.relu(Bi)
            XX.append(self.KNclose@xn)
        xn = self.KNclose @ xn
        xe = self.KEclose @ xe
        return xn, xe, XX


class diffusionNetworks(nn.Module):
    def __init__(self, G,nNIn, nEin,nopenN, nopenE, ncloseN, ncloseE,nlayer,h=0.1):
        super(diffusionNetworks, self).__init__()
        self.Graph  = G
        KNopen, KEopen, KNclose, KEclose, KN = \
            self.init_weights(nNIn,nEin,nopenN,nopenE,ncloseN, ncloseE,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.h       = h

    def init_weights(self,nNIn,nEIn,nopenN,nopenE,ncloseN, ncloseE,nlayer):

        KNopen = nn.Parameter(torch.randn(nopenN, nNIn)*1e-1)
        KEopen = nn.Parameter(torch.randn(nopenE, nEIn)*1e-1)
        KNclose = nn.Parameter(torch.randn(ncloseN,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.randn(ncloseE,nopenE)*1e-1)

        KN = nn.Parameter(torch.randn(nlayer,nopenE, nopenN)*1e-1)

        return KNopen, KEopen, KNclose, KEclose, KN

    def forward(self, xn, xe):
        # xn - node feature
        # xe - edge features
        XX = []
        nL = self.KN.shape[0]

        xn = self.KNopen@xn
        xe = self.KEopen@xe

        XX.append(self.KNclose@xn)

        for i in range(nL):
            Kni = self.KN[i]

            Coords = self.KNclose@xn
            R      = self.Graph.nodeGrad(Coords)
            dsq    = torch.sum(R**2,dim=1, keepdim=True)

            # Compute the "flux"
            #w      = 1/(dsq+1e-3) #torch.exp(-dsq/sigma)
            w = torch.exp(-dsq/0.1)

            Ai = Kni@self.Graph.nodeGrad(xn,w)
            Ai = F.instance_norm(Ai) #tv_norm(Ai)
            xe = torch.relu(Ai)

            #Bi = Kei@self.Graph.edgeDiv(xe, w)
            Bi = Kni.t()@self.Graph.edgeDiv(xe, w)

            xn = xn - self.h*Bi
            XX.append(self.KNclose@xn)
        xn = self.KNclose @ xn
        xe = self.KEclose @ xe
        return xn, xe, XX

test = True
if test:

    nNodes = 100
    nEdges = 100
    I = torch.tensor(np.arange(99),dtype=torch.long)
    I = torch.cat((I, 99+torch.zeros(1)))
    J = torch.tensor(np.arange(99)+1,dtype=torch.long)
    J = torch.cat((J, torch.zeros(1)))

    G = GO.graph(I,J,nNodes)

    nNin  = 20
    nEin  = 30
    nopenN = 512
    nopenE = 1024
    ncloseN = 3
    ncloseE = 1
    nlayer  = 50

    #model = varletNetworks(G,nNin, nEin,nopenN, nopenE, ncloseN,ncloseE,nlayer,h=.1)
    model = diffusionNetworks(G, nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.01)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ',total_params)

    xn = torch.zeros(1,20,nNodes)
    xe = torch.zeros(1,30,nEdges)
    #xn[0,10,:] = torch.linspace(0,1,100)
    xn[0, 0,  50] = 1
    xn[0, 19, 50] = -1

    xnOut, xeOut, XX = model(xn,xe)
    for i in range(model.KN.shape[0]):
        plt.figure(i+1)
        xx = XX[i]
        plt.plot(xx[0, 0, :].detach())
        plt.plot(xx[0, 1, :].detach())
        plt.plot(xx[0, 2, :].detach())
