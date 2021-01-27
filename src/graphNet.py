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

class verletNetworks(nn.Module):
    def __init__(self, nNIn, nEin,nopenN, nopenE, ncloseN, ncloseE,nlayer,h=0.1):
        super(verletNetworks, self).__init__()
        KNopen, KEopen, KNclose, KEclose, KN, KE, KD = \
            self.init_weights(nNIn,nEin,nopenN,nopenE,ncloseN, ncloseE,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.KE      = KE
        self.KD      = KD

        self.h       = h

    def init_weights(self,nNIn,nEIn,nopenN,nopenE,ncloseN, ncloseE,nlayer):

        KNopen = nn.Parameter(torch.randn(nopenN, nNIn)*1e-1)
        KEopen = nn.Parameter(torch.randn(nopenE, nEIn)*1e-1)
        KNclose = nn.Parameter(torch.randn(ncloseN,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.randn(ncloseE,nopenE)*1e-1)

        KN = nn.Parameter(torch.rand(nlayer,nopenE, nopenN)*1e-1)
        KE = nn.Parameter(torch.rand(nlayer,nopenN, nopenE)*1e-1)
        KD = nn.Parameter(torch.rand(nlayer, nopenN, nopenN) * 1e-1)

        return KNopen, KEopen, KNclose, KEclose, KN, KE, KD

    def forward(self, xn, xe, Graph):
        # xn - node feature
        # xe - edge features
        # XX = []; XE = []
        nL = self.KN.shape[0]

        xn = self.KNopen@xn
        xe = self.KEopen@xe

        #XX.append(self.KNclose@xn)
        #XE.append(xe)
        for i in range(nL):
            Kni = self.KN[i]
            Kei = self.KE[i]
            Kdi = self.KD[i]

            #Coords = self.KNclose@xn
            #R      = Graph.nodeGrad(Coords)
            #dsq    = torch.sum(R**2,dim=1, keepdim=True)

            w      = 1 #torch.exp(-dsq/0.1) #1/(dsq+1e-3) #
            # Move to edges and compute flux
            Ai = Kni@Graph.nodeGrad(xn,w)
            Ai = tv_norm(Ai)
            xe = xe + self.h*torch.relu(Ai)
            #XE.append(xe)

            # Edge to node
            Bi = Kei@Graph.edgeDiv(xe, w)
            Bi = tv_norm(Bi)
            Bi = torch.relu(Bi)

            # diffusion term
            Ci = Kdi@xn
            Ci = tv_norm(Ci)
            Ci = torch.relu(Ci)

            # Update
            xn = xn - self.h*(Bi+Ci)

            #XX.append(self.KNclose@xn)
        xn = self.KNclose @ xn
        xe = self.KEclose @ xe
        return xn, xe  #, XX, XE

    def backProp(self, xn, xe, Graph):
        # xn - node feature
        # xe - edge features
        # XX = []; XE = []
        nL = self.KN.shape[0]

        xn = self.KNclose.t()@xn
        xe = self.KEclose.t()@xe

        #XX.append(self.KNclose@xn)
        #XE.append(xe)
        for i in reversed(range(nL)):
            Kni = self.KN[i]
            Kei = self.KE[i]
            Kdi = self.KD[i]

            Coords = self.KNclose@xn
            #R      = Graph.nodeGrad(Coords)
            #dsq    = torch.sum(R**2,dim=1, keepdim=True)

            w      = 1 #torch.exp(-dsq/0.1) #1/(dsq+1e-3) #
            # Move to edges and compute flux
            Ai = Kni@Graph.nodeGrad(xn,w)
            Ai = tv_norm(Ai)
            xe = xe - self.h*torch.relu(Ai)
            #XE.append(xe)

            # Edge to node
            Bi = Kei@self.Graph.edgeDiv(xe, w)
            Bi = tv_norm(Bi)
            Bi = torch.relu(Bi)

            # diffusion term
            Ci = Kdi@xn
            Ci = tv_norm(Ci)
            Ci = torch.relu(Ci)

            # Update
            xn = xn + self.h*(Bi+Ci)

            #XX.append(self.KNclose@xn)
        xn = self.KNopen.t()@xn
        xe = self.KEopen.t()@xe
        return xn, xe #, XX, XE


class diffusionNetworks(nn.Module):
    def __init__(self, nNIn, nEin,nopenN, nopenE, ncloseN, ncloseE,nlayer,h=0.1):
        super(diffusionNetworks, self).__init__()
        KNopen, KEopen, KNclose, KEclose, KN, KE, KR = \
            self.init_weights(nNIn,nEin,nopenN,nopenE,ncloseN, ncloseE,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.KE      = KE
        self.KR      = KR

        self.h       = h

    def init_weights(self,nNIn,nEIn,nopenN,nopenE,ncloseN, ncloseE,nlayer):

        KNopen = nn.Parameter(torch.randn(nopenN, nNIn)*1e-1)
        KEopen = nn.Parameter(torch.randn(nopenE, nEIn)*1e-1)
        KNclose = nn.Parameter(torch.randn(ncloseN,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.randn(ncloseE,nopenE)*1e-1)

        KN = nn.Parameter(torch.rand(nlayer,nopenE, nopenN)*1e-1)
        KE = nn.Parameter(torch.rand(nlayer,nopenN, nopenN)*1e-1)
        KR = nn.Parameter(torch.rand(nlayer, nopenN, nopenN)*1e-1)

        return KNopen, KEopen, KNclose, KEclose, KN, KE, KR

    def forward(self, xn, Graph):
        # xn - node feature
        # xe - edge features
        #XX = []
        nL = self.KN.shape[0]

        xn = self.KNopen@xn

        #XX.append(self.KNclose@xn)

        for i in range(nL):
            Kni = self.KN[i]
            Kei = self.KE[i]
            Kri = self.KR[i]

            Coords = self.KNclose@xn
            #R      = Graph.nodeGrad(Coords)
            #dsq    = torch.sum(R**2,dim=1, keepdim=True)

            # Compute the "flux"
            w      = 1 #1/(dsq+1e-3) #torch.exp(-dsq/sigma)
            #w = torch.exp(-dsq/10)

            gradX  = Graph.nodeGrad(xn,w)
            # Diffusion
            Ai     = Kni@gradX
            Ai     = tv_norm(Ai)
            xeDiff = torch.relu(Ai)
            Jdiff = Kni.t()@self.Graph.edgeDiv(xeDiff, w)
            # Advection
            Ci    = Kei@gradX
            Ci    = tv_norm(Ci)
            xeAdv = torch.relu(Ci)
            Jadv  = Graph.edgeAve(xeAdv)
            # reaction
            Ri = Kri@xn
            Ri = tv_norm(Ri)
            ri = torch.relu(Ri)
            # Apdate
            xn = xn - self.h*(Jadv + Jdiff + ri)

            #XX.append(self.KNclose@xn)

        xn = self.KNclose@xn

        return xn #, XX

test = False
if test:

    nNodes = 100
    nEdges = 99
    I = torch.tensor(np.arange(nEdges),dtype=torch.long)
    #I = torch.cat((I, 99+torch.zeros(1)))
    J = torch.tensor(np.arange(nEdges)+1,dtype=torch.long)
    #J = torch.cat((J, torch.zeros(1)))

    G = GO.graph(I,J,nNodes)

    nNin  = 20
    nEin  = 30
    nopenN = 64
    nopenE = 128
    ncloseN = 3
    ncloseE = 1
    nlayer  = 50

    mm = 'adv'
    if mm == 'diff':
        model = diffusionNetworks(nNin, nEin, nopenN, nopenE, ncloseN, ncloseE, nlayer, h=.05)
    elif mm == 'adv':
        model = verletNetworks(nNin, nEin,nopenN, nopenE, ncloseN,ncloseE,nlayer,h=.1)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ',total_params)

    xn = torch.zeros(1,20,nNodes)
    xe = torch.zeros(1,30,nEdges)
    xn[0, 1, 40] = 1
    xn[0, 1, 60] = 1
    if mm=='diff':
        xnOut, XX = model(xn, G)
    elif mm=="adv":
        xnOut, xeOut, XX, XE = model(xn,xe, G)

    kk = 1
    for i in range(model.KN.shape[0]):
        if i%5 == 0:
            print(i)
            plt.figure(kk)
            xx = XX[i]
            xx = xx.detach().squeeze(0)
            plt.plot(xx.sum(dim=0))
            kk += 1
        #plt.plot(xx[0, 1, :].detach())
        #plt.plot(xx[0, 2, :].detach())
