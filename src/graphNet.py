import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim

from src import graphOps as GO

def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X,dim=1, keepdim=True)
    X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True) + eps)
    return X

class verletNetworks(nn.Module):
    def __init__(self, nNIn, nEin,nopenN, nopenE, ncloseN, ncloseE,nlayer,h=0.1):
        super(verletNetworks, self).__init__()
        KNopen, KEopen, KNclose, KEclose, KN, KE, KNa, KEa = \
            self.init_weights(nNIn,nEin,nopenN,nopenE,ncloseN, ncloseE,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.KE      = KE
        self.KNa     = KNa
        self.KEa     = KEa

        self.h       = h

    def init_weights(self,nNIn,nEIn,nopenN,nopenE,ncloseN, ncloseE,nlayer):

        stdv  = 1e-2
        stdvp = 1e-3
        KNopen = nn.Parameter(torch.randn(nopenN, nNIn)*stdv)
        #KEopen = nn.Parameter(torch.randn(nopenE, nEIn)*stdv)
        KEopen = nn.Parameter(torch.randn(2*nopenE, nEIn) * stdv)

        KNclose = nn.Parameter(torch.randn(ncloseN,nopenN)*stdv)
        #KEclose = nn.Parameter(torch.randn(ncloseE,nopenE)*stdv)
        KEclose = nn.Parameter(torch.randn(ncloseE, 2*nopenE) * stdv)

        KN  = nn.Parameter(torch.rand(nlayer,nopenE, nopenN)*stdvp)
        #KE  = nn.Parameter(torch.rand(nlayer,nopenN, nopenE)*stdvp)
        KE = nn.Parameter(torch.rand(nlayer, nopenN, 2*nopenE) * stdvp)

        KNa = nn.Parameter(torch.rand(nlayer, nopenE, nopenN)*stdvp)
        #KEa = nn.Parameter(torch.rand(nlayer, nopenN, nopenE)*stdvp)
        KEa = nn.Parameter(torch.rand(nlayer, nopenN, 2*nopenE) * stdvp)

        return KNopen, KEopen, KNclose, KEclose, KN, KE, KNa, KEa

    def forward(self, xn, xe, Graph):
        # xn - node feature
        # xe - edge features

        nL = self.KN.shape[0]

        xn = self.KNopen@xn
        xe = conv2(xe,self.KEopen.unsqueeze(2).unsqueeze(2))
        # xe = self.KEopen @ xe

        for i in range(nL):
            Kni = self.KN[i]
            Kna = self.KNa[i]
            Kei = self.KE[i]
            Kea = self.KEa[i]

            #w      = True #torch.ones(1) #torch.exp(-dsq/0.1) #1/(dsq+1e-3) #
            # Move to edges and compute flux
            AiG = Graph.nodeGrad(Kni@xn)
            AiA = Graph.nodeAve(Kna@xn)
            Ai  = torch.cat((AiG, AiA), dim=1)
            Ai  = tv_norm(Ai)
            xe = xe + self.h*torch.relu(Ai)

            # Edge to node
            BiD = Kei@Graph.edgeDiv(xe)
            BiA = Kea@Graph.edgeAve(xe,method='mean')
            Bi = BiD + BiA

            # Update
            xn = xn + self.h*Bi

        xn = self.KNclose@xn
        xe = conv2(xe, self.KEclose.unsqueeze(2).unsqueeze(2))
        #xe = self.KEclose @ xe
        return xn, xe

    def backProp(self, xn, xe, Graph):
        # xn - node feature
        # xe - edge features

        nL = self.KN.shape[0]

        xn = self.KNclose.t()@xn
        #xe = self.KEclose.t()@xe
        xe = conv2(xe, self.KEclose.t().unsqueeze(2).unsqueeze(2))

        for i in reversed(range(nL)):
            Kni = self.KN[i]
            Kei = self.KE[i]

            w      = torch.ones(1)
            # Move to edges and compute flux
            Ai = Graph.nodeGrad(Kni@xn,w)
            Ai = tv_norm(Ai)
            xe = xe - self.h*torch.relu(Ai)

            # Edge to node
            Bi = Kei@self.Graph.edgeDiv(xe, w)
            Bi = tv_norm(Bi)
            Bi = torch.relu(Bi)

            # Update
            xn = xn + self.h*Bi

            #XX.append(self.KNclose@xn)
        xn = self.KNopen.t()@xn
        xe = self.KEopen.t()@xe
        return xn, xe #, XX, XE



test = False
if test:

    nNodes = 32
    #nEdges = 99
    #I = torch.tensor(np.arange(nEdges),dtype=torch.long)
    #I = torch.cat((I, 99+torch.zeros(1)))
    #J = torch.tensor(np.arange(nEdges)+1,dtype=torch.long)
    #J = torch.cat((J, torch.zeros(1)))

    W = torch.diag(torch.ones(nNodes - 1), 1) + torch.diag(torch.ones(nNodes - 1), -1) + \
        torch.diag(torch.ones(nNodes - 2), 2) + torch.diag(torch.ones(nNodes - 2), -2) + \
        torch.diag(torch.ones(nNodes - 3), 3) + torch.diag(torch.ones(nNodes - 3), -3)

    G = GO.dense_graph(nNodes,W)

    nNin  = 40
    nEin  = 30
    nopenN = 64
    nopenE = 128
    ncloseN = 3
    ncloseE = 1
    nlayer  = 18

    model = verletNetworks(nNin, nEin,nopenN, nopenE, ncloseN,ncloseE,nlayer,h=0.10)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ',total_params)

    xn = torch.zeros(1,nNin,nNodes)
    xe = torch.zeros(1,nEin,nNodes,nNodes)
    xn[0, 1, 16] = 1
    #xe[0, 1, ] = 1

    xnOut, xeOut = model(xn,xe, G)
