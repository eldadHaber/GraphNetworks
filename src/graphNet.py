import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
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

def doubleLayer(x, K1, K2):
    x = F.conv1d(x, K1.unsqueeze(-1))
    x = F.layer_norm(x, x.shape)
    x = torch.relu(x)
    x = F.conv1d(x, K2.unsqueeze(-1))
    return x


class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=True):
        super(graphNetwork, self).__init__()

        self.h       = h
        stdv  = 1e-2
        stdvp = 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin)*stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        if dense:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin, 9, 9) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen, 9, 9) * stdv)
        else:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose,nopen)*stdv)

        Nfeatures = 3*nopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures, 9, 9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid, 9, 9) * stdvp)
        else:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.KN1  = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures)*stdvp)
        self.KN2  = nn.Parameter(torch.rand(nlayer,nopen, nhid)*stdvp)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def doubleLayer(self,x, K1, K2):
        x = self.edgeConv(x, K1)
        x = F.layer_norm(x, x.shape)
        x = torch.relu(x)
        x = self.edgeConv(x, K2)
        return x

    def forward(self,xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn,self.K1Nopen,self.K2Nopen)
        xe = self.doubleLayer(xe,self.K1Eopen,self.K2Eopen)

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):


            #gradX = torch.exp(-torch.abs(Graph.nodeGrad(xn)))
            gradX = Graph.nodeGrad(xn)
            intX  = Graph.nodeAve(xn)
            xec   = torch.cat([intX, xe, gradX], dim=1)
            xec   = self.doubleLayer(xec, self.KE1[i], self.KE2[i])

            xec = F.layer_norm(xec, xec.shape)
            xec = torch.relu(xec)

            #divE = torch.exp(-torch.abs(Graph.edgeDiv(xec)))
            divE = Graph.edgeDiv(xec)
            aveE = Graph.edgeAve(xec, method='ave')
            xnc  = torch.cat([aveE,divE,xn], dim=1)
            xnc  = self.doubleLayer(xnc, self.KN1[i], self.KN2[i])

            xn = xn + self.h * xnc
            xe = xe + self.h * xec

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))


        return xn, xe


#######
#
#  Varlet Network
#
######

class verletNetworks(nn.Module):
    def __init__(self, nNin, nEin, nopen, nhid, nNout, nEout, nlayer,h=0.1):
        super(verletNetworks, self).__init__()

        stdv = 1e-3
        self.KNopen = nn.Parameter(torch.randn(nopen, nNin)*stdv)
        self.KEopen = nn.Parameter(torch.randn(nopen, nEin)*stdv)

        self.KE1 = nn.Parameter(torch.randn(nlayer, nhid, 2*nopen)*stdv)
        self.KE2 = nn.Parameter(torch.randn(nlayer, nopen, nhid)*stdv)

        self.KN1 = nn.Parameter(torch.randn(nlayer, nhid, 2*nopen)*stdv)
        self.KN2 = nn.Parameter(torch.randn(nlayer, nopen, nhid)*stdv)

        self.KNclose = nn.Parameter(torch.randn(nNout, nopen)*stdv)
        self.KEclose = nn.Parameter(torch.randn(nEout, nopen)*stdv)

        self.h       = h


    def forward(self, xn, xe, Graph):
        # xn - node feature
        # xe - edge features

        nL = self.KN1.shape[0]

        xn = F.conv1d(xn, self.KNopen.unsqueeze(-1))
        xe = F.conv1d(xe, self.KEopen.unsqueeze(-1))

        for i in range(nL):
            Kni1 = self.KN1[i]
            Kni2 = self.KN2[i]
            Kei1 = self.KE1[i]
            Kei2 = self.KE2[i]

            # Move to edges and compute flux
            AiG = Graph.nodeGrad(xn)
            AiA = Graph.nodeAve(xn)
            Ai  = torch.cat((AiG, AiA), dim=1)
            Ai  = doubleLayer(Ai, Kei1, Kei2)
            xe = xe + self.h*Ai #torch.relu(Ai)

            # Edge to node
            BiD = Graph.edgeDiv(xe)
            BiA = Graph.edgeAve(xe, method='ave')
            Bi  = torch.cat((BiD, BiA), dim=1)
            Bi = doubleLayer(Bi, Kni1, Kni2)

            # Update
            xn = xn + self.h*Bi

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xe = F.conv1d(xe, self.KEclose.unsqueeze(-1))

        return xn, xe



Test = False
if Test:
    nNin = 20
    nEin = 3
    nNopen = 32
    nEopen = 16
    nEhid  = 128
    nNclose = 3
    nEclose = 2
    nlayer  = 18
    model = graphNetwork(nNin, nEin, nNopen, nEopen, nEhid, nNclose, nEclose,nlayer)

    L = 55
    xn = torch.zeros(1,nNin,L)
    xn[0,:,23] = 1
    xe = torch.ones(1,nEin,L,L)

    G = GO.dense_graph(L)

    xnout, xeout = model(xn,xe,G)


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
