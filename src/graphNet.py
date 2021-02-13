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

    def __init__(self, nNin, nEin, nNopen, nEopen, nEhid, nNclose, nEclose,nlayer,h=0.1,dense=True):
        super(graphNetwork, self).__init__()

        self.h       = h
        stdv  = 1e-2
        stdvp = 1e-3
        self.KNopen = nn.Parameter(torch.randn(nNopen, nNin)*stdv)
        if dense:
            self.KEopen = nn.Parameter(torch.randn(nEopen, nEin, 9, 9) * stdv)
        else:
            self.KEopen = nn.Parameter(torch.randn(nEopen, nEin) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose,nNopen)*stdv)
        if dense:
            self.KEclose = nn.Parameter(torch.randn(nEclose, nEopen, 9, 9) * stdv)
        else:
            self.KEclose = nn.Parameter(torch.randn(nEclose, nEopen) * stdv)

        NEfeatures =  2*nNopen + nEopen
        #NEfeatures =  2*nNopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nEhid, NEfeatures,9,9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nEhid, nEhid,9,9) * stdvp)
            self.KE3 = nn.Parameter(torch.rand(nlayer, nEopen, nEhid, 9, 9) * stdvp)
        else:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nEhid, NEfeatures) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nEhid, nEhid) * stdvp)
            self.KE3 = nn.Parameter(torch.rand(nlayer, nEopen, nEhid) * stdvp)

        Nnfeatures = 2*nEhid + nNopen
        #Nnfeatures = 2*nEhid
        self.KN  = nn.Parameter(torch.rand(nlayer,nNopen, Nnfeatures)*stdvp)

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

    def forward(self,xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, N]
        xn = F.conv1d(xn, self.KNopen.unsqueeze(-1))
        xe = self.edgeConv(xe, self.KEopen)

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):

            Kei1 = self.KE1[i]
            Kei2 = self.KE2[i]
            Kei3 = self.KE3[i]
            Kni  = self.KN[i]

            # Move from node to the edge space
            gradX =  Graph.nodeGrad(xn)
            intX  =  Graph.nodeAve(xn)
            xec    = torch.cat([intX, xe, gradX], dim=1)
            #xec    = torch.cat([intX, gradX], dim=1)

            # 1D convs on the edge space and nonlinearity
            xec    = self.edgeConv(xec, Kei1)
            xec = F.layer_norm(xec,xec.shape) #xec    = tv_norm(xec)
            xec = torch.relu(xec)
            xec = self.edgeConv(xec, Kei2)
            xec = F.layer_norm(xec, xec.shape)  #xec    = tv_norm(xec)
            xec = torch.relu(xec)

            # back to the node space
            divXe  = Graph.edgeDiv(xec)
            intXe  = Graph.edgeAve(xec)
            xnc    = torch.cat([intXe, xn, divXe], dim=1)

            # Edge and node updates
            # xe(i+1) = xe(i) + f(xn(i))
            # xn(i+1) = xn(i) + g(xe(i))
            xe    = xe + self.h*self.edgeConv(xec, Kei3)
            xn    = xn + self.h*F.conv1d(xnc, Kni.unsqueeze(-1))

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xe = self.edgeConv(xe, self.KEclose)

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
