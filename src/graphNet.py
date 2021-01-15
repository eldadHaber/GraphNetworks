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
    def __init__(self, G,nIn, nopenN, nopenE, nclose,nlayer,h=0.1):
        super(varletNetworks, self).__init__()
        self.Graph  = G
        KNopen, KEopen, KNclose, KEclose, KN, KE = self.init_weights(nIn,nopenN,nopenE,nclose,nlayer)
        self.KNopen = KNopen
        self.KEopen = KEopen
        self.KNclose = KNclose
        self.KEclose = KEclose
        self.KN      = KN
        self.KE      = KE

    def init_weights(self,nIn,nopenN,nopenE,nclose,nlayer):

        KNopen = nn.Parameter(torch.rand(nopenN, nIn)*1e-1)
        KEopen = nn.Parameter(torch.rand(nopenE, nIn)*1e-1)
        KNclose = nn.Parameter(torch.rand(nclose,nopenN)*1e-1)
        KEclose = nn.Parameter(torch.rand(nclose,nopenE)*1e-1)

        KN = torch.zeros(nlayer,nopenE, nopenN)
        KE = torch.zeros(nlayer,nopenN, nopenE)
        for i in range(nlayer):
            KN[i,:,:] = nn.Parameter(torch.rand(nopenE, nopenN)*1e-3)
            KE[i,:,:] = nn.Parameter(torch.rand(nopenN, nopenE) * 1e-3)

        return KNopen, KEopen, KNclose, KEclose, KN, KE

    def forward(self, xn, xe):
        # xn - node feature
        # xe - edge features

        nL = len(self.Kne)

        xn = self.KNopen@xn
        xe = self.KEopen@xe
        for i in range(nL):
            Kni = self.Kn[i]
            Kei = self.Ke[i]

            #Coords = self.KNclose@xe
            Ai = Kni@self.Graph.nodeGrad(xn)
            Ai = tv_norm(Ai)
            xe = xe + self.h*torch.relu(Ai)
            Bi = torch.relu(Kei@self.Graph.edgeDiv(xe))
            Bi = tv_norm(Bi)
            xn = xn + self.h*torch.relu(Bi)

        xn = self.KNclose @ xn
        xe = self.KNclose @ xe
        return xn, xe
