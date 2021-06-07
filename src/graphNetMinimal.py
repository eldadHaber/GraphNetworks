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

# t=1
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
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X



class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nNopen, nEopen, nNclose, nEclose, nlayer, h=0.1):
        super(graphNetwork, self).__init__()

        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nNopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nNopen, nNopen) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nEopen, nEin) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nEopen, nEopen) * stdv)

        nopen      = nNopen + 2*nEopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen

        nhid = nopen
        Id  = (torch.eye(nhid,5*nopen)).unsqueeze(0) #+ 1e-3*torch.randn(nhid,5*nopen)).unsqueeze(0)
        #Idt = torch.eye(Nfeatures,nhid).unsqueeze(0)
        Idt = (torch.eye(5*nopen, nhid)).unsqueeze(0) # + 1e-3*torch.randn(5*nopen,nhid)).unsqueeze(0)

        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.KNclose = nn.Parameter(torch.eye(nNclose, nopen))
        self.KEclose = nn.Parameter(torch.eye(nEclose, 5*nopen))

        self.Kw = nn.Parameter(torch.ones(nopen,1))

    def doubleLayer(self, x, K1, K2):

        x = torch.tanh(x)
        x = F.conv1d(x, K1.unsqueeze(-1))  # self.edgeConv(x, K1)
        x = tv_norm(x)
        x = torch.tanh(x)
        x = F.conv1d(x, K2.unsqueeze(-1))
        x = torch.tanh(x)

        return x

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe =  [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        xn = torch.cat([xn,Graph.edgeDiv(xe), Graph.edgeAve(xe)], dim=1)

        nlayers = self.KE1.shape[0]

        xnold = xn
        for i in range(nlayers):

            # Compute weights for the graph
            x3    = F.conv1d(xn, self.KNclose.unsqueeze(-1))
            w     = Graph.edgeLength(x3)
            w     = self.Kw@w
            w     = w/(torch.std(w)+1e-4)
            w     = torch.exp(-(w**2))
            #w     = torch.ones(xe.shape[2], device=xe.device)

            gradX   = Graph.nodeGrad(xn,w)
            intX    = Graph.nodeAve(xn,w)
            xgradX  = gradX*intX
            gradXsq = gradX*gradX
            xSq     = intX*intX

            dxe = torch.cat([gradX, intX, xgradX, gradXsq, xSq], dim=1)
            dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            divE = Graph.edgeDiv(dxe[:,:self.nopen,:],w)
            aveE = Graph.edgeAve(dxe[:,self.nopen:2*self.nopen,:],w)
            aveB = Graph.edgeAve(dxe[:,2*self.nopen:3*self.nopen, :], w)
            aveI = Graph.edgeAve(dxe[:,3*self.nopen:4*self.nopen, :], w)
            aveS = Graph.edgeAve(dxe[:,4*self.nopen:, :], w)

            tmp  = xn.clone()
            xn   = 2*xn - xnold - self.h * (divE + aveE + aveB + aveI + aveS)
            xnold = tmp

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xe = F.conv1d(dxe, self.KEclose.unsqueeze(-1))


        return xn, xe

