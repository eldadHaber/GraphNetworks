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
    X = X - torch.mean(X, dim=1, keepdim=True) # First we set the mean across all channels to 0
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


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
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


def diffX(X):
    X = X.squeeze()
    return X[:,1:] - X[:,:-1]

def diffXT(X):
    X  = X.squeeze()
    D  = X[:,:-1] - X[:,1:]
    d0 = -X[:,0].unsqueeze(1)
    d1 = X[:,-1].unsqueeze(1)
    D  = torch.cat([d0,D,d1],dim=1)
    return D


def constraint(X,d=3.8):
    X = X.squeeze()
    c = torch.ones(1,3,device=X.device)@(diffX(X)**2) - d**2

    return c

def dConstraint(S,X):
    dX = diffX(X)
    dS = diffX(S)
    e  = torch.ones(1,3,device=X.device)
    dc = 2*e@(dX*dS)
    return dc

def dConstraintT(c,X):
    dX = diffX(X)
    e = torch.ones(3, 1, device=X.device)
    C = (e@c)*dX
    C = diffXT(C)
    return 2*C

def proj(x,K,n=1, d=3.8):

    for j in range(n):

        x3 = F.conv1d(x, K.unsqueeze(-1))
        c = constraint(x3, d)
        lam = dConstraintT(c, x3)
        lam = F.conv_transpose1d(lam.unsqueeze(0), K.unsqueeze(-1))

        #print(j, 0, torch.mean(torch.abs(c)).item())

        with torch.no_grad():
            if j==0:
                alpha = 1.0/lam.norm()
            lsiter = 0
            while True:
                xtry = x - alpha * lam
                x3 = F.conv1d(xtry, K.unsqueeze(-1))
                ctry = constraint(x3, d)
                #print(j, lsiter, torch.mean(torch.abs(ctry)).item()/torch.mean(torch.abs(c)).item())

                if torch.norm(ctry) < torch.norm(c):
                    break
                alpha = alpha/2
                lsiter = lsiter+1
                if lsiter > 10:
                    break

            if lsiter==0:
                alpha = alpha*1.5



        x = x - alpha * lam

    return x

class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, const=False):
        super(graphNetwork, self).__init__()

        self.const = const
        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        nopen      = 3*nopen
        self.nopen = nopen
        Nfeatures  = 2 * nopen

        Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
        Idt = torch.eye(Nfeatures,nhid).unsqueeze(0)
        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.KNclose = nn.Parameter(torch.eye(nNclose, nopen))

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
        if self.const:
            xn = proj(xn, self.KNclose, n=100)
            #x3 = F.conv1d(xn, self.KNclose.unsqueeze(-1))
            #c = constraint(x3)
            #print(c.abs().mean().item())

        nlayers = self.KE1.shape[0]

        #xold = xn
        for i in range(nlayers):

            # Compute the distance in real space
            x3    = F.conv1d(xn, self.KNclose.unsqueeze(-1))
            w     = Graph.edgeLength(x3)
            w     = self.Kw@w
            w     = w/(torch.std(w)+1e-4)
            w     = torch.exp(-(w**2))
         #w     = torch.ones(xe.shape[2], device=xe.device)

            gradX = Graph.nodeGrad(xn,w)
            intX = Graph.nodeAve(xn,w)

            dxe = torch.cat([gradX, intX], dim=1)
            dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            divE = Graph.edgeDiv(dxe[:,:self.nopen,:],w)
            aveE = Graph.edgeAve(dxe[:,self.nopen:,:],w)

            xn = xn - self.h * (divE + aveE)

            #tmp = xn
            #xn = 2*xn - xold - (self.h**2) * (divE + aveE)
            #xold = tmp

            if self.const:
                xn = proj(xn, self.KNclose, n=5)
                #x3 = F.conv1d(xn, self.KNclose.unsqueeze(-1))
                #c = constraint(x3)
                #print(c.abs().mean().item())

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        if self.const:
            c = constraint(xn)
            if c.abs().mean() > 0.1:
                xn = proj(xn, torch.eye(3, 3), n=500)
                #c = constraint(xn)
            #print(c.abs().mean().item())

        return xn, xe




# class graphNetwork(nn.Module):
#
#     def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1):
#         super(graphNetwork, self).__init__()
#
#         self.h = h
#         stdv = 1.0 #1e-2
#         stdvp = 1.0 # 1e-3
#         embeddim = 8
#         self.Embed = nn.Embedding(100,embeddim)
#         self.K1Nopen = nn.Parameter(torch.randn(nopen, embeddim) * stdv)
#         self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
#         self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
#         self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
#
#         nopen      = 3*nopen
#         self.nopen = nopen
#         Nfeatures  = 2 * nopen
#
#         Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
#         Idt = torch.eye(Nfeatures,nhid).unsqueeze(0)
#         IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
#         IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
#         self.KE1 = nn.Parameter(IdTensor * stdvp)
#         self.KE2 = nn.Parameter(IdTensort * stdvp)
#
#         self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)
#
#         self.filters = nn.ModuleList()
#
#         self.filters.append(nn.Sequential(nn.Linear(25, nhid), nn.Tanh(), nn.Linear(nhid, nhid)))
#         self.filters.append(nn.Sequential(nn.Linear(25, nhid), nn.Tanh(), nn.Linear(nhid, nhid)))
#         for i in range(nlayer):
#             self.filters.append(nn.Sequential(nn.Linear(25,nhid*3), nn.Tanh(), nn.Linear(nhid*3,nhid*3)))
#             self.filters.append(nn.Sequential(nn.Linear(25, nhid*3), nn.Tanh(), nn.Linear(nhid*3, nhid*3)))
#             self.filters.append(nn.Sequential(nn.Linear(25, nhid*3), nn.Tanh(), nn.Linear(nhid*3, nhid*3)))
#             self.filters.append(nn.Sequential(nn.Linear(25, nhid*3), nn.Tanh(), nn.Linear(nhid*3, nhid*3)))
#         return
#
#     def doubleLayer(self, x, K1, K2):
#
#         x = torch.tanh(x)
#         x = F.conv1d(x, K1.unsqueeze(-1))  # self.edgeConv(x, K1)
#         x = tv_norm(x)
#         x = torch.tanh(x)
#         x = F.conv1d(x, K2.unsqueeze(-1))
#         x = torch.tanh(x)
#         return x
#
#     def forward(self, xn, xe, Graph):
#
#         # Opening layer
#         # xn = [B, C, N]
#         # xe =  [B, C, E]
#         # Opening layer
#
#         xn = self.Embed(xn.to(dtype=torch.int64)).transpose(2,3).squeeze(1)
#         xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
#         xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
#
#         WeDxe = self.filters[0](Graph.GSD).transpose(2,3).squeeze(1)
#         eDxe = Graph.edgeDiv(xe, W=WeDxe)
#
#         WeAxe = self.filters[1](Graph.GSD).transpose(2,3).squeeze(1)
#         eAxe = Graph.edgeAve(xe, W=WeAxe)
#
#         xn = torch.cat([xn,eDxe, eAxe], dim=1)
#
#         nlayers = self.KE1.shape[0]
#
#         for i in range(nlayers):
#
#             W = self.filters[i*4+2](Graph.GSD).transpose(2, 3).squeeze(1)
#             gradX = Graph.nodeGrad(xn,W=W)
#             W = self.filters[i*4+3](Graph.GSD).transpose(2, 3).squeeze(1)
#             intX = Graph.nodeAve(xn,W=W)
#
#             dxe = torch.cat([gradX, intX], dim=1)
#             dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])
#
#             W = self.filters[i*4+4](Graph.GSD).transpose(2, 3).squeeze(1)
#             divE = Graph.edgeDiv(dxe[:,:self.nopen,:],W=W)
#             W = self.filters[i*4+5](Graph.GSD).transpose(2, 3).squeeze(1)
#             aveE = Graph.edgeAve(dxe[:,self.nopen:,:],W=W)
#
#             xn = xn - self.h * (divE + aveE)
#
#         xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
#
#         return xn, xe
#

Test = False
if Test:
    nNin = 20
    nEin = 3
    nNopen = 32
    nEopen = 16
    nEhid = 64
    nNclose = 3
    nEclose = 2
    nlayer = 6
    model = graphNetwork(nNin, nEin, nNopen, nEopen, nEhid, nNclose, nEclose, nlayer)

    L = 55
    xn = torch.zeros(1, nNin, L)
    xn[0, :, 23] = 1
    xe = torch.ones(1, nEin, L, L)

    G = GO.dense_graph(L)

    xnout, xeout = model(xn, xe, G)

