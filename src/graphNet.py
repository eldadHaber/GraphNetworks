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

### General constraint
def constraintIJ(X, I, J, d=3.8):
    X = X.squeeze()
    XI = X[:,I]
    XJ = X[:,J]
    c = torch.ones(1,3,device=X.device)@((XI - XJ)**2) - d**2
    return c




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

        nopen      = 3*nopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen
        Nfeatures  = 2*nopen # [G*xn; Av*xn]

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


        return xn, xe


#===============================================================

def vectorLayer(V, wx, wy, wz):
    nvecs = V.shape[1]//3
    Vx    = F.conv1d(V[:,nvecs,:],wx.unsqueeze(1))
    Vy    = F.conv1d(V[:,nvecs:2*nvecs,:],wy.unsqueeze(1))
    Vz    = F.conv1d(V[:, 2*nvecs:,:],wz.unsqueeze(1))
    Vout  = torch.cat((Vx,Vy,Vz),dim=1)

    return Vout


class graphNetworkEqvrnt(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, const=False):
        super(graphNetworkEqvrnt, self).__init__()

        self.const = const
        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        nopen      = 3*nopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen
        Nfeatures  = 2*nopen+1 # [G*xn; Av*xn]

        Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
        Idt = torch.eye(Nfeatures,nhid).unsqueeze(0)
        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.Kw1 = nn.Parameter(torch.ones(nopen,1))
        self.Kw2 = nn.Parameter(torch.ones(3,Nfeatures))

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
        Nnodes = xn.shape[2]
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        xn = torch.cat([xn,Graph.edgeDiv(xe), Graph.edgeAve(xe)], dim=1)

        #Coords = (torch.tensor([1,0,0]).unsqueeze(1)@torch.arange(Nnodes).unsqueeze(0)).unsqueeze(0) + 0.1*torch.randn(1,3,Nnodes)
        Coords = torch.zeros(1, 3, Nnodes)
        j = 0
        for i in range(1,Nnodes):
            Coords[0,j,i] = Coords[0,j,i-1] + 3.8
            j = np.mod(j + 1,3)

        if self.const:
            Coords = proj(Coords, torch.eye(3, 3), n=1000)

        CoordsOld = Coords

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):

            # Compute the distance in real space
            w     = Graph.edgeLength(xn)
            w     = self.Kw1@w
            w     = w/(torch.std(w)+1e-4)
            w     = torch.tanh(w)

            # Node to edge
            gradX = Graph.nodeGrad(xn,w)
            intX = Graph.nodeAve(xn,w)
            d    = Graph.edgeLength(Coords).unsqueeze(0)

            # Edge attributes
            dxe = torch.cat([gradX, intX, d], dim=1)
            dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            # Update Coordinates
            w3         = self.Kw2@dxe
            w3         = w3/(torch.std(w3)+1e-4)
            w3         = torch.tanh(w3)
            gradCoords = Graph.nodeGrad(Coords, w3)
            aveCoords  = Graph.edgeAve(gradCoords, w3)

            tmp    = Coords
            Coords = CoordsOld + 2*self.h*aveCoords
            CoordsOld = tmp


            divE = Graph.edgeDiv(dxe[:,:self.nopen,:],w)
            aveE = Graph.edgeAve(dxe[:,self.nopen:-1,:],w)

            xn = xn - self.h * (divE + aveE)

            if self.const:
                Coords = proj(Coords, torch.eye(3, 3), n=50)


        if self.const:
            c = constraint(Coords)
            if c.abs().mean() > 0.1:
                Coords = proj(Coords, torch.eye(3, 3), n=500)


        return Coords, xn, xe