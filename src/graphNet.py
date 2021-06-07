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

### Triangles Constraint
def triConstraint(X,dab=1.52, dan=1.45, dbn=2.46):
    X = X.squeeze()
    Xa = X[:3,:]
    Xb = X[3:6,:]
    Xn = X[6:,:]

    cab = torch.sum((Xa-Xb)**2, dim=1) - dab**2
    can = torch.sum((Xa-Xn)**2, dim=1) - dan**2
    cbn = torch.sum((Xb-Xn)**2, dim=1) - dbn**2
    c   = torch.cat((cab,can,cbn))
    return c


def projTriConstraint(X, dab=1.52, dan=1.45, dbn=2.46):

    BAN = np.arccos((dab**2 + dan**2 - dbn**2)/(2*dab*dan))

    X = X.squeeze()
    Xa = X[:3, :]
    Xb = X[3:6, :]
    Xn = X[6:, :]

    Rab = Xa - Xb
    Rab = Rab/torch.sqrt(torch.sum(Rab**2,dim=0,keepdim=True) + 1e-5)
    Xb  = Xa + dab*Rab
    Ran = Xa - Xn
    Ran = Ran/torch.sqrt(torch.sum(Ran**2,dim=0,keepdim=True) + 1e-5)
    Xnhat  = Xa + dan*Ran

    # set n = (Ran X Rab)/(normalize)
    # Traingle with Xa, Xb, Xn
    # Then  (n, Xn-Xa)   = 0
    #       (Xa-Xb, Xa-Xn) = Q*cos(BAN)
    #       (Xb-Xn,Xb-Xa)  = Q*cos(NBA)


    return Xa, Xb, Xn


class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nNopen, nEopen, nNclose, nEclose, nlayer, h=0.1, const=False):
        super(graphNetwork, self).__init__()

        self.const = const
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
        if self.const:
            xn = proj(xn, self.KNclose, n=100)

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

            if self.const:
                xn = proj(xn, self.KNclose, n=5)

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xe = F.conv1d(dxe, self.KEclose.unsqueeze(-1))

        if self.const:
            c = constraint(xn)
            if c.abs().mean() > 0.1:
                xn = proj(xn, torch.eye(3, 3), n=500)


        return xn, xe


#===============================================================

def vectorLayer(V, W):
    # V = [B, C, 3  N]
    # W = [Cout, C, 3]
    Vx    = F.conv1d(V[:, :, 0, :],W[:,:,0].unsqueeze(2)).unsqueeze(2)
    Vy    = F.conv1d(V[:, :, 1, :],W[:,:,1].unsqueeze(2)).unsqueeze(2)
    Vz    = F.conv1d(V[:, :, 2, :],W[:,:,2].unsqueeze(2)).unsqueeze(2)
    Vout  = torch.cat((Vx,Vy,Vz),dim=2)

    return Vout

def vectorLayerT(V, W):
    # V = [B, C, 3  N]
    # W = [C, Cout, 3]
    Vx    = F.conv1d(V[:, :, 0, :],W[:,:,0].t().unsqueeze(2)).unsqueeze(2)
    Vy    = F.conv1d(V[:, :, 1, :],W[:,:,1].t().unsqueeze(2)).unsqueeze(2)
    Vz    = F.conv1d(V[:, :, 2, :],W[:,:,2].t().unsqueeze(2)).unsqueeze(2)
    Vout  = torch.cat((Vx,Vy,Vz),dim=2)

    return Vout

def vectorDotProdLayer(V, W):
    # V = [B, C, 3  N]
    # W = [B, C, 3, N]
    Vout  = torch.sum(V*W,dim=2,keepdim=True)
    return Vout

def vectorCrossProdLayer(V, W):
    # V = [B, C, 3  N]
    # W = [B, C, 3, N]
    Cx  = V[:, :, 1, :] * W[:, :, 2, :] - V[:, :, 2, :] * W[:, :, 1, :]
    Cy  = V[:, :, 2, :] * W[:, :, 0, :] - V[:, :, 0, :] * W[:, :, 2, :]
    Cz  = V[:, :, 0, :] * W[:, :, 1, :] - V[:, :, 1, :] * W[:, :, 0, :]

    C = torch.cat((Cx,Cy,Cz),dim=2)

    return C


def vectorRelU(V,Q):
    Qn = Q/torch.norm(Q)
    Qn = torch.reshape(Qn, (1, 1, 3, 1))
    Vn = V/torch.sqrt((V**2).sum(dim=2,keepdim=True)+1e-5)

    T = torch.sum(Vn * Qn,2,keepdim=True)
    V = V - T.sign() * Qn

    return V





#