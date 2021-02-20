import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def getConnectivity(X, nsparse=16):
    X2 = torch.pow(X,2).sum(dim=1,keepdim=True)
    D = X2 + X2.transpose(2,1) - 2*X.transpose(2,1)@X
    D = torch.exp(torch.relu(D))

    vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)
    IJ = torch.stack([I, J], dim=1)

    return IJ

def makeBatch(Ilist,Jlist,nnodesList, Wlist=[1.0]):

    I = torch.tensor(Ilist[0])
    J = torch.tensor(Jlist[0])
    W = torch.tensor(Wlist[0])
    nnodesList = torch.tensor(nnodesList,dtype=torch.long)
    n = nnodesList[0]
    for i in range(1,len(Ilist)):
        Ii = torch.tensor(Ilist[i])
        Ji = torch.tensor(Jlist[i])

        I = torch.cat((I, n+Ii))
        J = torch.cat((J, n+Ji))
        ni = nnodesList[i].long()
        n += ni
        if len(Wlist)>1:
            Wi = torch.tensor(Wlist[i])
            W = torch.cat((W, Wi))

    return I,J, nnodesList, W

class graph(nn.Module):

    def __init__(self, Ilist, Jlist, nnodesList, W= [1.0]):
        super(graph, self).__init__()

        I, J, nnodes, W = makeBatch(Ilist, Jlist, nnodesList, W)

        self.iInd    = I
        self.jInd    = J
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self,x):
        g = self.W*(x[:,:,self.iInd] - x[:,:,self.jInd])
        return g
    def nodeAve(self,x):
        g = self.W*(x[:,:,self.iInd] + x[:,:,self.jInd])/2.0
        return g


    def edgeDiv(self,g):
        x = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        #z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        #for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        #for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(2, self.iInd, self.W*g)
        x.index_add_(2, self.jInd, -self.W*g)

        return x

    def edgeAve(self,g, method='max'):
        x1 = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        x2 = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)

        x1.index_add_(2, self.iInd, self.W*g)
        x2.index_add_(2, self.jInd, self.W*g)
        if method=='max':
            x = torch.max(x1,x2)
        elif method=='ave':
            x = (x1 + x2)/2
        return x


    def nodeLap(self,x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def edgeLength(self,x):
        g = self.nodeGrad(x)
        L = torch.sqrt(torch.pow(g,2).sum(dim=1))
        return L



# test it
I1 = torch.tensor([0,1,2,3,4,5])
I2 = torch.tensor([0,1,2,3,4])
J1 = torch.tensor([1,2,3,4,5,0])
J2 = torch.tensor([1,2,3,4,0])
Ilist = [I1,I2]
Jlist = [J1,J2]
nodeslist = [6,5]
Wlist = [torch.rand(6),torch.rand(5)]

G = graph(Ilist,Jlist,nodeslist, Wlist)

G1 = graph([I1],[J1],[nodeslist[0]], [Wlist[0]])
G2 = graph([I2],[J2],[nodeslist[1]], [Wlist[1]])


x = torch.randn(1,3,11)
g  = G.nodeGrad(x)
g1 = G1.nodeGrad(x[:,:,:6])
g2 = G2.nodeGrad(x[:,:,6:])
gg = torch.cat((g1,g2),dim=-1)

print(torch.norm(g-gg))


