import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X,dim=1, keepdim=True)
    X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True) + eps)
    return X


def getConnectivity(X):

    D = torch.pow(X,2).sum(dim=1,keepdim=True) + \
        torch.pow(X,2).sum(dim=1,keepdim=True).transpose(2,1) - \
        2*X.transpose(2,1)@X
    D = F.softshrink(D,D.mean())
    IJ = torch.nonzero(D>1e-5)
    return IJ

class graph(nn.Module):

    def __init__(self, iInd, jInd, nnodes):
        super(graph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes

    def nodeGrad(self,x,w=1.0):
        g = w*(x[:,:,self.iInd] - x[:,:,self.jInd])
        return g

    def nodeAve(self,x,w=1.0):
        g = w*(x[:,:,self.iInd] + x[:,:,self.jInd])/2.0
        return g


    def edgeDiv(self,g, w=1.0):
        x = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        #z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        #for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        #for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(2, self.iInd, w*g)
        x.index_add_(2, self.jInd, -w*g)

        return x

    def edgeAve(self,g, w=1.0, method='max'):
        x1 = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        x2 = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)

        x1.index_add_(2, self.iInd, w*g)
        x2.index_add_(2, self.jInd, w*g)
        if method=='max':
            x = torch.max(x1,x2)
        else:
            x = (x1 + x2)/2
        return x


    def nodeLap(self,x,w=1.0):
        g = self.nodeGrad(x,w)
        d = self.edgeDiv(g,w)
        return d

    def edgeLength(self,x):
        g = self.nodeGrad(x)
        L = torch.sqrt(torch.pow(g,2).sum(dim=1))
        return L

class graphDiffusionLayer(nn.Module):
    def __init__(self, G,nIn,nhid):
        super(graphDiffusionLayer, self).__init__()
        self.Graph  = G
        W           = self.init_weights(G,nIn,nhid)
        self.Weight = W

    def init_weights(self,G,nIn,nhid):

        W  = nn.Parameter(torch.rand(nhid, nIn)*1e-1)
        return W

    def forward(self,x,w=1.0):
        z = self.Graph.nodeGrad(x,w)
        z = self.Weight@z
        z = F.instance_norm(z)
        z = torch.relu(z)
        z = self.Weight.t()@z
        z = self.Graph.edgeDiv(z,w)

        return z




###### Testing stuff
# tests = 0
# if tests:
#     nnodes = 512
#     II = torch.torch.zeros(nnodes*(nnodes-1)//2)
#     JJ = torch.torch.zeros(nnodes*(nnodes-1)//2)
#
#     k = 0
#     for i in range(nnodes):
#         for j in range(i+1,nnodes):
#             II[k] = i
#             JJ[k] = j
#             k+=1
#
#     G = graph(II,JJ,nnodes)
#     x  = torch.randn(1,128,nnodes)
#
#     test_adjoint = 0
#     if test_adjoint:
#         # Adjoint test
#         w = torch.rand(G.iInd.shape[0])
#         y = G.nodeGrad(x,w)
#         ne = G.iInd.numel()
#         z = torch.randn(1,128,ne)
#         a1 = torch.sum(z*y)
#         v = G.edgeDiv(z,w)
#         a2 = torch.sum(v*x)
#         print(a1,a2)
#
#
#
#     nhid = 8
#     L = graphDiffusionLayer(G,x.shape[1],nhid)
#
#     y = L(x)



