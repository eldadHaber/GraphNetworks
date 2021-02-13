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
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


def getConnectivity(X):
    D = torch.pow(X, 2).sum(dim=1, keepdim=True) + \
        torch.pow(X, 2).sum(dim=1, keepdim=True).transpose(2, 1) - \
        2 * X.transpose(2, 1) @ X
    D = torch.relu(D)
    D = F.softshrink(D, D.mean())
    IJ = torch.nonzero(D > 1e-5)
    return IJ


class graph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=1.0):
        super(graph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self, x):
        g = self.W * (x[:, :, self.iInd] - x[:, :, self.jInd])
        return g

    def nodeAve(self, x):
        g = self.W * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        return g

    def edgeDiv(self, g):
        x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(2, self.iInd, self.W * g)
        x.index_add_(2, self.jInd, -self.W * g)

        return x

    def edgeAve(self, g, method='max'):
        x1 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)

        x1.index_add_(2, self.iInd, self.W * g)
        x2.index_add_(2, self.jInd, self.W * g)
        if method == 'max':
            x = torch.max(x1, x2)
        elif method == 'ave':
            x = (x1 + x2) / 2
        return x

    def nodeLap(self, x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        return L


class dense_graph(nn.Module):

    def __init__(self, nnodes, W=torch.ones(1)):
        super(dense_graph, self).__init__()
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self, x):
        w = self.W
        x = x.squeeze(0).unsqueeze(1)
        g = w * (x - x.transpose(1, 2))
        g = g.unsqueeze(0)
        return g

    def nodeAve(self, x):
        w = self.W
        x = x.squeeze(0).unsqueeze(1)
        g = w * (x + x.transpose(1, 2)) / 2.0
        g = g.unsqueeze(0)
        return g

    def edgeDiv(self, g):
        w = self.W
        g = w * g
        x1 = g.sum(dim=2, keepdim=True).squeeze(0)
        x2 = g.sum(dim=3, keepdim=True).squeeze(0)
        x = x1 - x2.transpose(2, 1)
        x = x.squeeze(1).unsqueeze(0)
        return x

    def edgeAve(self, g, method='max'):
        w = self.W
        g = w * g
        x1 = g.mean(dim=2, keepdim=True).squeeze(0)
        x2 = g.mean(dim=3, keepdim=True).squeeze(0)
        x2 = x2.transpose(1, 2)
        if method == 'max':
            x = torch.max(x1, x2)
        else:
            x = (x1 + x2) / 2
        x = x.squeeze(1).unsqueeze(0)
        return x

    def nodeLap(self, x):
        w = self.W
        g = self.nodeGrad(x, w)
        d = self.edgeDiv(g, w)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = torch.pow(g, 2).sum(dim=1)
        return L

###### Testing stuff
# tests = 1
# if tests:
#    nnodes = 512
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
