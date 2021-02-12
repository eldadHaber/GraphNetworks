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


def doubleLayer(x,K1, K2, ln=True, drop=True):

    x = F.conv1d(x, K1.unsqueeze(-1))
    x = torch.relu(x)
    x = F.conv1d(x, K2.unsqueeze(-1))
    if ln:
        x = F.layer_norm(x, x.shape)
    if drop:
        x = F.dropout(x,p=0.2)
    return x

class gNet(nn.Module):
    def __init__(self, x_input_size, edj_input_size, hidden_size, output_size, nlayers):
        super().__init__()

        stdv = 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(hidden_size, x_input_size)*stdv)
        self.K2Nopen = nn.Parameter(torch.randn(hidden_size, hidden_size) * stdv)

        self.K1Eopen = nn.Parameter(torch.randn(hidden_size, edj_input_size)*stdv)
        self.K2Eopen = nn.Parameter(torch.randn(hidden_size, hidden_size) * stdv)

        self.KNout = nn.Parameter(torch.randn(output_size, hidden_size)*stdv)

        self.KE1 = nn.Parameter(torch.randn(nlayers, 2*hidden_size, 3*hidden_size)*stdv)
        self.KE2 = nn.Parameter(torch.randn(nlayers, hidden_size, 2*hidden_size)*stdv)

    def forward(self, xn, xe, Graph):

        # Opening layer
        xn = doubleLayer(xn,self.K1Nopen,self.K2Nopen)
        xe = doubleLayer(xe,self.K1Eopen,self.K2Eopen)

        row = Graph.iInd
        col = Graph.iInd

        nlayers = self.KE1.shape[0]
        for i in range(nlayers):

            xec = torch.cat([xn[:,:,row], xn[:,:,col], xe], dim=1)
            xec  = doubleLayer(xec, self.KE1[i], self.KE2[i],ln=False,drop=False)
            xnc  = Graph.edgeAve(xec, method='ave')
            xnc = F.layer_norm(xnc, xnc.shape)
            xnc = F.dropout(xnc, p=0.2)

            xn   = xn + xnc
            xe   = xe + xec

            xn    = F.relu(xn)
            xe    = F.relu(xe)

        xn = F.conv1d(xn, self.KNout.unsqueeze(-1))

        return xn, xe

Test = False
if Test:
    nNodes = 100
    nEdges = 99
    I = torch.tensor(np.arange(nEdges),dtype=torch.long)
    J = torch.tensor(np.arange(nEdges)+1,dtype=torch.long)

    x_input_size = 40
    edj_input_size = 2
    hidden_size = 80
    output_size = 3
    nlayers = 18

    model = gNet(x_input_size, edj_input_size, hidden_size, output_size, nlayers)

    G = GO.graph(I,J, nNodes)

    xn = torch.randn(1,x_input_size, nNodes)
    xe = torch.randn(1, edj_input_size, nEdges)

    xnOut, xeOut = model(xn,xe,G)
