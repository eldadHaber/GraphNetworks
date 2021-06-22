import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##### Scalar graph #################################################
class scalarGraph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=torch.tensor([1.0])):
        super(scalarGraph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, self.iInd] - x[:, :, self.jInd])
        return g

    def nodeAve(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        return g


    def edgeDiv(self, g, W=[]):
        if len(W)==0:
            W = self.W
        x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(2, self.iInd, W * g)
        x.index_add_(2, self.jInd, -W * g)

        return x

    def edgeAve(self, g,  W=[], method='max'):
        if len(W)==0:
            W = self.W
        x1 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)

        x1.index_add_(2, self.iInd, W * g)
        x2.index_add_(2, self.jInd, W * g)
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
        #L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        L = torch.pow(g, 2).sum(dim=1)

        return L

########## Vector Graph ##########

class vectorGraph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=torch.tensor([1.0])):
        super(vectorGraph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, :, self.iInd] - x[:, :, :, self.jInd])
        return g

    def nodeAve(self, x, W=[]):
        if len(W)==0:
            W = self.W
        g = W * (x[:, :, :, self.iInd] + x[:, :, :, self.jInd]) / 2.0
        return g


    def edgeDiv(self, g, W=[]):
        if len(W)==0:
            W = self.W
        x = torch.zeros(g.shape[0], g.shape[1], 3, self.nnodes, device=g.device)
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]

        x.index_add_(3, self.iInd, W * g)
        x.index_add_(3, self.jInd, -W * g)

        return x

    def edgeAve(self, g,  W=[], method='max'):
        if len(W)==0:
            W = self.W
        x1 = torch.zeros(g.shape[0], g.shape[1], 3, self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], 3, self.nnodes, device=g.device)

        x1.index_add_(3, self.iInd, W * g)
        x2.index_add_(3, self.jInd, W * g)
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
        #L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        L = torch.pow(g, 2).sum(dim=(1,2))

        return L

    def edgeAngle(self, x):

        V = vectorCrossCrossProd(x[:,:,:,self.iInd], x[:,:,:,self.jInd])
        L = torch.sum(V**2, dim=2, keepdim=True)
        return L


######### END SCALAR GRAPH ####################################


### Graph Functions ############################################

def vectorLayer(V, W):
    # V = [B, C, 9  N]
    # W = [Cout, C, 9]
    Vout = torch.zeros(V.shape[0],W.shape[0],9,V.shape[3],device=V.device)
    Vout[:,:,[0,3,6],:]  = F.conv1d(V[:, :, [0,3,6], :],W[:,:,:3]).unsqueeze(2)
    Vout[:,:,[1,4,7],:]  = F.conv1d(V[:, :, [1,4,7], :],W[:,:,3:6]).unsqueeze(2)
    Vout[:,:,[2,5,8],:]  = F.conv1d(V[:, :, [2,5,8], :],W[:, :, 3:6]).unsqueeze(2)

    return Vout

def vectorDotProdLayer(V, W):
    # V = [B, C, 9  N]
    # W = [B, C, 9, N]
    V1 = torch.sum(V[:,:,[0,3,6],:]*W[:,:,[0,3,6],:],dim=2,keepdim=True)
    V2 = torch.sum(V[:,:,[1,4,7],:]*W[:,:,[1,4,7], :],dim=2, keepdim=True)
    V3 = torch.sum(V[:,:,[2,5,8],:]*W[:,:,[2,5,8], :],dim=2, keepdim=True)
    Vout = torch.cat((V1,V2,V3),dim=2)

    return Vout


def vectorDDotProdLayer(V, W):
    # V = [B, C, 9  N]
    # W = [B, C, 9, N]
    Vout = torch.sum(V*W, dim=2, keepdim=True)

    return Vout


def vectorCrossProd(V):
    # V = [B, C, 9  N]
    # W = [B, C, 9, N]
    # vy*wz - vz*wy
    # vz*wx - vx*wz
    # vx*wy - vy*wx
    U = V[:, :, :3, :] - V[:, :, 3:6, :]
    W = V[:, :, :3, :] - V[:, :, 6:, :]

    Cx  = (U[:, :, 1, :] * W[:, :, 2, :] - U[:, :, 2, :] * W[:, :, 1, :]).unsqueeze(2)
    Cy  = (U[:, :, 2, :] * W[:, :, 0, :] - U[:, :, 0, :] * W[:, :, 2, :]).unsqueeze(2)
    Cz  = (U[:, :, 0, :] * W[:, :, 1, :] - U[:, :, 1, :] * W[:, :, 0, :]).unsqueeze(2)

    C = torch.cat((Cx,Cy,Cz),dim=2)

    return C


def vectorCrossCrossProd(V1, V2):
    # V1 = [B, C, 9  N]
    # V2 = [B, C, 9, N]
    # vy*wz - vz*wy
    # vz*wx - vx*wz
    # vx*wy - vy*wx
    n1 = vectorCrossProd(V1)
    n2 = vectorCrossProd(V2)

    normn1 = torch.sqrt(torch.sum(n1**2, dim=2,keepdim=True)+1e-5)
    normn2 = torch.sqrt(torch.sum(n2**2, dim=2, keepdim=True)+1e-5)
    n1 = n1/normn1
    n2 = n2/normn2

    Cx = (n1[:, :, 1, :] * n2[:, :, 2, :] - n1[:, :, 2, :] * n2[:, :, 1, :]).unsqueeze(2)
    Cy = (n1[:, :, 2, :] * n2[:, :, 0, :] - n1[:, :, 0, :] * n2[:, :, 2, :]).unsqueeze(2)
    Cz = (n1[:, :, 0, :] * n2[:, :, 1, :] - n1[:, :, 1, :] * n2[:, :, 0, :]).unsqueeze(2)

    C = torch.cat((Cx, Cy, Cz), dim=2)

    return C

def vectorRelU(V,Q):
    Q = torch.reshape(Q, (1, 1, 3, 1))
    Qn = Q/torch.norm(Q)

    T = torch.sum(V * Q,2,keepdim=True)
    W = V - torch.relu(T/torch.norm(Q)) * Qn

    return W

########### END GRAPH FUNCTIONS ##################################################

class graphVectorNetwork(nn.Module):

    def __init__(self, nNin, nEin, nNopen, nEopen, nNclose, nEclose, nlayer, h=0.1):
        super(graphVectorNetwork, self).__init__()

        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nNopen, nNin,3) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nNopen, nNopen,3) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nEopen, nEin,3) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nEopen, nEopen,3) * stdv)

        nopen      = nNopen + 2*nEopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen

        nhid = 2*nopen
        Id  = torch.eye(nhid,2*nopen).unsqueeze(0).unsqueeze(3)
        Id  = torch.cat((Id,Id, Id),dim=3)
        Idt = torch.eye(2*nopen,nhid).unsqueeze(0).unsqueeze(3)
        Idt = torch.cat((Idt, Idt, Idt), dim=3)

        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen, 3))
        #self.KEclose = nn.Parameter(torch.randn(nEclose, 3*nopen, 3))
        self.Q       = nn.Parameter(torch.randn(3))

    def doubleVectorLayer(self, x, K1, K2, Q):

        x = vectorRelU(x, Q)
        x = vectorLayer(x, K1)
        x = vectorRelU(x,Q)
        x = vectorLayer(x, K2)
        x = vectorRelU(x, Q)

        return x

    def doubleScalarLayer(self, x, K1, K2):

        x = torch.tanh(x)
        x = F.conv1d(x, K1)
        x = torch.tanh(x)
        x = F.conv1d(x, K2)
        x = torch.tanh(x)

        return x


    def forward(self, xnV, xnS, xeV, xeS, GraphV, GraphS):
        # xnV - vector nodal input
        # xeV - vector edge input
        # xnS - scalar nodal input
        # xeS - scalar edge input

        # Opening layer
        # xn = [B, C, N]
        # xe =  [B, C, E]
        # Opening layer
        xnV = self.doubleVectorLayer(xnV, self.KV1Nopen, self.KV2Nopen, self.Q)
        xeV = self.doubleVectorLayer(xeV, self.KV1Eopen, self.KV2Eopen, self.Q)
        xnS = self.doubleScalarLayer(xnS, self.KS1Nopen, self.KS2Nopen)
        xeS = self.doubleScalarLayer(xeS, self.KS1Eopen, self.KS2Eopen)

        # move everything to nodes
        xnV = torch.cat([xnV,GraphV.edgeDiv(xeV), GraphV.edgeAve(xeV)], dim=1)
        xnS = torch.cat([xnS, GraphS.edgeDiv(xeS), GraphV.edgeAve(xeS)], dim=1)

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):

            # Get scalar quantities to edges
            gradSX   = GraphV.nodeGrad(xnS)
            intSX    = GraphV.nodeAve(xnS)

            # Get vector -> scalar quantities to edges


            #dxe = torch.cat([gradX, intX, cpX], dim=1)
            dxe = torch.cat([gradX, intX], dim=1)
            Flux  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i], self.Q)

            dxe = Graph.edgeDiv(Flux[:,:self.nopen,:,:]) + Graph.edgeAve(Flux[:,self.nopen:,:,:])

            #tmp  = xn.clone()
            xn   = xn - self.h * dxe
            #xnold = tmp

        xn = vectorLayer(xn, self.KNclose)


        return xn, Flux


#===============================================================

nNin    = 3
nEin    = 1
nNopen  = 16
nEopen  = 1
nNclose = 3
nEclose = 1
nlayer  = 5

model = graphVectorNetwork(nNin, nEin, nNopen, nEopen, nNclose, nEclose, nlayer)

n = 17
A = torch.rand(n,n)
A = A.t()@A
vals, indices = torch.topk(A, k=9, dim=1)
nd = A.shape[1]
I = torch.ger(torch.arange(nd), torch.ones(9, dtype=torch.long))
I = I.view(-1)
J = indices.view(-1).type(torch.LongTensor)

G = vectorGraph(I,J,17)
xn = torch.randn(1,3,3,17)
xe = torch.randn(1,1,3,I.shape[0])

yn, Flux = model(xn,xe, G)