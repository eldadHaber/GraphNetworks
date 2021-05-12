import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler
import time


from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Data loading
caspver = "casp11"  # Change this to choose casp version

if "s" in sys.argv:
    base_path = '/home/eliasof/pFold/data/'
    import graphOps as GO
    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA


elif "e" in sys.argv:
    base_path = '/home/cluster/users/erant_group/pfold/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA


else:
    base_path = '../../../data/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA

# load training data
Aind = torch.load(base_path + caspver + '/AminoAcidIdx.pt')
Yobs = torch.load(base_path + caspver + '/RCalpha.pt')
MSK = torch.load(base_path + caspver + '/Masks.pt')
S = torch.load(base_path + caspver + '/PSSM.pt')
# load validation data
AindVal = torch.load(base_path + caspver + '/AminoAcidIdxVal.pt')
YobsVal = torch.load(base_path + caspver + '/RCalphaVal.pt')
MSKVal = torch.load(base_path + caspver + '/MasksVal.pt')
SVal = torch.load(base_path + caspver + '/PSSMVal.pt')

# load Testing data
AindTest = torch.load(base_path + caspver + '/AminoAcidIdxTesting.pt')
YobsTest = torch.load(base_path + caspver + '/RCalphaTesting.pt')
MSKTest = torch.load(base_path + caspver + '/MasksTesting.pt')
STest = torch.load(base_path + caspver + '/PSSMTesting.pt')

Aind = AindTest
Yobs = YobsTest
MSK  = MSKTest
S    = STest


nodeProperties, X, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs, MSK, 10, device=device)

def gradNode(X,I,J):
    X = X.squeeze(0)
    XI = X[:,I]
    XJ = X[:, J]
    return (XI - XJ)

def divEdge(g,I,J,nnodes):

    x = torch.zeros(g.shape[0], g.shape[1], nnodes, device=g.device)

    x.index_add_(2, I, g)
    x.index_add_(2, J, -g)

    return x


def constraintIJ(X, I, J, d=6):
    X = X.squeeze()
    G = gradNode(X,I,J)
    c = torch.ones(1,3,device=X.device)@(G**2) - d**2
    return c

def dConstraintIJ(S,X,I,J):
    dX = gradNode(X,I,J)
    dS = gradNode(S,I,J)
    e  = torch.ones(1,3,device=X.device)
    dc = 2*e@(dX*dS)
    return dc

def dConstraintTIJ(c,X, I, J):

    nnodes = X.shape[2]
    dX = gradNode(X,I,J)
    e = torch.ones(3, 1, device=X.device)
    C = (e@c)*dX
    C = divEdge(C.unsqueeze(0),I,J,nnodes)

    return 2*C

def penalty(X,I,J, dist):

    d = torch.sum(X[0,:,:]**2, dim=0, keepdim=True)
    D = d + d.t() - 2*X[0,:,:].t()@X[0,:,:]
    D[I,J] = 0
    Iz,Jz = torch.nonzero(D, as_tuple=True)
    R     = D[Iz,Jz]

    p = (R-dist**2)





def projIJ(x, I, J,K=torch.eye(3) ,d=6, n=10000):

    for j in range(n):

        x3 = F.conv1d(x, K.unsqueeze(-1))
        c = constraintIJ(x3, I, J, d)
        lam = dConstraintTIJ(c, x3, I, J)
        lam = F.conv_transpose1d(lam, K.unsqueeze(-1))

        print(j, torch.mean(torch.abs(c)).item() )

        with torch.no_grad():
            if j==0:
                alpha = 1.0/lam.norm()
            lsiter = 0
            while True:
                xtry = x - alpha * lam
                x3 = F.conv1d(xtry, K.unsqueeze(-1))
                ctry = constraintIJ(x3, I, J, d)
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


# Find contact map
X = X.squeeze()
D = torch.relu(torch.sum(X**2, dim=0, keepdim=True) + torch.sum(X**2, dim=0, keepdim=True).t() - \
    2 * X.t()@X)
D = D.sqrt()
D[D>8] = 0
D = D - torch.diag(torch.diag(D))

I,J = torch.nonzero(D,as_tuple = True)

X = X.unsqueeze(0)
c = constraintIJ(X, I, J, d=6)
print(c.abs().mean())

Xr = torch.rand(1,3,X.shape[2]) - 0.5
Xr = Xr/torch.sqrt(torch.sum(Xr**2, dim=1,keepdim=True)+1e-4)
#Xr[0,0,:] = torch.arange(X.shape[2])*6
Xout = projIJ(Xr, I, J, n=2000)
Xout = Xout.squeeze()
Dout = torch.relu(torch.sum(Xout**2, dim=0, keepdim=True) + torch.sum(Xout**2, dim=0, keepdim=True).t() - \
       2*Xout.t()@Xout)

Dout = Dout.sqrt()
Dout[Dout>8] = 0
Dout = Dout - torch.diag(torch.diag(Dout))


plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(D)
plt.subplot(1,2,2)
plt.imshow(Dout)


# def resnet(x,A):
#     for i in range(A.shape[0]):
#         Ai = A[i,:,:]
#         x  = x - Ai.t()@(torch.relu(Ai@x))
#     return x
#
# nL = 100
# nc = 128
# nw = 512
# ne = 1024
# A = nn.Parameter(torch.randn(nL, nc, nw))*1e-2
# B = nn.Parameter(torch.randn(nL, nc, nw))*1e-2
#
# x = torch.rand(nw,ne)
# z = torch.rand(nw,ne)
#
# Yobs = torch.randn(nw,ne)
#
# tic = time.perf_counter()
# Y = resnet(x,A)
# loss = torch.norm((Y.t()-Yobs.t()))**2
#
# loss.backward()
#
# toc = time.perf_counter()
# print(toc - tic)
#
# ns   = 17
# nr   = 17
# S    = torch.sign(torch.randn(ne,ns))
# P    = torch.sign(torch.randn(nr,nw))
# Z = resnet(z,B)
# lossa = torch.norm((P@Z@S-P@Yobs@S))**2/nr/ns
#
# lossa.backward()
#
# toc = time.perf_counter()
# print(toc - tic)
#
# print(loss.item(), lossa.item(), lossa.item()/loss.item())
#
#
