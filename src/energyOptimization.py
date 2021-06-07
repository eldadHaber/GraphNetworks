import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

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
coordN     = torch.load(base_path + caspver + '/CoordN.pt')
coordAlpha = torch.load(base_path + caspver + '/CoordAlpha.pt')
coordBeta  = torch.load(base_path + caspver + '/CoordBeta.pt')

pssm    = torch.load(base_path + caspver + '/pssm.pt')
entropy = torch.load(base_path + caspver + '/entropy.pt')
seq     = torch.load(base_path + caspver + '/seq.pt')
msk     = torch.load(base_path + caspver + '/mask.pt')

def getDistMat(Coords):

    # Find neibours
    n = Coords.shape[0]
    D = torch.zeros(9,n,n)
    cnt = 0
    for j in [0,3,6]:
        for k in [0,3,6]:
            Xj = Coords[:, j:j+3]
            Xk = Coords[:, k:k+3]
            D[cnt,:,:] = torch.relu(torch.sum(Xj**2,dim=1, keepdim=True) + \
                                    torch.sum(Xk**2, dim=1, keepdim=True).t() - \
                                    2*Xj@Xk.t())
            cnt += 1

    D = D / D.std()
    D = torch.exp(-2 * D)

    return D

def getTrainingData(coordAlpha, coordBeta, coordN, seq, pssm, entropy, msk, i):
    scale = 1e-2

    PSSM = pssm[i].t()
    n = PSSM.shape[1]
    M = msk[i][:n]
    A = seq[i].t()

    X1 = coordAlpha[i].t()
    X2 = coordBeta[i].t()
    X3 = coordN[i].t()

    X1 = X1 - torch.mean(X1, dim=1, keepdim=True)
    X2 = X2 - torch.mean(X1, dim=1, keepdim=True)
    X3 = X3 - torch.mean(X1, dim=1, keepdim=True)
    Coords = scale * torch.cat((X1,X2,X3),dim=1)

    Coords = Coords.type('torch.FloatTensor')
    PSSM   = PSSM.type(torch.float32)


    nodalFeat = torch.cat((PSSM, A, entropy[i].unsqueeze(1)),dim=1)
    nodalFeat = nodalFeat.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)

    D = getDistMat(Coords)

    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)


    nsparse = 32
    vals, indices = torch.topk(torch.mean(D,dim=0), k=min(nsparse, D.shape[1]), dim=1)
    nd = D.shape[1]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)

    nEdges = I.shape[0]

    edgeFeat = torch.zeros(1, 0, nEdges, device=device)
    for k in range(PSSM.shape[1]):
        for l in range(k,PSSM.shape[1]):
            xei = torch.ger(PSSM[:,k],PSSM[:,l])
            xei = xei[I,J]
            edgeFeat  = torch.cat((edgeFeat,xei.unsqueeze(0).unsqueeze(0)),dim=1)

    for k in range(9):
        edgeFeat = torch.cat((edgeFeat, D[k,I,J].unsqueeze(0).unsqueeze(0)), dim=1)

    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)
    edgeFeat = edgeFeat.to(torch.float32)
    nodalFeat = nodalFeat.to(torch.float32)

    edgeFeat = edgeFeat.to(device=device, non_blocking=True)
    #D = D.to(device=device, non_blocking=True)

    return nodalFeat.t().unsqueeze(0), Coords, M.unsqueeze(0).unsqueeze(0), I, J, edgeFeat, D

i=1

nodalFeat, Coords, M, I, J, edgeFeat, D = getTrainingData(coordAlpha, coordBeta, coordN, seq, pssm, entropy, msk, i)

nNin    = 41
nEin    = 219
nNopen  = 32
nEopen  = 32
nNclose = 1
nNclose = 1
nEclose = 1
nlayer  = 18

model = GN.graphNetwork(nNin, nEin, nNopen, nEopen, nNclose, nEclose, nlayer, h=.1, const=False)
#model.to(device)

G = GO.graph(I,J,D.shape[1])
#xn, xe = model(nodalFeat, edgeFeat, G)