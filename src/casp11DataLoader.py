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

    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    # Find neibours
    D = torch.relu(torch.sum(Coords ** 2, dim=1, keepdim=True) + \
                   torch.sum(Coords ** 2, dim=1, keepdim=True).t() - \
                   2 * Coords @ Coords.t())

    D = D / D.std()
    D = torch.exp(-2 * D)
    # Make sure we take the closest off diagonals
    D1 = torch.diag(torch.diag(D, 1), 1)
    D2 = torch.diag(torch.diag(D, 2), 2)
    D3 = torch.diag(torch.diag(D, 3), 3)

    D  = D - D1 - D1.t() - D2 - D2.t() - D3 - D3.t()
    e1 = torch.ones(D.shape[0]-1)
    e2 = torch.ones(D.shape[0]-2)
    e3 = torch.ones(D.shape[0]-3)

    E1 = torch.diag(e1 ,1)
    E2 = torch.diag(e2, 2)
    E3 = torch.diag(e3, 3)

    D = D + E1 + E1.t() + E2 + E2.t() + E3 + E3.t()

    nsparse = 32
    vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)

    nEdges = I.shape[0]

    edgeFeat = torch.zeros(1, 0, nEdges, device=device)
    for k in range(PSSM.shape[1]):
        for l in range(PSSM.shape[1]):
            xei = torch.ger(PSSM[:,k],PSSM[:,l])
            xei = xei[I,J]
            edgeFeat  = torch.cat((edgeFeat,xei.unsqueeze(0).unsqueeze(0)),dim=1)


    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)
    edgeFeat = edgeFeat.to(device=device, non_blocking=True)
    #D = D.to(device=device, non_blocking=True)

    return nodalFeat.unsqueeze(0), Coords.unsqueeze(0), M.unsqueeze(0).unsqueeze(0), I, J, edgeFeat


i = 20
nodalFeat, Coords, M, I, J, xe = getTrainingData(coordAlpha, coordBeta, coordN, seq, pssm, entropy, msk, i)