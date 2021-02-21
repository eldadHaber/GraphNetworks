import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import utils


def getIterData(S, Aind, Yobs, MSK, i, device='cpu'):

    scale = 1e-2

    PSSM = S[i].t()
    n = PSSM.shape[1]
    M = MSK[i][:n]
    a = Aind[i]

    # X = Yobs[i][0, 0, :n, :n]
    X = Yobs[i].t()
    X = utils.linearInterp1D(X, M)
    X = torch.tensor(X)

    X = X - torch.mean(X, dim=1, keepdim=True)
    #U, Lam, V = torch.svd(X)

    Coords = scale * X  #torch.diag(Lam) @ V.t()
    Coords = Coords.type('torch.FloatTensor')

    PSSM = PSSM.type(torch.float32)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    D = torch.relu(torch.sum(Coords ** 2, dim=0, keepdim=True) + \
                   torch.sum(Coords ** 2, dim=0, keepdim=True).t() - \
                   2 * Coords.t() @ Coords)

    D = D / D.std()
    D = torch.exp(-2*D)

    nsparse = 16
    vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)
    #IJ = torch.stack([I, J], dim=1)

    #print("IJ shape:", IJ.shape)
    # Organize the edge data
    nEdges = I.shape[0]
    xe = torch.zeros(1, 1, nEdges, device=device)
    for i in range(nEdges):
        if I[i] + 1 == J[i]:
            xe[:, :, i] = 1
        if I[i] - 1 == J[i]:
            xe[:, :, i] = 1

    Seq = Seq.to(device=device, non_blocking=True)
    Coords = Coords.to(device=device, non_blocking=True)
    M = M.to(device=device, non_blocking=True)
    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)
    xe = xe.to(device=device, non_blocking=True)
    D = D.to(device=device, non_blocking=True)

    return Seq.unsqueeze(0), Coords.unsqueeze(0), M.unsqueeze(0).unsqueeze(0), I, J, xe, D

def getBatchData(S, Aind, Yobs, MSK, IND, device='cpu'):

    Seq    = torch.tensor([])
    Coords = torch.tensor([])
    I      = torch.tensor([])
    J      = torch.tensor([])
    xe     = torch.tensor([])
    w      = torch.tensor([])
    M      = []
    D      = []
    nnodes = []
    for i in range(len(IND)):
        Seqi, Coordi, Mi, Ii, Ji, xei, Di = getIterData(S, Aind, Yobs, MSK, IND[i], device)
        wi = Di[Ii,Ji]
        if Di.shape[0] < 500:
            Seq    = torch.cat((Seq,Seqi),dim=-1)
            Coords = torch.cat((Coords,Coordi),dim=-1)
            I      = torch.cat((I,Ii))
            J      = torch.cat((J, Ji))
            xe     = torch.cat((xe, xei),dim=-1)
            w      = torch.cat((w, wi),dim=-1)
            M.append([Mi])
            D.append([Di])
            nnodes.append([Di.shape[0]])
    return Seq, Coords, M, I.long(), J.long(), xe, D, nnodes, w
