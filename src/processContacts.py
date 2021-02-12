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
    U, Lam, V = torch.svd(X)

    Coords = scale * torch.diag(Lam) @ V.t()
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
    D = torch.exp(-D)
    Ds = F.softshrink(D, 0.92)
    Ds[Ds > 0] = 1
    IJ = torch.nonzero(Ds)

    # Organize the edge data
    nEdges = IJ.shape[0]
    xe = torch.zeros(1, 1, nEdges, device=device)
    for i in range(nEdges):
        if IJ[i, 0] + 1 == IJ[i, 1]:
            xe[:, :, i] = 1
        if IJ[i, 0] - 1 == IJ[i, 1]:
            xe[:, :, i] = 1

    Seq = Seq.to(device=device, non_blocking=True)
    Coords = Coords.to(device=device, non_blocking=True)
    M = M.to(device=device, non_blocking=True)
    IJ = IJ.to(device=device, non_blocking=True)
    xe = xe.to(device=device, non_blocking=True)
    Ds = Ds.to(device=device, non_blocking=True)

    return Seq.unsqueeze(0), Coords.unsqueeze(0), M.unsqueeze(0).unsqueeze(0), IJ, xe, D
