import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
#from torch_geometric.utils import grid

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

print('Number of data: ', len(S))
n_data_total = len(S)

# Setup the network and its parameters
nNin = 1
nEin = 1
nopen = 1
nhid = 1
nNclose = 3
nlayer = 50

batchSize = 32

model = GN.graphNetwork_try(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=True)

model.to(device)

import torch
from torch_sparse import coalesce


def grid(height, width, dtype=None, device=None):
    r"""Returns the edge indices of a two-dimensional grid graph with height
    :attr:`height` and width :attr:`width` and its node positions.
    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned position tensor.
        dtype (:obj:`torch.device`, optional): The desired device of the
            returned tensors.
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    edge_index = grid_index(height, width, device)
    pos = grid_pos(height, width, dtype, device)
    return edge_index, pos


def grid_index(height, width, device=None):
    w = width
    kernel = [-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1]
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(3, row.size(1) - 3, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, height * width, height * width)

    return edge_index


def grid_pos(height, width, dtype=None, device=None):
    dtype = torch.float if dtype is None else dtype
    x = torch.arange(width, dtype=dtype, device=device)
    y = (height - 1) - torch.arange(height, dtype=dtype, device=device)

    x = x.repeat(height)
    y = y.unsqueeze(-1).repeat(1, width).view(-1)

    return torch.stack([x, y], dim=-1)


def testImpulseResponse():
    test_index = 0
    nodeProperties, Coords, M, I, J, edgeProperties, Ds, nNodes, w = prc.getBatchData(S, Aind, Yobs,
                                                                                      MSK, [test_index], device=device)
    if 1 == 1:
        [edge_index, pos] = grid(height=32, width=32, dtype=torch.float, device=device)
        print("grid graph:", edge_index)
        print("grid graph edge index shape:", edge_index.shape)
        print("pos shape:", pos.shape)
        I = edge_index[0, :]
        print("I shape:", I.shape)
        J = edge_index[1, :]
        N = 32 * 32
        G = GO.graph(I, J, N)

        xn = torch.zeros(1, 1, 32, 32).float()
        xn[0, 0, 5:10, 5:10] = 1
        xn[0, 0, 20:23, 20:23] = 1
        xn = xn.view(1, 1, 32 * 32)
        xe = torch.ones(1, 1, edge_index.shape[1])

        xnOut, xeOut = model(xn, xe, G)
        # L = 55
        # xn = torch.zeros(1, nNin, L).to(device)
        # xn[0, :, 23] = 1
        # xe = torch.ones(1, nEin, L, L).to(device)
        #
        # G = GO.dense_graph(L).to(device)
        #
        # xnout, xeout = model(xn, xe, G)

    if 1 == 0:
        N = torch.sum(torch.tensor(nNodes))
        G = GO.graph(I, J, N, w)
        xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xn = nodeProperties
        xnOut, xeOut = model(xn, xe, G)


testImpulseResponse()
exit()
total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####

lrO = 1e-2 * batchSize
lrC = 1e-2 * batchSize
lrN = 1e-2 * batchSize
lrE1 = 1e-2 * batchSize
lrE2 = 1e-2 * batchSize

optimizer = optim.Adam([{'params': model.K1Nopen, 'lr': lrO},
                        {'params': model.K2Nopen, 'lr': lrC},
                        {'params': model.K1Eopen, 'lr': lrO},
                        {'params': model.K2Eopen, 'lr': lrC},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.KN1, 'lr': lrE1},
                        {'params': model.KN2, 'lr': lrE2},
                        {'params': model.KNclose, 'lr': lrE2}])

alossBest = 1e6
epochs = 100000

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    k = ndata // batchSize
    for i in range(k):

        IND = torch.arange(i * batchSize, (i + 1) * batchSize)
        # Get the data
        nodeProperties, Coords, M, I, J, edgeProperties, Ds, nNodes, w = prc.getBatchData(S, Aind, Yobs,
                                                                                          MSK, IND, device=device)
        N = torch.sum(torch.tensor(nNodes))
        G = GO.graph(I, J, N, w)
        xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xn = nodeProperties
        optimizer.zero_grad()
        xnOut, xeOut = model(xn, xe, G)
        loss = 0.0
        cnt = 0
        for batch_idx, kk in enumerate(range(len(nNodes))):
            xnOuti = xnOut[:, :, cnt:cnt + nNodes[kk]]
            Coordsi = Coords[:, :, cnt:cnt + nNodes[kk]]
            Mi = M[batch_idx].squeeze()
            cnt = cnt + nNodes[kk]
            if len(nNodes) > 1:
                Mi = M[batch_idx].squeeze()
            else:
                Mi = M[0].squeeze()
            Mi = torch.ger(Mi, Mi)
            lossi = utils.dRMSD(xnOuti, Coordsi, Mi)
            if torch.isnan(lossi).float().sum() > 0:
                print("Problem in instance ", batch_idx, " in batch:", i)
                print("Mi sum:", Mi.sum())
                print("Loss:", lossi)
            else:
                loss += torch.sqrt(lossi)

        loss.backward()
        optimizer.step()

        aloss += loss.detach() / batchSize
        alossAQ += loss.detach() / batchSize

        nprnt = 1
        if i % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            print("%2d.%1d   %10.3E   %10.3E" % (j, i, aloss, alossAQ), flush=True)
            aloss = 0.0
            alossAQ = 0.0

        # Test
    nextval = 1
    if (j + 1) % nextval == 0:
        with torch.no_grad():
            misVal = 0
            AQdis = 0
            nVal = len(STest)
            for jj in range(nVal):
                nodeProperties, Coords, M, I, J, edgeProperties, Ds, nNodes, w = \
                    prc.getBatchData(S, Aind, Yobs, MSK, [jj], device=device, maxlen=50000)

                N = torch.sum(torch.tensor(nNodes))
                G = GO.graph(I, J, N, w)
                xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
                xn = nodeProperties

                xnOut, xeOut = model(xn, xe, G)
                M = M[0].squeeze()
                loss = utils.dRMSD(xnOut, Coords, M)
                AQdis += torch.sqrt(loss)
                misVal += loss.detach()

            print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
            print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    scheduler.step()
