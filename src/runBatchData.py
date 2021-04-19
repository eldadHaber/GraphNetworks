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

# Functions for batch training


def getBatchData(S, Aind, Yobs, MSK, IND, device=device):
    nbatch = IND.numel()
    xn, CO, M, I, J, xe, Ds = prc.getIterData(S, Aind, Yobs, MSK, IND[0], device=device)
    M = torch.ger(M.squeeze(),M.squeeze())
    N = xn.shape[-1]

    for i in range(1,nbatch):
        # Get the data
        xni, Coordsi, Mi, Ii, Ji, xei, Dsi = prc.getIterData(S, Aind, Yobs, MSK, IND[i], device=device)
        I = torch.cat((I, Ii+N))
        J = torch.cat((J, Ji+N))
        xe = torch.cat((xe, xei),dim=2)
        xn = torch.cat((xn, xni),dim=2)
        CO = torch.cat((CO, Coordsi),dim=2)
        M  = torch.block_diag(M,torch.ger(Mi.squeeze(),Mi.squeeze()))
        N = xni.shape[-1]

    return xn, CO, M, I, J, xe, Ds

##

print('Number of data: ', len(S))
n_data_total = len(S)



# Setup the network and its parameters
nNin = 40
nEin = 1
nNopen = 128
nEopen = 128
nEhid = 128
nNclose = 3
nEclose = 1
nlayer = 18

model = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.1, dense=False)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####
lrO = 1e-1
lrC = 1e-1
lrN = 1e-1
lrE1 = 1e-1
lrE2 = 1e-1

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
epochs = 500

ndata = 8 #n_data_total
hist = torch.zeros(epochs)
batchSize = 8

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata//batchSize):

        # Get the data
        IND = torch.arange(i*batchSize,(i+1)*batchSize)
        nodeProperties, Coords, M, I, J, edgeProperties, Ds = getBatchData(S, Aind, Yobs,
                                                                            MSK, IND, device=device)
        nNodes = Coords.shape[2]
        # G = GO.dense_graph(nNodes, Ds)
        G = GO.graph(I, J, nNodes)
        # Organize the node data
        xn = nodeProperties
        # edgeProperties
        xe = torch.zeros(edgeProperties.shape)

        optimizer.zero_grad()

        xnOut, xeOut = model(xn, xe, G)
        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)

        loss = F.mse_loss(M * Dout, M * Dtrue)
        #loss = F.mse_loss(maskMat(Dout,M), maskMat(Dtrue,M))

        loss.backward()

        aloss += loss.detach()
        alossAQ += (torch.norm(M * Dout - M * Dtrue) / torch.sqrt(torch.sum(M)).detach())
        gN = model.KNclose.grad.norm().item()
        gE1 = model.KE1.grad.norm().item()
        gE2 = model.KE2.grad.norm().item()
        gO = model.KN1.grad.norm().item()
        gC = model.KN2.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC), flush=True)
            aloss = 0.0
            alossAQ = 0.0
        # Validation
        nextval = 300
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                nVal = len(STest)
                for jj in range(nVal):
                    nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(STest, AindTest, YobsTest,
                                                                                        MSKTest, jj, device=device)

                    nNodes = Ds.shape[0]
                    # G = GO.dense_graph(nNodes, Ds)
                    w = Ds[I, J]
                    G = GO.graph(I, J, nNodes, w)
                    # Organize the node data
                    xn = nodeProperties
                    # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xe = w.unsqueeze(0).unsqueeze(0)

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G)

                    Dout = utils.getDistMat(xnOut)
                    Dtrue = utils.getDistMat(Coords)
                    loss = F.mse_loss(M * Dout, M * Dtrue)

                    AQdis += (torch.norm(M * Dout - M * Dtrue) / torch.sqrt(torch.sum(M))).detach()
                    misVal += loss.detach()

                print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model


##


