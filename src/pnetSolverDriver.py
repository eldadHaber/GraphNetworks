import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

from src import graphOps as GO
from src import processContacts as prc
from src import utils
# from src import graphNet as GN
from src import pnetArch as PNA

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

caspver = "casp11"  # Change this to choose casp version

if "s" in sys.argv:
    base_path = '/home/eliasof/pFold/data/'
    import graphOps as GO
    import processContacts as prc
    import utils
    import graphNet as GN

elif "e" in sys.argv:
    base_path = '/home/cluster/users/erant_group/pfold/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN

else:
    base_path = '../../../data/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN

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

nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs, MSK, 0, device=device)

# Setup the network and its parameters
nNin = 40
nEin = 1
nEhid = 128
nNclose = 3
nEclose = 1
nlayer = 6

model = PNA.gNet(nNin, nEin, nEhid, nNclose, nlayer)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####

lrO = 5e-3
lrC = 5e-3
lrN = 5e-3
lrE1 = 5e-3
lrE2 = 5e-3

optimizer = optim.Adam([{'params': model.K1Nopen, 'lr': lrO},
                        {'params': model.K2Nopen, 'lr': lrC},
                        {'params': model.K1Eopen, 'lr': lrO},
                        {'params': model.K2Eopen, 'lr': lrC},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.KN1, 'lr': lrE1},
                        {'params': model.KN2, 'lr': lrE2},
                        {'params': model.KNout, 'lr': lrE2}])

alossBest = 1e6
epochs = 300

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, 0, device=device)

        nNodes = Ds.shape[0]
        if nNodes < 500:
            w = Ds[IJ[:, 0], IJ[:, 1]]
            G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
            xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
            xn = nodeProperties

            M = torch.ger(M.squeeze(), M.squeeze())

            optimizer.zero_grad()

            xnOut, xeOut = model(xn, xe, G)
            loss = utils.dRMSD(xnOut, Coords, M)
            loss.backward()

            aloss += loss.detach()
            alossAQ += torch.sqrt(loss)
            gN = model.K1Nopen.grad.norm().item()
            gE1 = model.KE1.grad.norm().item()
            gE2 = model.KE2.grad.norm().item()
            gO = model.K2Nopen.grad.norm().item()
            gC = model.KNout.grad.norm().item()

            optimizer.step()
            # scheduler.step()
            nprnt = 100
            if (i + 1) % nprnt == 0:
                aloss = aloss / nprnt
                alossAQ = alossAQ / nprnt
                print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                      (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC), flush=True)
                aloss = 0.0
                alossAQ = 0.0
        # Validation
        nextval = 1e9
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                nVal = len(STest)
                for jj in range(nVal):
                    nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                                        MSK, 0, device=device)

                    nNodes = Ds.shape[0]
                    w = Ds[IJ[:, 0], IJ[:, 1]]
                    G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
                    xe = w.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xn = nodeProperties

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G, Ds)
                    loss = utils.dRMSD(xnOut, Coords, M)
                    AQdis += torch.sqrt(loss)
                    misVal += loss.detach()

                print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
                print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model
