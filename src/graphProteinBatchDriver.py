import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

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
nNin = 40
nEin = 1
nopen = 128
nhid = 128
nNclose = 3
nlayer = 6

model = GN.graphNetwork(nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False)
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
epochs = 100000

ndata = 48 #n_data_total
bestModel = model
hist = torch.zeros(epochs)
batchSize = 8

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    k = ndata // batchSize

    for i in range(k):
        #if i % 8 == 0:

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
            # print("nNodes[kk][0]:", nNodes[kk])
            xnOuti = xnOut[:, :, cnt:cnt + nNodes[kk]]
            Coordsi = Coords[:, :, cnt:cnt + nNodes[kk]]
            # print("M len:", len(M))
            Mi = M[batch_idx].squeeze()
            # print("Mi:", Mi)

            Mi = torch.ger(Mi, Mi)
            lossi = utils.dRMSD(xnOuti, Coordsi, Mi)
            loss += lossi
            cnt = cnt+nNodes[kk]

        loss.backward()

        optimizer.step()

        aloss += loss.detach()
        alossAQ += torch.sqrt(loss)

        # scheduler.step()
        nprnt = 1
        if i % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            print("%2d.%1d   %10.3E   %10.3E" % (j, i, aloss, alossAQ))
            aloss = 0.0
            alossAQ = 0.0
        # Validation
        nextval = 3
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                nVal = len(STest)
                for jj in range(nVal):
                    nodeProperties, Coords, M, I, J, edgeProperties, Ds, nNodes, w = \
                        prc.getBatchData(S, Aind, Yobs,MSK, [jj],device=device,maxlen=50000)

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
