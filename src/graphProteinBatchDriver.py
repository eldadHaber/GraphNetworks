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

lrO = 1e-2
lrC = 1e-2
lrN = 1e-2
lrE1 = 1e-2
lrE2 = 1e-2

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
batchSize = 64

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    k = ndata // batchSize
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    #with torch.autograd.detect_anomaly():
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
        # start.record()
        xnOut, xeOut = model(xn, xe, G)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time for model:", start.elapsed_time(end))
        loss = 0.0
        cnt = 0
        # start.record()
        for batch_idx, kk in enumerate(range(len(nNodes))):
            xnOuti = xnOut[:, :, cnt:cnt + nNodes[kk]]
            Coordsi = Coords[:, :, cnt:cnt + nNodes[kk]]
            Mi = M[batch_idx].squeeze()

            Mi = torch.ger(Mi, Mi)
            lossi = utils.dRMSD(xnOuti, Coordsi, Mi)
            loss += lossi
        # end.record()
        # torch.cuda.synchronize()
        # print("Time for loss:", start.elapsed_time(end))
        loss.backward()
        for param in model.parameters():
            print("param.data", torch.isfinite(param.data).all())
            print("param.grad.data", torch.isfinite(param.grad.data).all(), "\n")
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
        nextval = 1e9
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                # nVal = len(STest)
                for jj in range(nVal):
                    nodeProperties, Coords, M, IJ, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                                        MSK, 0, device=device)

                    nNodes = Ds.shape[0]
                    if dense:
                        G = GO.dense_graph(nNodes, Ds)
                        xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    else:
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
