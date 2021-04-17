import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

caspver = "casp12"  # Change this to choose casp version

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

# Setup the network and its parameters
nNin = 40
nEin = 1
nNopen = 128
nEopen = 128
nEhid = 128
nNclose = 3
nEclose = 1
nlayer = 6


model = GN.graphNetwork_proteins(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.1, dense=False, varlet=True)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####

lrO = 1e-4
lrC = 1e-4
lrN = 1e-4
lrE1 = 1e-4
lrE2 = 1e-4

lrO = 1e-3
lrC = 1e-3
lrN = 1e-3
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
                        {'params': model.KNclose, 'lr': lrC}])

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)


alossBest = 1e6
epochs = 1000


def maskMat(T,M):
    M = M.squeeze()
    MT = (M*(M*T).t()).t()
    return MT

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)
ndata = 8 #len(STest)
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(STest, AindTest, YobsTest,
                                                                              MSKTest, i, device=device)

        if nodeProperties.shape[2] > 700:
            continue
        nNodes = Ds.shape[0]
        # G = GO.dense_graph(nNodes, Ds)
        w = Ds[I, J]
        G = GO.graph(I, J, nNodes, w)
        # Organize the node data
        xn = nodeProperties
        # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xe = w.unsqueeze(0).unsqueeze(0)


        #M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        ## Profiler:
        # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        #     with profiler.record_function("model_inference"):
        #         xnOut, xeOut = model(xn, xe, G)
        # print(prof.key_averages())

        xnOut, xeOut = model(xn, xe, G)
        xnOut = utils.distConstraint(xnOut, dc=3.8)
        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)

        Medge = torch.ger(M.squeeze(), M.squeeze())

        #loss = F.mse_loss(M * Dout, M * Dtrue)
        loss = F.mse_loss(maskMat(Dout, M), maskMat(Dtrue, M))
        loss.backward()

        aloss += loss.detach()
        alossAQ += (torch.norm(maskMat(Dout, M) - maskMat(Dtrue, M)) / torch.sqrt(torch.sum(Medge)).detach())
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
                    # G = GO.dense_graph(nNodes, Ds)
                    w = Ds[IJ[:, 0], IJ[:, 1]]
                    G = GO.graph(IJ[:, 0], IJ[:, 1], nNodes, w)
                    # Organize the node data
                    xn = nodeProperties
                    # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
                    xe = w.unsqueeze(0).unsqueeze(0)

                    M = torch.ger(M.squeeze(), M.squeeze())

                    xnOut, xeOut = model(xn, xe, G, Ds)

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
