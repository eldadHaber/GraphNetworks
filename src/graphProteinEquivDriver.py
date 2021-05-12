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

Aind = AindTest
Yobs = YobsTest
MSK  = MSKTest
S    = STest


def maskMat(T,M):
    M = M.squeeze()
    MT = (M*(M*T).t()).t()
    return MT

##

print('Number of data: ', len(S))
n_data_total = len(S)



# Setup the network and its parameters
nNin = 40
nEin = 1
nopen = 16
nhid  = 16
nNclose = 3
nEclose = 1
nlayer = 18

model = GN.graphNetworkEqvrnt(nNin, nEin, nopen, nhid, nNclose, nlayer, h=.01, const=False)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####
lrO = 1e-3
lrC = 1e-3
lrN = 1e-1
lrE1 = 1e-1
lrE2 = 1e-1
lrw  = 1e-1

optimizer = optim.Adam([{'params': model.K1Nopen, 'lr': lrO},
                        {'params': model.K2Nopen, 'lr': lrO},
                        {'params': model.K1Eopen, 'lr': lrO},
                        {'params': model.K2Eopen, 'lr': lrO},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.Kw1, 'lr': lrw},
                        {'params': model.Kw2, 'lr': lrw}])


alossBest = 1e6
epochs = 700

ndata = 1 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

#dst = torch.linspace(100*3.8, 3*3.8, epochs)
dst = torch.ones(epochs)*1e6
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)
        if nodeProperties.shape[2] > 700:
            continue
        nNodes = Ds.shape[0]
        # G = GO.dense_graph(nNodes, Ds)
        w = torch.ones(I.shape,device=I.device)
        G = GO.graph(I, J, nNodes, w)
        # Organize the node data
        xn = nodeProperties
        # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xe = edgeProperties #w.unsqueeze(0).unsqueeze(0)

        #M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        CoordsOut, xnOut, xeOut = model(xn, xe, G)
        #xnOut = utils.distConstraint(xnOut)

        Dout = utils.getDistMat(CoordsOut)
        Dtrue = utils.getDistMat(Coords)

        DtrueM = maskMat(Dtrue, M)
        DoutM = maskMat(Dout, M)

        If, Jf = torch.nonzero(DtrueM < dst[j], as_tuple=True)
        DtrueM = DtrueM[If, Jf]
        DoutM = DoutM[If, Jf]

        loss = F.mse_loss(DoutM, DtrueM) / F.mse_loss(DtrueM * 0, DtrueM)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.K1Nopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.K2Nopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.K1Eopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.K2Eopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.KE1, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.KE2, 1.0e-2, norm_type=2.0)

        aloss += loss.detach()
        alossAQ += (torch.norm(DoutM - DtrueM)) / np.sqrt(torch.numel(DtrueM))
        gN = model.Kw1.grad.norm().item()
        gE1 = model.KE1.grad.norm().item()
        gE2 = model.KE2.grad.norm().item()
        gO = model.K1Nopen.grad.norm().item()
        gC = model.K2Nopen.grad.norm().item()

        optimizer.step()

        #d1 = torch.diag(maskMat(Dtrue,M),-1)
        #d1 = d1[d1 > 0.01]
        #print(' ')
        #print('Estimated noise level ', (torch.norm(d1-3.8)/torch.norm(d1)).item())
        #print(' ')

        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            c       = GN.constraint(CoordsOut)
            c       = c.abs().mean().item()
            if c>0.4:
                print('warning constraint non fulfilled ')

            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC, c), flush=True)

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

