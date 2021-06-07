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
for i in range(81):
    print(i)
    # Get the data
    nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                            MSK, i, device=device)
    if nodeProperties.shape[2] > 700:
        continue
    nNodes = Ds.shape[0]
    w = torch.ones(I.shape,device=I.device)
    G = GO.graph(I, J, nNodes, w)
    # Organize the node data
    xn = nodeProperties
    # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
    xe = edgeProperties #w.unsqueeze(0).unsqueeze(0)


    xnOut, xeOut = model(xn, xe, G)

    #Dout = utils.getDistMat(xnOut)
    #Dtrue = utils.getDistMat(Coords)
    txt = 'CoordsPred'+str(i)
    torch.save(xnOut, txt)
    txt = 'CoordsTrue'+str(i)
    torch.save(Coords, txt)




### For saving model

for i in range(80):
    # Get the data
    nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                          MSK, i, device=device)
    if nodeProperties.shape[2] > 700:
        continue
    nNodes = Ds.shape[0]
    # G = GO.dense_graph(nNodes, Ds)
    w = torch.ones(I.shape, device=I.device)
    G = GO.graph(I, J, nNodes, w)
    # Organize the node data
    xn = nodeProperties
    # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
    xe = edgeProperties  # w.unsqueeze(0).unsqueeze(0)

    xnOut, xeOut = model(xn, xe, G)

    Dout = utils.getDistMat(xnOut)
    Dtrue = utils.getDistMat(Coords)
    M = torch.ger(M.squeeze(), M.squeeze())

    # loss = F.mse_loss(M * Dout, M * Dtrue)
    DtrueM = maskMat(Dtrue, M)
    DoutM = maskMat(Dout, M)
    If, Jf = torch.nonzero(DtrueM < 7 * 3.8, as_tuple=True)
    DtrueM = DtrueM[If, Jf]
    DoutM = DoutM[If, Jf]

    loss = F.mse_loss(DoutM, DtrueM) / F.mse_loss(DtrueM * 0, DtrueM)
    print(i, loss)

    a = 'Xc' + str(i) + '.pt'
    b = 'Xt' + str(i) + '.pt'
    m = 'Msk' + str(i) + '.pt'
    torch.save(xnOut,a)
    torch.save(Coords, b)
    torch.save(M, m)

    plt.figure(i+1)
    plt.subplot(1,2,1)
    plt.imshow(M*Dout.detach())
    plt.subplot(1, 2, 2)
    plt.imshow(M * Dtrue.detach())




### For reading the model
for i in range(5):
    a = 'Xc' + str(i) + '.pt'
    b = 'Xt' + str(i) + '.pt'
    m = 'Msk' + str(i) + '.pt'
    x1 = torch.load(a)
    x2 = torch.load(b)
    M  = torch.load(m)

    x1 = x1.squeeze()
    x2 = x2.squeeze()
    #mm = M.clone()
    #M  = torch.ger(M.squeeze(),M.squeeze())

    D1 = torch.sqrt(torch.relu(
         torch.sum(x1 ** 2, dim=0, keepdim=True) +
         torch.sum(x1 ** 2, dim=0, keepdim=True).t() - 2 * x1.t() @ x1))

    D2 = torch.sqrt(torch.relu(
         torch.sum(x2 ** 2, dim=0, keepdim=True) +
         torch.sum(x2 ** 2, dim=0, keepdim=True).t() - 2 * x2.t() @ x2))

    plt.figure(i+1)
    plt.subplot(1,2,1)
    plt.imshow(M*D1.detach())
    plt.subplot(1, 2, 2)
    plt.imshow(M * D2.detach())

    #loss_tr, r1cr, r2cr = utils.coord_loss(x1.unsqueeze(0), x2.unsqueeze(0), mm)

    print(i, torch.norm(M*(D1-D2))/torch.norm(M*D1))



############# min 0.5 |A(x)R - BR|^2