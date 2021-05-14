import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler
from torch.utils.data import Dataset, DataLoader

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
import prody
from collections import OrderedDict

AA_DICT = OrderedDict(
    {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
     'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
     'Y': '19', '-': '20'})
inv_AA_DICT = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
               'Y', '-']
# inv_AA_DICT = list({k for (k, v) in AA_DICT.items()})

amino_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
              'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
              'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', '-': '-'}

amino_dict = {v: k for k, v in amino_dict.items()}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Data loading
caspver = "casp11"  # Change this to choose casp version

if "s" in sys.argv:
    base_path = '/home/eliasof/pFold/data/'
    import graphOps_proteins_eldad as GO
    import proteinLoader

    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA


elif "e" in sys.argv:
    base_path = '/home/cluster/users/erant_group/pfold/'
    from src import graphOps_proteins_eldad as GO
    from src import proteinLoader

    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA


else:
    base_path = '../../../data/'
    from src import graphOps_proteins_eldad as GO
    from src import proteinLoader

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
MSK = MSKTest
S = STest

train_dataset = proteinLoader.CaspDataset(S, Aind, Yobs, MSK, device=device, return_a=False)
test_dataset = proteinLoader.CaspDataset(STest, AindTest, YobsTest, MSKTest, device=device, return_a=True)

trainLoader = dataloader = DataLoader(train_dataset, batch_size=1,
                                      shuffle=True, num_workers=6)

testLoader = dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=6)


def maskMat(T, M):
    M = M.squeeze()
    MT = (M * (M * T).t()).t()
    return MT


##

print('Number of data: ', len(S))
n_data_total = len(S)

# Setup the network and its parameters
nNin = 40
nEin = 1
nopen = 8
nhid = 16
nNclose = 3
nEclose = 1
nlayer = 18

model = GN.graphNetwork_pFold(nNin, nEin, nopen, nhid, nNclose, nlayer, h=.1, const=True)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

#### Start Training ####
lrO = 1e-3
lrC = 1e-3
lrE1 = 1e-3
lrE2 = 1e-3
lrw = 1e-3

optimizer = optim.Adam([{'params': model.K1Nopen, 'lr': lrO},
                        {'params': model.K2Nopen, 'lr': lrO},
                        # {'params': model.K1Eopen, 'lr': lrO},
                        # {'params': model.K2Eopen, 'lr': lrO},
                        {'params': model.KE1, 'lr': lrE1},
                        {'params': model.KE2, 'lr': lrE2},
                        {'params': model.KNclose, 'lr': lrC},
                        {'params': model.Kw, 'lr': lrw}])

alossBest = 1e6
epochs = 200

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)

checkpoints_path = "/home/cluster/users/erant_group/moshe/"
filename = filename = caspver + "nopen" + str(nopen) + "nhid" + str(nhid) + "nclose" + str(nNclose) + "nlayers" + str(
    nlayer)
checkpoints_path = checkpoints_path + "/" + filename + "/"
import os

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

dst = torch.linspace(100 * 3.8, 3 * 3.8, epochs) * 0 + 1e4
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    # for i in range(ndata):
    for (i, data) in enumerate(testLoader):
        # Get the data
        # nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
        #                                                                      MSK, i, device=device)

        nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(STest, AindTest, YobsTest,
                                                                              MSKTest, i, device=device)
        print("old:")
        print("nodeProperties:", nodeProperties.shape)
        print("Coords:", Coords.shape)
        print("edgeProperties:", edgeProperties.shape)

        nodeProperties, Coords, M, I, J, edgeProperties, Ds, a = data
        nodeProperties = nodeProperties.to(device)
        Coords = Coords.to(device)
        M = M.to(device).squeeze()
        I = I.to(device).squeeze()
        J = J.to(device).squeeze()
        edgeProperties = edgeProperties.to(device).squeeze()
        Ds = Ds.to(device).squeeze()

        print("new:")
        print("nodeProperties:", nodeProperties.shape)
        print("Coords:", Coords.shape)
        print("edgeProperties:", edgeProperties.shape)

        if nodeProperties.shape[2] > 700:
            continue
        nNodes = Ds.shape[0]
        # G = GO.dense_graph(nNodes, Ds)
        w = torch.ones(I.shape, device=I.device)
        G = GO.graph(I, J, nNodes, w).to(device)
        # Organize the node data
        xn = nodeProperties
        # xe = Ds.unsqueeze(0).unsqueeze(0)  # edgeProperties
        xe = edgeProperties  # w.unsqueeze(0).unsqueeze(0)

        # M = torch.ger(M.squeeze(), M.squeeze())

        optimizer.zero_grad()

        xnOut, xeOut = model(xn, xe, G)
        # xnOut = utils.distConstraint(xnOut)

        Dout = utils.getDistMat(xnOut)
        Dtrue = utils.getDistMat(Coords)
        W = 1 / torch.sqrt(Dtrue + 2)

        # loss = F.mse_loss(M * Dout, M * Dtrue)
        # dm    = Dtrue.max()
        # Dtrue = torch.exp(-sigma[j] * Dtrue/Dtrue.max())
        # Dout  = torch.exp(-sigma[j] * Dout/Dtrue.max())
        DtrueM = maskMat(W * Dtrue, M)
        DoutM = maskMat(W * Dout, M)

        If, Jf = torch.nonzero(DtrueM < dst[j], as_tuple=True)
        DtrueM = DtrueM[If, Jf]
        DoutM = DoutM[If, Jf]

        loss = F.mse_loss(DoutM, DtrueM) / F.mse_loss(DtrueM * 0, DtrueM)

        loss.backward()

        gN = model.KNclose.grad.norm().item()
        gE1 = model.KE1.grad.norm().item()
        gE2 = model.KE2.grad.norm().item()
        gO = model.K1Nopen.grad.norm().item()
        gC = model.K2Nopen.grad.norm().item()
        gw = model.Kw.grad.norm().item()

        torch.nn.utils.clip_grad_norm_(model.K1Nopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.K2Nopen, 1.0e-2, norm_type=2.0)
        # torch.nn.utils.clip_grad_norm_(model.K1Eopen, 1.0e-2, norm_type=2.0)
        # torch.nn.utils.clip_grad_norm_(model.K2Eopen, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.KE1, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.KE2, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.KNclose, 1.0e-2, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(model.Kw, 1.0e-2, norm_type=2.0)

        aloss += loss.detach()
        alossAQ += (torch.norm(DoutM - DtrueM)) / np.sqrt(torch.numel(DtrueM))

        optimizer.step()

        # d1 = torch.diag(maskMat(Dtrue,M),-1)
        # d1 = d1[d1 > 0.01]
        # print(' ')
        # print('Estimated noise level ', (torch.norm(d1-3.8)/torch.norm(d1)).item())
        # print(' ')

        # scheduler.step()
        nprnt = 10
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            alossAQ = alossAQ / nprnt
            c = GN.constraint(xnOut)
            c = c.abs().mean().item()
            if c > 0.4:
                print('warning constraint non fulfilled ')

            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, aloss, alossAQ, gO, gN, gE1, gE2, gC, gw, c), flush=True)

            aloss = 0.0
            alossAQ = 0.0
    # Validation
    nextval = 1
    if (j + 1) % nextval == 0:
        with torch.no_grad():
            aloss = 0
            AQdis = 0
            nVal = len(STest)
            for jj in range(nVal):
                nodeProperties, Coords, M, I, J, edgeProperties, Ds, a = prc.getIterData(STest, AindTest, YobsTest,
                                                                                         MSKTest, jj, device=device,
                                                                                         return_a=True)
                if nodeProperties.shape[2] > 700:
                    continue
                nNodes = Ds.shape[0]
                w = Ds[I, J]
                G = GO.graph(I, J, nNodes, w)
                xn = nodeProperties
                xe = w.unsqueeze(0).unsqueeze(0)

                xnOut, xeOut = model(xn, xe, G)
                # xnOut = utils.distConstraint(xnOut, dc=3.8)
                Dout = utils.getDistMat(xnOut)
                Dtrue = utils.getDistMat(Coords)

                Medge = torch.ger(M.squeeze(), M.squeeze())
                Medge = Medge > 0
                # loss = F.mse_loss(M * Dout, M * Dtrue)
                # loss = F.mse_loss(maskMat(Dout, M), maskMat(Dtrue, M))

                n = xnOut.shape[-1]
                Xl = torch.zeros(3, n, device=xnOut.device)
                Xl[0, :] = 3.9 * torch.arange(0, n)
                Dl = torch.sum(Xl ** 2, dim=0, keepdim=True) + torch.sum(Xl ** 2, dim=0,
                                                                         keepdim=True).t() - 2 * Xl.t() @ Xl
                Dl = torch.sqrt(torch.relu(Dl))
                ML = (Medge * Dl - Medge * (torch.relu(Dtrue))) > 0
                MS = (torch.relu(Dtrue)) < 7 * 3.9
                Medge = (Medge & MS & ML) * 1.0
                Medge = torch.triu(Medge, 1)
                R = torch.triu(Dout - (torch.relu(Dtrue)), 1)
                loss = torch.norm(Medge * R) ** 2 / torch.sum(Medge)
                loss = torch.sqrt(loss)

                aloss += loss.detach()
                AQdis += (torch.norm(maskMat(Dout, M) - maskMat(Dtrue, M)) / torch.sqrt(
                    torch.sum(Medge)).detach())

                if 1 == 1:
                    known_idx = (M == 1).squeeze()
                    gt_coords = Coords.clone().squeeze().t().detach().cpu().numpy()
                    if j == 0:
                        ##SAVE GT FILE
                        ind = a.clone()[known_idx].detach().cpu().numpy().astype(int)
                        # print("ind:", ind)
                        atoms = [inv_AA_DICT[i] for i in ind]
                        # print("atoms:", atoms)
                        atoms_group = prody.AtomGroup('prot' + str(jj))
                        gt_coords = Coords.clone().squeeze()[:, known_idx].t().detach().cpu().numpy()

                        # print("coords shape:", gt_coords.shape)

                        chids = len(atoms) * ['CA']
                        # atoms_group.setChids(chids)
                        atoms_group.setNames(chids)
                        # print("len(atoms):", len(atoms))
                        atoms_group.setResnums(range(1, len(atoms) + 1))

                        res_names = [amino_dict[i] for i in atoms]
                        # print("res_names:", res_names)
                        # exit()
                        atoms_group.setResnames(res_names)
                        labels = len(atoms) * ['ATOM']
                        atoms_group.setCoords(gt_coords, label=labels)

                        prody.writePDB(checkpoints_path + 'gt_prot' + str(jj) + '.pdb', atoms_group)

                    ind = a.clone()[known_idx].detach().cpu().numpy().astype(int)
                    atoms = [inv_AA_DICT[i] for i in ind]
                    atoms_group = prody.AtomGroup('prot' + str(jj))
                    # print("pred coords shape:", xnOut.shape)
                    pred_coords = xnOut.clone().squeeze()[:, known_idx].t().detach().cpu().numpy()

                    chids = len(atoms) * ['CA']
                    atoms_group.setNames(chids)
                    atoms_group.setResnums(range(1, len(atoms) + 1))

                    res_names = [amino_dict[iii] for iii in atoms]
                    atoms_group.setResnames(res_names)
                    labels = len(atoms) * ['ATOM']
                    atoms_group.setCoords(pred_coords, label=labels)

                    prody.writePDB(checkpoints_path + 'epoch' + str(j) + '_pred_prot' + str(jj) + '.pdb',
                                   atoms_group)

                    fname = checkpoints_path + 'epoch' + str(j) + '_pred_protCoords' + str(jj) + '.txt'

                    np.savetxt(fname, pred_coords)
                    if (jj < 1000) and 1 == 1:
                        gtC = Coords.clone().squeeze()[:, known_idx].squeeze()
                        if j == 0:
                            distMap = utils.getDistMat(gtC)  # * M
                            distMap = distMap.cpu().numpy()
                            # indices_bad = distMap > 26.6
                            # distMap[distMap > 26.6] = 0
                            plt.figure()
                            plt.imshow(distMap)
                            # plt.clim(0, 26.6)
                            plt.colorbar()
                            plt.axis('off')
                            plt.savefig(
                                checkpoints_path + '_epoch_' + str(
                                    j) + 'distmap_gt_' + str(jj) + ".jpg")
                            plt.close()
                            plt.cla()

                        ##Save distmap:
                        # predC = Cout.clone().squeeze()
                        predC = torch.from_numpy(pred_coords).t()
                        distMap = utils.getDistMat(predC)  # * M
                        distMap = distMap.cpu().numpy()
                        # distMap[indices_bad] = 0
                        plt.figure()
                        plt.imshow(distMap)
                        # plt.clim(0, 26.6)

                        plt.colorbar()
                        plt.axis('off')
                        plt.savefig(
                            checkpoints_path + '_epoch_' + str(
                                j) + 'distmap_pred_' + str(jj) + ".jpg")
                        plt.close()
                        plt.cla()

            print("%2d       %10.3E   %10.3E" % (j, aloss / nVal, AQdis / nVal))
            print('===============================================')

    if aloss < alossBest:
        alossBest = aloss
        bestModel = model
