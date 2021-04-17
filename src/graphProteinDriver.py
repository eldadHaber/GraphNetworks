import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler
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

caspver = "casp12"  # Change this to choose casp version
checkpoints_path = "/home/cluster/users/erant_group/moshe/"
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


model = GN.graphNetwork_proteins(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=0.5, dense=False, varlet=True)
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

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


alossBest = 1e6
epochs = 1000


def maskMat(T,M):
    M = M.squeeze()
    MT = (M*(M*T).t()).t()
    return MT

ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)
#ndata = 8 #len(STest)
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    alossAQ = 0.0
    optimizer.zero_grad()

    for i in range(ndata):

        # Get the data
        nodeProperties, Coords, M, I, J, edgeProperties, Ds = prc.getIterData(S, Aind, Yobs,
                                                                              MSK, i, device=device)

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

        if i%4 == 3:
            optimizer.step()
            optimizer.zero_grad()

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
        nextval = 10000
        if (i + 1) % nextval == 0:
            with torch.no_grad():
                aloss = 0
                AQdis = 0
                nVal = len(STest)
                for jj in range(nVal):
                    nodeProperties, Coords, M, I, J, edgeProperties, Ds, a = prc.getIterData(STest, AindTest, YobsTest,
                                                                                          MSKTest, jj, device=device, return_a=True)
                    if nodeProperties.shape[2] > 700:
                        continue
                    nNodes = Ds.shape[0]
                    w = Ds[I, J]
                    G = GO.graph(I, J, nNodes, w)
                    xn = nodeProperties
                    xe = w.unsqueeze(0).unsqueeze(0)

                    xnOut, xeOut = model(xn, xe, G)
                    xnOut = utils.distConstraint(xnOut, dc=3.8)
                    Dout = utils.getDistMat(xnOut)
                    Dtrue = utils.getDistMat(Coords)

                    Medge = torch.ger(M.squeeze(), M.squeeze())

                    # loss = F.mse_loss(M * Dout, M * Dtrue)
                    loss = F.mse_loss(maskMat(Dout, M), maskMat(Dtrue, M))
                    aloss += loss.detach()
                    AQdis += (torch.norm(maskMat(Dout, M) - maskMat(Dtrue, M)) / torch.sqrt(
                        torch.sum(Medge)).detach())

                    if 1 == 1:
                        known_idx = M == 1
                        gt_coords = Coords.clone().t().detach().cpu().numpy()
                        if j == 0:
                            ##SAVE GT FILE
                            ind = a.clone()[known_idx].detach().cpu().numpy().astype(int)
                            # print("ind:", ind)
                            atoms = [inv_AA_DICT[i] for i in ind]
                            # print("atoms:", atoms)
                            atoms_group = prody.AtomGroup('prot' + str(jj))
                            gt_coords = Coords.clone()[:, known_idx].t().detach().cpu().numpy()

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
                        pred_coords = xnOut.clone()[:, known_idx].t().detach().cpu().numpy()

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
                            gtC = Coords.clone()[:, known_idx].squeeze()
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
