import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import torch.autograd.profiler as profiler

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.utils.data as data

class Dataset_MD17(data.Dataset):
    def __init__(self, R, F, E, z):
        self.R = R
        self.F = F
        self.E = E
        self.z = z
        return

    def __getitem__(self, index):
        R = self.R[index]
        F = self.F[index]
        E = self.E[index]
        z = self.z
        return R, F, E, z

    def __len__(self):
        return len(self.R)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'



def dataloader_MD17(data,batch_size):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloader

def compute_inverse_square_distogram(r):
    D2 = torch.relu(torch.sum(r ** 2, dim=1, keepdim=True) + \
                   torch.sum(r ** 2, dim=1, keepdim=True).transpose(1,2) - \
                   2 * r.transpose(1,2) @ r)
    iD2 = 1 / D2
    tmp = iD2.diagonal(0,dim1=1,dim2=2)
    tmp[:] = 0
    return D2, iD2



def getBatchData_MD17_fast(Coords, z, use_mean_map=False, R_mean=None):
    nb = Coords.shape[0]
    nn = Coords.shape[-1]

    D2, iD2 = compute_inverse_square_distogram(Coords)
    if R_mean is not None:
        D2_mean, iD2_mean = compute_inverse_square_distogram(R_mean[None,:,:])
        iD2 = torch.abs(iD2 - iD2_mean)


    vals, J = torch.topk(iD2, k=nn-1, dim=-1)

    wiD2 = torch.gather(iD2, -1, J).view(-1)[None,None,:]
    wD2 = torch.gather(D2, -1, J).view(-1)[None,None,:]


    I = (torch.ger(torch.arange(nn), torch.ones(nn-1, dtype=torch.long))[None,:,:]).repeat(nb,1,1).to(device=z.device)
    I = I.view(nb,-1)
    J = J.view(nb,-1)
    M1 = I + 1 == J
    M2 = I - 1 == J
    M = M1 + M2

    xe = torch.zeros_like(I).to(dtype=torch.float32)
    xe[M] = 1

    one = (torch.arange(nb,device=z.device, dtype=torch.float32)*nn).repeat_interleave(nn*(nn-1))

    xe = xe.view(-1)[None,None,:]
    xn = z.view(-1)[None,None,:]
    I = I.view(-1) + one
    J = J.view(-1) + one


    # if use_mean_map:
    #     dr = h * (torch.randint(0,2,Coords.shape,device=z.device)*2-1)
    #     r1 = Coords + dr
    #     r2 = Coords - dr
    #     iD1 = compute_inverse_square_distogram(r1).view(-1)[None,None,:]
    #     iD2 = compute_inverse_square_distogram(r2).view(-1)[None,None,:]
    #     iD_diff = 2*iD - iD1 - iD2
    #     iD_stack = torch.cat((iD,iD1,iD2),dim=1)
    #     iD_var = torch.var(iD_stack,1,keepdim=True)
    #
    #     iD = torch.cat((iD,iD_diff,iD_var), dim=1)
    #     xe = xe.repeat(1,3,1)
    return I, J, xn, xe, nn*nb, wD2, wiD2

def getBatchData_MD17(Coords, device='cpu'):
    I = torch.tensor([]).to(device)
    J = torch.tensor([]).to(device)
    xe = torch.tensor([]).to(device)
    w = torch.tensor([]).to(device)
    D = []
    iD = []
    nnodes = torch.zeros(1,device=device,dtype=torch.int64)
    for i in range(Coords.shape[0]):
        Coordsi, Ii, Ji, xei, Di, iDi = getIterData_MD17(Coords[i,:], device=device)
        wi = Di[Ii, Ji]
        I = torch.cat((I, Ii))
        J = torch.cat((J, Ji))
        xe = torch.cat((xe, xei), dim=-1)
        w = torch.cat((w, wi), dim=-1)
        D.append(Di)
        iD.append(iDi)
        nnodes += Di.shape[0]
    return I.long(), J.long(), xe, D, iD, nnodes, w


def calculate_mean_coordinates(dataloader):
    R = None
    for i, (Ri, Fi, Ei, zi) in enumerate(dataloader):
        if R is None:
            R = torch.sum(Ri,dim=0)
        else:
            R += torch.sum(Ri,dim=0)

    return R / len(dataloader.dataset)



def use_model(model,dataloader,train,max_samples,optimizer,device,batch_size=1, use_mean_map=False, channels=1, R_mean=None):
    aloss = 0.0
    aloss_E = 0.0
    aloss_F = 0.0
    Fps = 0.0
    Fts = 0.0
    MAE = 0.0
    t_dataload = 0.0
    t_prepare = 0.0
    t_model = 0.0
    t_backprop = 0.0
    # if train:
    #     model.train()
    # else:
    #     model.eval()
    t3 = time.time()
    for i, (Ri, Fi, Ei, zi) in enumerate(dataloader):
        t0 = time.time()
        Ri.requires_grad_(True)
        if batch_size == 1:
            _, I, J, xe, Ds, iDs = getIterData_MD17(Ri.squeeze(), device=device)
            nnodes = Ds.shape[-1]
            w = iDs[I, J].to(device=device, dtype=torch.float32)
        else:
            I, J, xn, xe, nnodes, D2, iD2 = getBatchData_MD17_fast(Ri, zi, use_mean_map=use_mean_map,R_mean=R_mean)

        G = GO.graph(I, J, nnodes, D2)

        optimizer.zero_grad()
        t1 = time.time()
        xnOut, xeOut = model(xn, xe, G)
        t2 = time.time()

        E_pred = torch.sum(xnOut)
        if train:
            F_pred = -grad(E_pred, Ri, create_graph=True)[0].requires_grad_(True)
        else:
            F_pred = -grad(E_pred, Ri, create_graph=False)[0]
        loss_F = F.mse_loss(F_pred, Fi)
        # loss_E = F.mse_loss(E_pred, Ei)
        loss = loss_F
        Fps += torch.mean(torch.sqrt(torch.sum(F_pred.detach() ** 2, dim=1)))
        Fts += torch.mean(torch.sqrt(torch.sum(Fi ** 2, dim=1)))
        MAEi = torch.mean(torch.abs(F_pred - Fi)).detach()
        MAE += MAEi
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        # aloss_E += loss_E.detach()
        aloss_F += loss_F.detach()
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
        # gNclose = model.KNclose.grad.norm().item()
        # gE1 = model.KE1.grad.norm().item()
        # gE2 = model.KE2.grad.norm().item()
        # gN1 = model.KN1.grad.norm().item()
        # gN2 = model.KN2.grad.norm().item()
        #
        # Nclose = model.KNclose.norm().item()
        # E1 = model.KE1.norm().item()
        # E2 = model.KE2.norm().item()
        # N1 = model.KN1.norm().item()
        # N2 = model.KN2.norm().item()
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    aloss_E /= (i+1)
    aloss_F /= (i+1)
    MAE /= (i+1)
    Fps /= (i+1)
    Fts /= (i+1)
    t_dataload /= (i+1)
    t_prepare /= (i+1)
    t_model /= (i+1)
    t_backprop /= (i+1)

    return aloss,aloss_E,aloss_F,MAE,Fps,Fts, t_dataload, t_prepare, t_model, t_backprop




def getIterData_MD17(Coords, device='cpu'):
    D = torch.relu(torch.sum(Coords ** 2, dim=0, keepdim=True) + \
                   torch.sum(Coords ** 2, dim=0, keepdim=True).transpose(0,1) - \
                   2 * Coords.transpose(0,1) @ Coords)
    iD = 1 / D
    iD.fill_diagonal_(0)

    nsparse = Coords.shape[-1]
    vals, indices = torch.topk(D, k=min(nsparse, D.shape[1]), dim=-1)
    nd = D.shape[-1]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.type(torch.LongTensor)
    J = J.view(-1)

    nEdges = I.shape[-1]
    xe = torch.zeros(1, nEdges, device=device)
    for i in range(nEdges):
        if I[i] + 1 == J[i]:
            xe[:, i] = 1
        if I[i] - 1 == J[i]:
            xe[:, i] = 1

    Coords = Coords.to(device=device, non_blocking=True)
    I = I.to(device=device, non_blocking=True)
    J = J.to(device=device, non_blocking=True)
    xe = xe.to(device=device, non_blocking=True)
    D = D.to(device=device, non_blocking=True, dtype=torch.float32)
    iD = iD.to(device=device, non_blocking=True, dtype=torch.float32)

    return Coords, I, J, xe, D, iD


def print_distogram(Ds,Ei,iDs,i):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(torch.sqrt(Ds).detach(), vmin=0, vmax=9)
    plt.colorbar()
    plt.title('Distance, E={:}'.format(Ei))
    plt.subplot(1, 2, 2)
    plt.imshow(iDs.detach(), vmin=0, vmax=1.2)
    plt.colorbar()
    plt.title('Inverse square Distance')
    plt.savefig("./../results/{}.png".format(i))
    return

def print_3d_structure(fig,z,Ri,Fi):
    plt.clf()

    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    idx = z == 6
    rr = Ri.detach()
    fs = 0.05
    for ii in range(Ri.shape[0]):
        axes.plot3D([rr[ii, 0], rr[ii, 0] + Fi[ii, 0] * fs], [rr[ii, 1], rr[ii, 1] + Fi[ii, 1] * fs],
                    [rr[ii, 2], rr[ii, 2] + Fi[ii, 2] * fs], 'blue', marker='')
    axes.scatter(rr[idx, 0], rr[idx, 1], rr[idx, 2], s=120, c='black', depthshade=True)
    idx = z == 8
    axes.scatter(rr[idx, 0], rr[idx, 1], rr[idx, 2], s=160, c='red', depthshade=True)
    idx = z == 1
    axes.scatter(rr[idx, 0], rr[idx, 1], rr[idx, 2], s=20, c='green', depthshade=True)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    axes.set_zbound([-1.5, 1.5])
    carbon_patch = mpatches.Patch(color='black', label='Carbon')
    oxygen_patch = mpatches.Patch(color='red', label='Oxygen')
    hydrogen_patch = mpatches.Patch(color='green', label='Hydrogen')
    force_patch = mpatches.Patch(color='blue', label='Force')
    plt.legend(handles=[carbon_patch, oxygen_patch, hydrogen_patch, force_patch])
    axes.view_init(elev=65, azim=0)
    save = "./../results/3d_azim0_v2/{}.png".format(i)
    fig.savefig(save)
    # axes.view_init(elev=65,azim=90)
    # save = "./../results/3d_azim90_v2/{}.png".format(i)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    # axes.set_zbound([-1.5, 1.5])
    #
    # fig.savefig(save)
    # axes.view_init(elev=65,azim=45)
    # save = "./../results/3d_azim45_v2/{}.png".format(i)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    # axes.set_zbound([-1.5, 1.5])
    # fig.savefig(save)
    return
