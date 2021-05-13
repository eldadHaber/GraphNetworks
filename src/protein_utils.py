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
from src import utils
from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch_cluster import radius_graph
from src.Equivariant.eq_utils import coordinate_loss
from src.utils import plot_protein_comparison


def use_proteinmodel(model,dataloader,train,max_samples,optimizer, w, batch_size=1, debug=False, viz=False, epoch=0):
    aloss = 0.0
    aloss_distogram_rel = 0.0
    aloss_distogram = 0.0
    aloss_coords = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (batch, seq, seq_1hot, pssm, coords, coords_init, mask, I, J, V, I_all, J_all) in enumerate(dataloader):
        mask = mask.to(dtype=torch.bool)
        nb = len(torch.unique(batch))

        optimizer.zero_grad()
        node_attr = torch.cat([pssm,seq_1hot],dim=1)
        t0 = time.time()
        # with profiler.profile(record_shapes=True) as prof:
        #     with profiler.record_function("model_inference"):
        node_vec = model(input=coords_init,xn_attr=node_attr)
        t1 = time.time()

        Dout = torch.norm(node_vec[I_all] - node_vec[J_all],p=2,dim=-1).view(-1,1)
        Dtrue = torch.norm(coords[I_all]-coords[J_all],p=2,dim=-1).view(-1,1)

        If, _ = torch.nonzero(Dtrue < 7*3.8, as_tuple=True)
        Dtrue = Dtrue[If]
        Dout = Dout[If]
        loss_coordinates = 0
        for i in range(nb):
            idx = batch == i
            maski = mask[idx]
            loss_i, coords_pred, coords_true = coordinate_loss(node_vec[idx][maski], coords[idx][maski])
            loss_coordinates += loss_i

        loss_distogram = F.mse_loss(Dout, Dtrue)

        loss = (1-w)*loss_distogram + w * loss_coordinates

        loss_distogram_rel = loss_distogram / F.mse_loss(Dtrue * 0, Dtrue)

        t2 = time.time()
        # with profiler.record_function("backward"):
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_distogram += loss_distogram.detach()
        aloss_distogram_rel += loss_distogram_rel.detach()
        aloss_coords += loss_coordinates.detach()
        if debug:
            print(f"dataloader:{t0-t3:2.4f}, model:{t1-t0:2.4f}, loss:{t2-t1:2.4f}, backward:{time.time()-t2:2.4f}")
        if viz:
            plot_protein_comparison(coords_pred.cpu().detach(),coords_true.cpu().detach(),"protein_{:}".format(epoch))
        t3 = time.time()
        if (i + 1) * batch_size >= max_samples:
            break
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    aloss /= (i + 1)
    aloss_distogram_rel /= (i + 1)
    aloss_coords /= (i + 1)
    aloss_distogram /= (i + 1)

    return aloss, aloss_distogram, aloss_distogram_rel, aloss_coords





def compute_all_connections(n,mask=None, include_self_connections=False, device='cpu'):
    tmp = torch.arange(n, device=device)
    if mask is not None:
        tmp = tmp[mask]
    I = tmp.repeat_interleave(len(tmp))
    J = tmp.repeat(len(tmp))
    if not include_self_connections:
        m = I != J
        I = I[m]
        J = J[m]
    return I,J



class GraphCollate:
    """
    """
    def __init__(self):
        return

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_collate(self, data):
        """
        This functions collates our data and transforms it into torch format. numpy arrays are padded according to the longest sequence in the batch in all dimensions.
        The padding mask is created according to mask_var and mask_dim, and is appended as the last variable in the output.
        args:
            data - Tuple of length nb, where nb is the batch size.
            data[0] contains the first batch and will also be a tuple with length nv, equal to the number of variables in a batch.
            data[0][0] contains the first variable of the first batch, this should also be a tuple with length nsm equal to the number of samples in the variable, like R1,R2,R3 inside coords.
            data[0][0][0] contains the actual data, and should be a numpy array.
            If any numpy array of ndim=0 is encountered it is assumed to be a string object, in which case it is turned into a string rather than a torch object.
            The datatype of the output is inferred from the input.
        return:
            A tuple of variables containing the input variables in order followed by the mask.
            Each variable is itself a tuple of samples
        """
        # find longest sequence
        nb = len(data)    # Number of batches
        nv = len(data[0]) # Number of variables in each batch

        seqs, seqs_1hot, pssms, coords, coords_init, masks,Is,Js,Vs = zip(*data)
        batchs = [i+0*datai[0] for i,datai in enumerate(data)]

        batch = torch.cat(batchs)
        seq = torch.cat(seqs)
        seq_1hot = torch.cat(seqs_1hot,dim=0)

        pssm = torch.cat(pssms,dim=0)
        coord = torch.cat(coords, dim=0)
        coords_init = torch.cat(coords_init, dim=0)
        mask = torch.cat(masks,dim=0)

        n_nodes = torch.tensor([cc.shape[0] for cc in coords], device=seq.device)
        n_edges = torch.tensor([len(cc) for cc in Is], device=seq.device)


        pp = torch.cumsum(n_nodes,dim=0)
        index_shift = torch.zeros((nb),dtype=torch.int64, device=seq.device)
        index_shift[1:] = pp[:-1]

        I = torch.cat(Is,dim=0)
        index_shift_vec = index_shift.repeat_interleave(n_edges)
        Ishift = I + index_shift_vec

        J = torch.cat(Js,dim=0)
        Jshift = J + index_shift_vec

        V = torch.cat(Vs,dim=0)

        I_all = []
        J_all = []
        for maski, n_node,i_shift in zip(masks,n_nodes,index_shift):
            ii, jj = compute_all_connections(n_node, mask=maski.to(dtype=torch.bool), device=seq.device)
            I_all.append(ii+i_shift)
            J_all.append(jj+i_shift)

        I_all = torch.cat(I_all)
        J_all = torch.cat(J_all)

        return batch, seq, seq_1hot, pssm, coord, coords_init, mask, Ishift, Jshift, V, I_all, J_all

class Dataset_protein(data.Dataset):
    def __init__(self, seq, coords, mask, pssm, device):
        self.scale = 1e-2
        self.seq = seq
        self.coords = coords
        self.mask = mask
        self.pssm = pssm
        self.device = device
        return

    def __getitem__(self, index):
        pssm = self.pssm[index]
        seq = self.seq[index]
        coords = (self.coords[index]*self.scale) #This gives coordinates in Angstrom, with a typical amino acid binding distance of 3.8 A
        mask = self.mask[index]
        n = seq.shape[0]
        seq_1hot = F.one_hot(seq, num_classes=20)
        # node_features = torch.cat((pssm,seq_1hot),dim=1)

        D = torch.relu(torch.sum(coords.t() ** 2, dim=0, keepdim=True) + \
                       torch.sum(coords.t() ** 2, dim=0, keepdim=True).t() - \
                       2 * coords @ coords.t())
        D2 = D / D.std()
        D2 = torch.exp(-2 * D2)

        nsparse = 16
        vals, indices = torch.topk(D2, k=min(nsparse, D.shape[0]), dim=1)
        indices_ext = torch.empty((n, nsparse + 4), dtype=torch.int64)
        indices_ext[:, :16] = indices
        indices_ext[:, 16] = torch.arange(n) - 1
        indices_ext[:, 17] = torch.arange(n) - 2
        indices_ext[:, 18] = torch.arange(n) + 1
        indices_ext[:, 19] = torch.arange(n) + 2
        nd = D.shape[0]
        I = torch.ger(torch.arange(nd), torch.ones(nsparse + 4, dtype=torch.long))
        I = I.view(-1)
        J = indices_ext.view(-1).type(torch.LongTensor)

        IJ = torch.stack([I, J], dim=1)
        IJ_unique = torch.unique(IJ, dim=0)
        I = IJ_unique[:, 0]
        J = IJ_unique[:, 1]
        M1 = torch.sum(IJ_unique < 0, dim=1).to(dtype=torch.bool)
        M2 = torch.sum(IJ_unique > nd - 1, dim=1).to(dtype=torch.bool)
        MM = ~ (M1 + M2)
        I = I[MM]
        J = J[MM]

        V = torch.zeros((I.shape[0],3),dtype=torch.float32)
        V[::3,0] = math.sqrt(3)
        V[1::3,1] = math.sqrt(3)
        V[2::3,2] = math.sqrt(3)
        # V = torch.cumsum(V,0)

        nn_dist = 3.8 #Angstrom
        coords_init = compute_spherical_coords_init(n, nn_dist)
        # coords_init = torch.zeros((n,3),dtype=torch.float32)
        # coords_init[::3,0] = math.sqrt(3)
        # coords_init[1::3,1] = math.sqrt(3)
        # coords_init[2::3,2] = math.sqrt(3)
        # coords_init = torch.cumsum(coords_init,0)

        return seq.to(device=self.device), seq_1hot.to(device=self.device), pssm.to(device=self.device,dtype=torch.float32), coords.to(device=self.device,dtype=torch.float32), coords_init.to(device=self.device), mask.to(device=self.device), I.to(device=self.device), J.to(device=self.device), V.to(device=self.device)

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'


def compute_spherical_coords_init(n,nn_dist):
    dist = n*nn_dist
    radius = dist/math.pi
    radians = torch.arange(n) * (math.pi / n)
    x = radius * torch.cos(radians)
    y = radius * torch.sin(radians)
    z = x*0
    coords_init = torch.cat([x[:,None],y[:,None],z[:,None]],dim=1)
    return coords_init.to(dtype=torch.float32)

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
        I, J, xn, xe, nnodes, D2, iD2 = getBatchData_MD17_fast(Ri, zi, use_mean_map=use_mean_map,R_mean=R_mean)

        G = GO.graph(I, J, nnodes, D2)

        optimizer.zero_grad()
        t1 = time.time()
        xnOut, xeOut = model(xn, xe, G)
        t2 = time.time()

        # E_1 = torch.sum(xnOut[:,:,:21])
        # F1 = -grad(E_1, Ri, create_graph=True)[0].requires_grad_(True)

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
