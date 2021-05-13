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
from torch_cluster import radius_graph

from src import graphOps as GO
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch_geometric.data import Data, DataLoader



def coordinate_loss(rp,rt):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    r_pred -> Tensors of shape (n,3d)
    r_true -> Tensors of shape (n,3d)
    '''


    #First we translate the two sets, by setting both their centroids to origin
    rpc = rp - torch.mean(rp,dim=0, keepdim=True)
    rtc = rt - torch.mean(rt,dim=0, keepdim=True)

    H = rpc.t() @ rtc
    U, S, V = torch.svd(H)

    d = torch.sign(torch.det(V @ U.t()))

    ones = torch.ones_like(d)
    a = torch.stack((ones, ones, d), dim=-1)
    tmp = torch.diag_embed(a)

    R = V @ (tmp @ U.t())

    rpcr = (R @ rpc.t()).t()

    distance_error = torch.norm(rpcr - rtc,p=2,dim=1)
    loss = torch.sum(distance_error)
    # loss = F.mse_loss(rpcr, rtc)
    return loss #, rpcr,rtc

def use_model_eq(model,dataloader,train,max_samples,optimizer,batch_size=1):
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
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (Ri, Fi, Ei, zi) in enumerate(dataloader):
        t0 = time.time()
        Ri.requires_grad_(True)
        Ri_vec = Ri.reshape(-1,Ri.shape[-1])
        zi_vec = zi.reshape(-1,zi.shape[-1])
        batch = torch.arange(Ri.shape[0]).repeat_interleave(Ri.shape[1]).to(device=Ri.device)

        data = {
                'batch': batch,
                'pos': Ri_vec,
                'z': zi_vec
                }

        optimizer.zero_grad()
        t1 = time.time()
        E_pred = model(data)
        E_pred_tot = torch.sum(E_pred)
        t2 = time.time()

        if train:
            F_pred = -grad(E_pred_tot, Ri, create_graph=True)[0].requires_grad_(True)
        else:
            F_pred = -grad(E_pred_tot, Ri, create_graph=False)[0]
        loss = F.mse_loss(F_pred, Fi)
        Fps += torch.mean(torch.sqrt(torch.sum(F_pred.detach() ** 2, dim=1)))
        Fts += torch.mean(torch.sqrt(torch.sum(Fi ** 2, dim=1)))
        MAEi = torch.mean(torch.abs(F_pred - Fi)).detach()
        MAE += MAEi
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
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

    return aloss,MAE,Fps,Fts, t_dataload, t_prepare, t_model, t_backprop

def maskMat(T,M):
    M = M.squeeze()
    MT = (M*(M*T).t()).t()
    return MT


def use_proteinmodel_eq(model,dataloader,train,max_samples,optimizer,batch_size=1, w=0):
    if train:
        model.train()
        output = use_proteinmodel_eq_inner(model, dataloader, train, max_samples, optimizer, batch_size, w)
    else:
        model.eval()
        with torch.no_grad():
            output = use_proteinmodel_eq_inner(model, dataloader, train, max_samples, optimizer, batch_size, w)
    return output

def use_proteinmodel_eq_inner(model,dataloader,train,max_samples,optimizer,batch_size, w):
    aloss = 0.0
    aloss_distogram_rel = 0.0
    aloss_coords = 0.0
    for i, (batch, seq, pssm, coords, mask, I, J, V, I_all, J_all) in enumerate(dataloader):
        print("protein length {:}".format(len(seq)))
        mask = mask.to(dtype=torch.bool)
        nb = len(torch.unique(batch))
        data = {
                'batch': batch,
                'pos': coords,
                'edge_src': J,
                'edge_dst': I,
                'edge_vec': V,
                'seq': seq,
                'pssm': pssm
                }

        optimizer.zero_grad()
        t1 = time.time()
        node_vec = model(data)

        Dout = torch.norm(node_vec[I_all] - node_vec[J_all],p=2,dim=-1).view(-1,1)
        Dtrue = torch.norm(coords[I_all]-coords[J_all],p=2,dim=-1).view(-1,1)

        If, _ = torch.nonzero(Dtrue < 7*3.8, as_tuple=True)
        Dtrue = Dtrue[If]
        Dout = Dout[If]

        # xpred=torch.tensor([[0.0,1,2,3,4,5],[0.1,0,0,0,0,-0.1],[0,0,0,0,0,0]]).t()
        # xtrue = torch.tensor([[0,0,0,0,0,0],[0.0,1,2,3,4,5],[0,0,0,0,0,0]]).t()
        # xpred_c = xpred - torch.mean(xpred, dim=0, keepdim=True)

        # loss_test,xpred_rotated,xtrue_centered = coordinate_loss(xpred, xtrue)
        # loss_coordinates = coordinate_loss(node_vec[mask], coords[mask])
        loss_coordinates = 0
        for ii in range(nb):
            idx = batch == ii
            maski = mask[idx]
            loss_i = coordinate_loss(node_vec[idx][maski], coords[idx][maski])
            loss_coordinates += loss_i

        loss_distogram = F.mse_loss(Dout, Dtrue)

        loss = (1-w)*loss_distogram + w * loss_coordinates

        loss_distogram_rel = loss_distogram / F.mse_loss(Dtrue * 0, Dtrue)

        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_distogram_rel += loss_distogram_rel.detach()
        aloss_coords += loss_coordinates.detach()
        if (i + 1) * batch_size >= max_samples:
            break
    aloss /= (i + 1)
    aloss_distogram_rel /= (i + 1)
    aloss_coords /= (i + 1)

    return aloss, aloss_distogram_rel, aloss_coords



