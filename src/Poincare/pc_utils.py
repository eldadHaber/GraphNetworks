import time
import torch
import torch.utils.data as data
import numpy as np
import random


def fix_seed(seed, include_cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Dataset_MD17_pc(data.Dataset):
    def __init__(self, Rin, Rout, z):
        self.Rin = Rin
        self.Rout = Rout
        self.z = z
        return

    def __getitem__(self, index):
        Rin = self.Rin[index]
        Rout = self.Rout[index]
        z = self.z[:,None]
        return Rin, Rout, z

    def __len__(self):
        return len(self.Rin)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'


def use_model_eq_pc(model,dataloader,train,max_samples,optimizer,batch_size=1):
    aloss = 0.0
    aloss_ref = 0.0
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
    for i, (Rin, Rout, z) in enumerate(dataloader):
        nb, natoms, nhist, ndim = Rin.shape
        t0 = time.time()
        Rin_vec = Rin.reshape(-1,Rin.shape[-1]*Rin.shape[-2])
        Rout_vec = Rout.reshape(-1,Rout.shape[-1])
        z_vec = z.reshape(-1,z.shape[-1])
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        data = {
                'batch': batch,
                'pos': Rin_vec,
                'z': z_vec
                }

        optimizer.zero_grad()
        t1 = time.time()
        Rpred = model(data)
        t2 = time.time()

        loss = torch.sum(torch.norm(Rpred-Rout_vec,p=2,dim=1))/nb
        loss_last_step = torch.sum(torch.norm(Rin[:,:,-1,:].reshape(Rout_vec.shape) - Rout_vec, p=2,dim=1))/nb
        MAEi = torch.mean(torch.abs(Rpred - Rout_vec)).detach()

        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_ref += loss_last_step
        MAE += MAEi
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    aloss_ref /= (i+1)
    MAE /= (i+1)
    t_dataload /= (i+1)
    t_prepare /= (i+1)
    t_model /= (i+1)
    t_backprop /= (i+1)

    return aloss, aloss_ref, MAE, t_dataload, t_prepare, t_model, t_backprop
