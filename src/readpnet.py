import numpy as np
import matplotlib.pyplot as plt
import torch

def weight_msa(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """

    id_min = msa_1hot.shape[1] * cutoff
    #id_mtx = np.tensordot(msa_1hot, msa_1hot, axes=([1, 2], [1, 2]))
    id_mtx = torch.tensordot(msa_1hot, msa_1hot, dims=([1, 2], [1, 2]))

    id_mask = id_mtx > id_min
    #w = 1.0 / np.sum(id_mask, axis=-1)
    w = 1.0 / torch.sum(id_mask, dim=-1)

    return w

def msa2pssm(msa_1hot, w):
    """
    Computes the position scoring matrix, f_i, given an MSA and its weight.
    Furthermore computes the sequence entropy, h_i.
    Follows the procedures used by trRossetta
    """
    #neff = np.sum(w)
    neff = torch.sum(w)

    #f_i = np.sum(w[:, None, None] * msa_1hot, axis=0) / neff + 1e-9
    f_i = torch.sum(w[:, None, None] * msa_1hot, dim=0) / neff + 1e-9

    h_i = torch.sum(- f_i * torch.log(f_i), dim=1)
    return torch.cat([f_i, h_i[:, None]], dim=1)

def dca(msa_1hot, w, penalty=4.5):
    """
    This follows the procedures used by trRossetta.
    Computes the covariance and inverse covariance matrix (equation 2), as well as the APC (equation 4).
    """
    nr, nc, ns = msa_1hot.shape
    x = msa_1hot.reshape(nr, nc * ns)

    #num_points = np.sum(w) - np.sqrt(np.mean(w))
    num_points = torch.sum(w) - torch.sqrt(torch.mean(w))

    #mean = np.sum(x * w[:, None], axis=0, keepdims=True) / num_points
    mean = torch.sum(x * w[:, None], dim=0, keepdim=True) / num_points

    #x = (x - mean) * np.sqrt(w[:, None])
    x = (x - mean) * torch.sqrt(w[:, None])

    #cov = np.matmul(x.T, x) / num_points
    cov = (x.t()@x) / num_points

    cov_reg = cov + torch.eye(nc * ns) * penalty/torch.sqrt(torch.sum(w))
    #inv_cov = np.linalg.inv(cov_reg)
    inv_cov = torch.inverse(cov_reg)

    x1 = inv_cov.reshape(nc, ns, nc, ns)
    x2 = x1.permute((0,2,1,3))
    features = x2.reshape(nc, nc, ns * ns)

    #x3 = np.sqrt(np.sum(np.square(x1[:, :-1, :, :-1]), axis=(1,3))) * (1 - np.eye(nc))
    x3 = torch.sqrt(torch.sum(torch.square(x1[:, :-1, :, :-1]), dim=(1, 3))) * (1 - torch.eye(nc))

    #apc = np.sum(x3, axis=0, keepdims=True) * np.sum(x3, axis=1, keepdims=True) / np.sum(x3)
    apc = torch.sum(x3, dim=0, keepdim=True) * torch.sum(x3, dim=1, keepdim=True) / torch.sum(x3)

    contacts = (x3 - apc) * (1 - torch.eye(nc))
    return features, contacts, x1


if __name__ == "__main__":

    a = np.load("/Users/eldadhaber/Your team Dropbox/eldad haber/ComputationalBio/data/pnet_with_msa_testing/2MQB_1_A.npz")
    print(a.files)

    msa = a['msa']
    msa_1hot = torch.tensor(np.eye(21, dtype=np.float32)[msa])

    cutoff = 0.8
    w = weight_msa(msa_1hot,cutoff)
    pssm = msa2pssm(msa_1hot, w)
    features, contacts, x1 = dca(msa_1hot, w, penalty=10)

    plt.imshow(contacts)