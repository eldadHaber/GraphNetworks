import time
import gzip
import numpy as np
import string
import glob
import matplotlib.pyplot as plt
import torch
from scipy.io import savemat
import torch.nn.functional as F



def read_a2m_gz_folder(folder):
    """
    This will read and return the MSAs from all a2m.gz files in the folder given.
    The MSAs will be returned as a list of numpy arrays, where each list element/numpy array corresponds to a a2m.gz file.
    Each MSA will have the shape (n x l) where n is the number of sequence alignments found, and l is the sequence length.
    The MSA will be returned as numbers ranging from 0-20, which cover the 20 common amino acids as well as '-' which contains everything else.

    The a2m.gz format is expected to be similar to the following example:

    >XXXX_UPI0000E497C4/159-301 [subseq from] XXXX_UPI0000E497C4
    --DERQKTLVENTWKTLEKNTELYGSIMFAKLTTDHPDIGKLFPFGgkNLTYgellVDPD
    VRVHGKRVIETLGSVVEDLDDmelVIQILEDLGQRHNA-YNAKKTHIIAVGGALLFTIEE
    ALGAGFTPEVKAAWAAVYNIVSDTMS----
    >XXXX_UPI0000E497C4/311-417 [subseq from] XXXX_UPI0000E497C4
    ---AREQELVQKTWGVLSLDTEQHGAAMFAKLISAHPAVAQMFPFGeNLSYsqlvQNPTL
    RAHGKRVMETIGQTVGSLDDldiLVPILRDLARRHVG-YSVTRQHFEGPKE---------
    -----------------------------
    >1A00_1_A
    VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK
    KVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPA
    VHASLDKFLASVSTVLTSKYR

    Where each line starting with ">" signals a header line, meaning that a new sequence is comming on the following lines.
    Note that the last sequence in the sequence should be the origin sequence.
    So in the above example we have 2 sequences and the origin sequence.
    Note this has only been tested on windows.
    """

    search_command = folder + "*.a2m.gz"
    a2mfiles = [f for f in glob.glob(search_command)]
    encoding = 'utf-8'
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    proteins = []
    msas = []
    for a2mfile in a2mfiles:
        seqs = []
        seq = ""
        # read file line by line
        with gzip.open(a2mfile,'r') as fin:
            for line in fin:
                text = line.decode(encoding)
                # skip labels
                if text[0] == '>':
                    if seq != "":
                        # remove lowercase letters and right whitespaces
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += text.rstrip().translate(table)
        proteins.append(seq) # A2m ends with the parent protein.
        seqs.append(seq) #We include the parent protein in the MSA sequence
        # convert letters into numbers

        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
        msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            msa[msa == alphabet[i]] = i

        # treat all unknown characters as gaps
        msa[msa > 20] = 20
        msas.append(msa)
    return msas


def speye(size,tt=torch.float32):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0,dtype=tt).expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


def weight_msa(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    id_min = msa_1hot.shape[1] * cutoff
    id_mtx = np.tensordot(msa_1hot, msa_1hot, axes=([1, 2], [1, 2]))
    id_mask = id_mtx > id_min
    w = 1.0 / np.sum(id_mask, axis=-1)

    return w


def weight_msaFast(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    T = msa_1hot.reshape((37538, 141 * 21))
    q = T@(T.transpose()@np.ones(37538))
    wq = 1/q
    wq = wq/wq.max()
    return wq

def msa2pssm(msa_1hot, w):
    """
    Computes the position scoring matrix, f_i, given an MSA and its weight.
    Furthermore computes the sequence entropy, h_i.
    Follows the procedures used by trRossetta
    """
    neff = np.sum(w)
    f_i = np.sum(w[:, None, None] * msa_1hot, axis=0) / neff + 1e-9
    h_i = np.sum(- f_i * np.log(f_i), axis=1)
    return np.concatenate([f_i, h_i[:, None]], axis=1)

def dca(msa_1hot, w, penalty=4.5):
    """
    This follows the procedures used by trRossetta.
    Computes the covariance and inverse covariance matrix (equation 2), as well as the APC (equation 4).
    """
    nr, nc, ns = msa_1hot.shape
    x = msa_1hot.reshape(nr, nc * ns)

    num_points = np.sum(w) - np.sqrt(np.mean(w))
    mean = np.sum(x * w[:, None], axis=0, keepdims=True) / num_points
    x = (x - mean) * np.sqrt(w[:, None])
    cov = np.matmul(x.T, x) / num_points
    cov_reg = cov + np.eye(nc * ns) * penalty / np.sqrt(np.sum(w))
    inv_cov = np.linalg.inv(cov_reg)

    x1 = inv_cov.reshape(nc, ns, nc, ns)
    x2 = x1.transpose((0,2,1,3))
    features = x2.reshape(nc, nc, ns * ns)

    x3 = np.sqrt(np.sum(np.square(x1[:, :-1, :, :-1]), axis=(1,3))) * (1 - np.eye(nc))
    apc = np.sum(x3, axis=0, keepdims=True) * np.sum(x3, axis=1, keepdims=True) / np.sum(x3)
    contacts = (x3 - apc) * (1 - np.eye(nc))
    return features, contacts[:, :, None]


aadic = {
    'A': 1,
    'B': 0,
    'C': 2,
    'D': 3,
    'E': 4,

    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 0,

    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'O': 0,

    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'U': 0,

    'V': 18,
    'W': 19,
    'X': 0,
    'Y': 20,
    'Z': 0,
    '-': 0,
    '*': 0,
}


def read_msa(file_path):
    lines = open(file_path).readlines()
    lines = [line.strip() for line in lines]
    n = len(lines)
    d = len(lines[0])  # CR AND LF
    msa = np.zeros([n, d], dtype=int)
    for i in range(n):
        aline = lines[i]
        for j in range(d):
            msa[i, j] = aadic[aline[j]]
    return msa


def cal_large_matrix1(msa, weight):
    # output:21*l*21*l
    ALPHA = 21
    pseudoc = 1
    M = msa.shape[0]
    N = msa.shape[1]
    pab = np.zeros((ALPHA, ALPHA))
    pa = np.ones((N, ALPHA))

    cov = np.zeros([N * ALPHA, N * ALPHA])

    #pat  = torch.ones(N, ALPHA)
    #msat = torch.tensor(msa, dtype=torch.long)
    #wt = torch.tensor(weight)
    #pat = pat[:, msat]
    #pat = pat.sum(dim=2)

    neff = weight.sum()
    for i in range(N):
        #pa[i, msa[:, i]] += weight
        for k in range(M):
            pa[i, msa[k, i]] += weight[k]

    pa /= pseudoc * ALPHA * 1.0 + neff

    # print(pab)
    for i in range(N):
        for j in range(i, N):
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if i == j:
                        if a == b:
                            pab[a, b] = pa[i, a]
                        else:
                            pab[a, b] = 0.0
                    else:
                        pab[a, b] = pseudoc * 1.0 / ALPHA
            if (i != j):
                msa1 = msa[:M, i]
                msa2 = msa[:M, j]
                neff2 = np.sum(weight[:M])
                pab[msa1, msa2] += weight[:M]

                for a in range(ALPHA):
                    for b in range(ALPHA):
                        pab[a, b] /= pseudoc * ALPHA * 1.0 + neff2
            j21 = j * 21

            for a in range(ALPHA):
                i21a = i * 21 + a
                for b in range(ALPHA):
                    if (i != j or a == b):
                        if (pab[a][b] > 0.0):
                            cov[i21a][j21 + b] = pab[a][b] - pa[i][a] * pa[j][b]
                            cov[j21 + b][i21a] = cov[i21a][j21 + b]

    return cov


if __name__ == "__main__":

    #N = 50
    #M = 10
    #msa = torch.randint(21,(M,N))
    #weight = torch.rand(10)
    #pa     = torch.ones(N,21)
    #pb     = torch.ones(N,21)

    #indxj = msa.view(-1)
    #indxi = torch.ger(torch.ones(M,dtype=torch.long),torch.arange(N)).view(-1)

    #pb = pb.reshape(N*21)

    #for i in range(N):
        #pb[i, msa[:, i]] += weight
        #pb.index_add_(1, self.iInd, weight)
        #for k in range(M):
        #    pa[i, msa[k, i]] += weight[k]

    path = "/Users/eldadhaber/Dropbox/ComputationalBio/data/raw_MSA/" # Path to a folder with a2m.gz files in it.
    msas = read_a2m_gz_folder(path)

    msa = msas[1]
    #msa = msa[:10000,:]
    msa = msa[:10000, :]

    t0 = time.time()
    msa_1hot = np.eye(21, dtype=np.float32)[msa]

    cutoff = 0.8
    w = weight_msa(msa_1hot,cutoff)
    t1 = time.time()
    pssm = msa2pssm(msa_1hot, w)
    t2 = time.time()
    features, contacts = dca(msa_1hot, w, penalty=10)
    t3 = time.time()
    print("time taken: {:2.2f}, {:2.2f}, {:2.2f}".format(t1-t0,t2-t1,t3-t2))

    CM = torch.abs(torch.tensor(contacts))
    #mx = CM.max()
    #CM = F.softshrink(torch.tensor(CM),mx/4)
    #T = torch.eye(CM.shape[0]) + torch.diag(torch.ones(CM.shape[0]-1),1) + torch.diag(torch.ones(CM.shape[0]-1),-1)
    #CM = torch.tanh(500*CM.squeeze())  + T
    #CM = torch.tanh(500 * CM)
    #CM = torch.tanh(CM.t()@CM)
    #CM = torch.tanh(CM.t() @ CM)
    #plt.imshow(CM)
    #plt.colorbar()


    t4 = time.time()
    cov = cal_large_matrix1(msa, np.ones(msa.shape[0]))
    t5 = time.time()
    print("cov time taken: {:2.2f}".format(t5 - t4))
    n = msa.shape[1]
    cov = cov.reshape(n, 21, n, 21)
    CW  = -features.reshape(n, n, 21, 21)

    plt.figure(1)
    plt.imshow(CW[:, :, 1, 1].squeeze())
    plt.colorbar()
    plt.figure(2)
    plt.imshow(cov[:, 1, :, 1].squeeze())
    plt.colorbar()