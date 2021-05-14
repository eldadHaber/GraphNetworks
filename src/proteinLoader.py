import torch
from torch.utils.data import Dataset, DataLoader
try:
    from src import utils
except:
    import utils
class CaspDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, S, Aind, Yobs, MSK, device='cpu', return_a=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.S = S
        self.Aind = Aind
        self.Yobs = Yobs
        self.MSK = MSK
        self.device = device
        self.return_a = return_a

    def __len__(self):
        return len(self.S)

    def __getitem__(self, i):
        scale = 1e-2

        PSSM = self.S[i].t()
        n = PSSM.shape[1]
        M = self.MSK[i][:n]
        a = self.Aind[i]

        # X = Yobs[i][0, 0, :n, :n]
        X = self.Yobs[i].t()
        X = utils.linearInterp1D(X, M)
        X = torch.tensor(X)

        X = X - torch.mean(X, dim=1, keepdim=True)
        # U, Lam, V = torch.svd(X)

        Coords = scale * X  # torch.diag(Lam) @ V.t()
        Coords = Coords.type('torch.FloatTensor')

        PSSM = PSSM.type(torch.float32)

        A = torch.zeros(20, n)
        A[a, torch.arange(0, n)] = 1.0
        Seq = torch.cat((PSSM, A))
        #Seq = Seq.to(device=self.device)

        Coords = Coords.to(device=self.device)
        M = M.type('torch.FloatTensor')
        #M = M.to(device=self.device)

        D = torch.relu(torch.sum(Coords ** 2, dim=0, keepdim=True) + \
                       torch.sum(Coords ** 2, dim=0, keepdim=True).t() - \
                       2 * Coords.t() @ Coords)

        D = D / D.std()
        D = torch.exp(-2 * D)

        nsparse = 16
        vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
        nd = D.shape[0]
        I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
        I = I.view(-1)
        J = indices.view(-1).type(torch.LongTensor)
        # IJ = torch.stack([I, J], dim=1)

        # print("IJ shape:", IJ.shape)
        # Organize the edge data
        nEdges = I.shape[0]
        xe = torch.zeros(1, 1, nEdges) # device=self.device
        for i in range(nEdges):
            if I[i] + 1 == J[i]:
                xe[:, :, i] = 1
            if I[i] - 1 == J[i]:
                xe[:, :, i] = 1

        #Seq = Seq.to(device=self.device, non_blocking=True)
        #Coords = Coords.to(device=self.device, non_blocking=True)
        #M = M.to(device=self.device, non_blocking=True)
        #I = I.to(device=self.device, non_blocking=True)
        #J = J.to(device=self.device, non_blocking=True)
        #xe = xe.to(device=self.device, non_blocking=True)
        #D = D.to(device=self.device, non_blocking=True)
        if self.return_a:
            return Seq.unsqueeze(0), Coords.unsqueeze(0), M.unsqueeze(0).unsqueeze(0), I, J, xe, D, a

        return Seq.unsqueeze(0), Coords.unsqueeze(0), M.unsqueeze(0).unsqueeze(0), I, J, xe, D