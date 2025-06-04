import torch
from torch.utils.data import DataLoader, Dataset, Subset

class TripletDataset(Dataset):
    def __init__(self, X, m, n_1, n_2, device):
        self.m = m
        self.X = X

        # Generate all random indices at once (vectorized)
        self.u = torch.randint(0, n_1, (m,), dtype = torch.long, device=device)
        self.i = torch.randint(0, n_2, (m,), dtype = torch.long, device=device)
        self.j = torch.randint(0, n_2, (m,), dtype = torch.long, device=device)

        # Compute score differences and apply sigmoid
        scorediff = X[self.u, self.i] - X[self.u, self.j]
        self.z = torch.sigmoid(scorediff)

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        return (
            self.u[idx],
            self.i[idx],
            self.j[idx],
            self.z[idx], 
        )        


def generate_X(n_1, n_2, r, e, device):
    A = torch.randn(n_1, n_2).to(device)
    U, S, V = torch.svd(A)
    S[r:] *= e
    X = U @ torch.diag(S) @ V.t()
    return X

def generate_data(n_1, n_2, r, e, m, device):
    X = generate_X(n_1, n_2, r, e, device)
    dataset = TripletDataset(X, m, n_1, n_2, device)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    return X, dataloader

def generate_data_with_thinning(n_1, n_2, r, e, m_range, device):
    X = generate_X(n_1, n_2, r, e, device)
    m = max(m_range)
    dataset = TripletDataset(X, m, n_1, n_2, device)
    dataloaders = {}
    for m_small in m_range:
        subset = Subset(dataset, range(m_small))
        try:
            m_small == len(subset)
        except:
            print('Error: m_small is not equal to the length of the subset')
            raise ValueError
        dataloaders[m_small] = DataLoader(subset, batch_size=100, shuffle=True)
    return X, dataloaders