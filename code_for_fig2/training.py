import numpy as np
import torch
import torch.nn as nn

# define the model
class MF_comp(nn.Module):
    def __init__(self, n_1, n_2, r):
        super(MF_comp, self).__init__()
        self.U = nn.Embedding(n_1, r)  # Use nn.Embedding instead of nn.Parameter
        self.V = nn.Embedding(n_2, r)
        
        # Initialize embeddings (optional)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, u, i, j):
        # Get the user and item embeddings
        u_emb = self.U(u)  # Shape: (batch_size, r)
        i_emb = self.V(i)  # Shape: (batch_size, r)
        j_emb = self.V(j)  # Shape: (batch_size, r)

        # Compute score difference
        score_diff = (u_emb * i_emb).sum(dim=1) - (u_emb * j_emb).sum(dim=1)
        
        # Apply sigmoid
        return torch.sigmoid(score_diff)
    
# function to estimate error
def compute_error(model, X, device):
    with torch.no_grad():
        # Get user and item embeddings
        U = model.U.weight.detach()
        V = model.V.weight.detach()
        n_1, n_2 = X.shape
        X_hat = U @ V.t()
        # define J≜In2 −11T/(n2)
        J = torch.eye(n_2, device=device) - torch.ones(n_2, n_2, device=device) / n_2
        # compute the error
        X_diff = X - X_hat
        return torch.norm(X_diff @ J, p='fro').item()/ np.sqrt(n_1 * n_2)

# function to train the model    
def training_loop(X, model, dataloader, optimizer, criterion, n_epochs, device):
    error = []
    model.to(device)
    for epoch in range(n_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            u, i, j, z = batch
            z_pred = model(u, i, j)
            loss = criterion(z_pred, z)
            # add a regularization term: ||U^TU - V^TV||_F^2
            loss += 0.01 * (model.U.weight.t() @ model.U.weight - model.V.weight.t() @ model.V.weight).pow(2).sum()
            loss.backward()
            optimizer.step()
        error.append(compute_error(model, X, device))
        # print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Error: {error[-1]}')
    return error
    