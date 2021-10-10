import torch
from torch import nn
from hyperspn.BaseSPN import BaseSPN

class HyperSPN(BaseSPN):
    '''
        Edges parameterized by a neural net
    '''
    
    def __init__(self, xdim, latent_dim, h, N, R=1, seed=0):
        super(HyperSPN, self).__init__(xdim, N, R, seed)

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.N, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, self.N * self.N)
        )
        self.layer_embedding = nn.Parameter( torch.rand(R, 2*self.xdim, self.N) )

    def get_edges(self, r_idx, grp_idx):
        grp_embedding = self.layer_embedding[r_idx,grp_idx]
        return self.net(grp_embedding)

    def get_device(self):
        return self.layer_embedding.device
