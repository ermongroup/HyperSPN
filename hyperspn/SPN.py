import torch
from torch import nn
from hyperspn.BaseSPN import BaseSPN

class SPN(BaseSPN):
    '''
        Normal Edges
    '''
    
    def __init__(self, xdim, N, R=1, seed=0):
        super(SPN, self).__init__(xdim, N, R, seed)

        self.edges = nn.Parameter( torch.rand( self.R, self.xdim*2, self.N, self.N ) )

    def get_edges(self, r_idx, grp_idx):
        return self.edges[r_idx,grp_idx]

    def get_device(self):
        return self.edges.device