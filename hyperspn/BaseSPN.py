from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import distributions as D

class BaseSPN(nn.Module, ABC):
    '''
        based on a RAT-SPN structure

        batch:           batch size
        R:               number of replicas in the RAT-SPN
        xdim:            number of dimensions of each data point
        PN:              number of nodes in each sector of the parent (current) layer
        CN:              number of nodes in each sector of the child layer
        layer_width[i]:  number of sectors in the i-th layer
    '''
    
    def __init__(self, xdim, N, R=1, seed=0):
        super(BaseSPN, self).__init__()

        self.xdim, self.N, self.R = xdim, N, R

        if seed: torch.random.manual_seed(seed)
        self.leaf_perms = torch.stack([torch.randperm( xdim ) for r in range(R) ])                                     # (R, xdim)
        self.inverse_leaf_perms = torch.stack([torch.argsort(leaf_perm) for leaf_perm in self.leaf_perms ])            # (R, xdim)

        self.layers = xdim.bit_length() + 1
        self.layer_widths = [xdim]
        while self.layer_widths[-1] != 1:
            width = self.layer_widths[-1] // 2
            self.layer_widths.append( width )
        self.layer_widths.append(1)
        
        self.layer_widths.reverse()
        assert(len(self.layer_widths) == self.layers)

        self.mix = nn.Parameter(torch.rand( R ))

    @abstractmethod 
    def get_edges(self, r_idx, grp_idx):
        # Gets the edge parameters for the sector of the RAT-SPN corresponding to replica r_idx and sector grp_idx.
        # The sector grp_idx is assigned based on a level-order traversal of the sectors.
        pass

    @abstractmethod
    def get_device(self):
        pass

    def forward(self, x):
        _, _, xdim = x.shape
        p = torch.stack( [torch.zeros(xdim, device=x.device), torch.ones(xdim, device=x.device)], dim=0)               # (2, xdim)
        log_y = D.Bernoulli(probs=p).log_prob( x )                                                                     # (batch, 2, xdim)
        log_y = log_y[:,:,self.leaf_perms]                                                                             # (batch, 2, R, xdim)
        log_y = log_y.transpose(1,2).transpose(0,1)                                                                    # (R, batch, 2, xdim)

        for i in reversed(range(self.layers - 1)):
            if i == self.layers-2:  CN = 2                                                                             # 2 children in leaf layer (true/false leaves)
            else:                   CN = self.N

            if i == 0:              PN = 1                                                                             # 1 parent in top layer (root of the spn)
            else:                   PN = self.N

            start_idx, end_idx = sum(self.layer_widths[:i+1]), sum(self.layer_widths[:i+2])
            grp_idx = torch.arange(start_idx, end_idx,device=x.device).unsqueeze(0).repeat(self.R,1)                   # (R, layer_width[i+1])
            r_idx = torch.arange(self.R,device=x.device).unsqueeze(1).repeat(1,self.layer_widths[i+1])                 # (R, layer_width[i+1])
            edges = self.get_edges(r_idx,grp_idx).reshape(self.R, self.layer_widths[i+1], self.N, self.N)
            edges = edges[:,:,:PN,:CN]                                                                                 # (R, layer_width[i+1], PN, CN)
            edges = edges - edges.logsumexp(dim=3, keepdim=True)                                                       # (R, layer_width[i+1], PN, CN)

            cross_idx = torch.arange(CN).unsqueeze(0).repeat(PN,1) 
            log_y = log_y[:, :, cross_idx, :]                                                                          # (R, batch, PN, CN, layer_width[i+1])
            log_y = log_y + edges.unsqueeze(4).transpose(1,4)
            log_y = torch.logsumexp(log_y, dim=3)                                                                      # (R, batch, PN, layer_width[i+1])

            if i > 0:
                if log_y.shape[-1] % 2:
                    log_y = torch.cat([log_y[...,:-2], log_y[...,-2:-1] + log_y[...,-1:]], dim=-1)                     # add together last 2 columns
                log_y = log_y[...,::2] + log_y[...,1::2]

        log_y = log_y[:,:,0,0]                                                                                         # (R, batch)
        mix = self.mix - self.mix.logsumexp(dim=0)
        log_y = (log_y + mix.unsqueeze(1)).logsumexp(dim=0)
        return log_y                                                                                                   # (batch)

    def sample(self, batch):
        device = self.get_device()
        
        p = torch.stack( [torch.zeros(self.xdim, device=device), torch.ones(self.xdim, device=device)], dim=0)         # (2, xdim)
        p = p.unsqueeze(0).repeat(batch, 1, 1)                                                                         # (batch, 2, xdim)
        p = p[:,:,self.leaf_perms]                                                                                     # (batch, 2, R, xdim)
        p = p.transpose(1,2).transpose(0,1)                                                                            # (R, batch, 2, xdim)
        samp = p.unsqueeze(-1)                                                                                         # (R, batch, CN=2, xdim, D=1) last dimensions will grow as we merge samps
        samp_tail = samp[...,:1,:]

        for i in reversed(range(self.layers - 1)):
            if i == self.layers-2:  CN = 2
            else:                   CN = self.N

            if i == 0:              PN = 1
            else:                   PN = self.N

            start_idx, end_idx = sum(self.layer_widths[:i+1]), sum(self.layer_widths[:i+2])
            grp_idx = torch.arange(start_idx, end_idx,device=device).unsqueeze(0).repeat(self.R,1)                     # (R, layer_width[i+1])
            r_idx = torch.arange(self.R,device=device).unsqueeze(1).repeat(1,self.layer_widths[i+1])                   # (R, layer_width[i+1])
            edges = self.get_edges(r_idx,grp_idx).reshape(self.R, self.layer_widths[i+1], self.N, self.N)
            edges = edges[:,:,:PN,:CN]                                                                                 # (R, layer_width[i+1], PN, CN)
            edges = edges - edges.logsumexp(dim=3, keepdim=True)                                                       # (R, layer_width[i+1], PN, CN)
            
            batch_edges = edges.unsqueeze(1).repeat(1,batch,1,1,1)
            samp_ch = D.categorical.Categorical(logits=batch_edges).sample()                                           # (R, batch, layer_width[i+1], PN)
            samp_ch = samp_ch.transpose(2,3)                                                                           # (R, batch, PN, layer_width[i+1])

            samp_ch_idx = samp_ch.unsqueeze(4).repeat(1,1,1,1,samp.size(4))                                            # (R, batch, PN, layer_width[i+1], D)
            samp = torch.gather(samp, 2, samp_ch_idx)
            samp_ch_idx = samp_ch.unsqueeze(4).repeat(1,1,1,1,samp_tail.size(4))                                       # (R, batch, PN, layer_width[i+1], D)
            samp_tail = torch.gather(samp_tail, 2, samp_ch_idx[...,-1:,:])
            
            if i > 0:
                if samp.shape[-2] % 2:
                    samp_tail = torch.cat([samp[...,-3:-2,:], samp[...,-2:-1,:], samp_tail], dim=-1)
                    samp = samp[...,:-1,:]
                    samp = torch.cat([samp[...,::2,:], samp[...,1::2,:]], dim=-1)                                      # concat together adjacent children
                else:
                    samp_tail = torch.cat([samp[...,-2:-1,:], samp_tail], dim=-1)
                    samp = torch.cat([samp[...,::2,:], samp[...,1::2,:]], dim=-1)                                      # concat together adjacent children

        # samp tail contains the samples
        # samp tail dim = (R, batch, 1, 1, xdim)
        
        samps = samp_tail.squeeze(2).squeeze(2)                                                                        # (R, batch, xdim)
        inv_perm_idx = self.inverse_leaf_perms.unsqueeze(1).repeat(1,batch,1)                                          # (R, batch, xdim)
        inv_perm_idx = inv_perm_idx.to(device)
        samps = torch.gather(samps, 2, inv_perm_idx)                                                                   # (R, batch, xdim) invert permutation
        
        mix = self.mix - self.mix.logsumexp(dim=0)
        mix = mix.unsqueeze(0).repeat(batch, 1)
        mix_samp = D.categorical.Categorical(logits=mix).sample()                                                      # (batch)
        samps = samps[mix_samp, torch.arange(batch)]
        
        return samps