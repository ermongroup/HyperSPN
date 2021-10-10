########################
# INFERENCE UTILS
########################

import torch
import numpy as np

def timestep_config(dataset):
    TIMESTEPS, EVAL_PERIOD = 4000, 200
    if dataset == "toy":
        TIMESTEPS, EVAL_PERIOD = 4000, 10
    if dataset in ['kdd', 'plants', 'accidents', 'ad']:
        TIMESTEPS, EVAL_PERIOD = 20000, 1000
    if dataset in ['msnbc', 'pumsb_star']:
        TIMESTEPS, EVAL_PERIOD = 80000, 1000
    if dataset[:4] == "amzn":
        TIMESTEPS, EVAL_PERIOD = 1000, 10
    if dataset in ["amzn_carseats", "amzn_gifts", "amzn_moms", "amzn_safety"]:
        TIMESTEPS, EVAL_PERIOD = 4000, 50
    return TIMESTEPS, EVAL_PERIOD

def log_density_fn(data, model):
    x = data.float().unsqueeze(1).repeat(1, 1, 1)      # (batch, 1, xdim)
    log_y = model(x)
    
    ld = torch.sum(log_y)
    return ld

def compute_parzen(test_data, samples, batch, std=3.):
    device = test_data.device
    
    llh_total = torch.Tensor(0).to(device)
    split_data = torch.split(test_data, batch)
    for batch_data in split_data:
        pairwise_dist = (batch_data.unsqueeze(1) - samples.unsqueeze(0)).norm(dim=-1)    
        m = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([std]).to(device))
        pairwise_llh = m.log_prob(pairwise_dist)

        llh = pairwise_llh.logsumexp(dim=0) - np.log(pairwise_llh.size(0))
        llh_total = torch.cat([llh_total, llh], dim=0)

    return llh_total.mean().item(), llh_total.std().item() / np.sqrt(llh_total.size(0))