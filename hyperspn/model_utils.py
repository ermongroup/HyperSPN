########################
# MODEL UTILS
########################

import torch
from hyperspn.HyperSPN import HyperSPN
from hyperspn.SPN import SPN

def create_model(data, args, device):
    xdim = data.shape[1]

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.modeltype == 'hyperspn':
        model = HyperSPN(xdim=xdim, latent_dim=20, h=args.h, N=args.N, R=args.R).to(device)
    if args.modeltype == 'spn':
        model = SPN(xdim=xdim, N=args.N, R=args.R).to(device)

    print('params: %u' % (count_parameters(model)))

    return model

def load_model(modelpath, data, args, device):
    print("Trying to loading model from %s" % modelpath)
    model = create_model(data, args, device)
    try:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from %s" % modelpath)
    except Exception as e:
        print("Could not load model from %s: %s" % (modelpath, e))
        print("Creating new model")
    return model