import numpy as np
import torch
from torch import optim
import argparse
import csv

from hyperspn.dataset_utils import load_dataset
from hyperspn.model_utils import load_model
from hyperspn.inference_utils import log_density_fn, compute_parzen, timestep_config

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device: ", DEVICE)
print("torch version: ", torch.__version__)

def run(train_data, valid_data, test_data, model, device):
    train_data = torch.tensor(train_data).float().to(device)
    valid_data = torch.tensor(valid_data).float().to(device)
    test_data = torch.tensor(test_data).float().to(device)

    def eval_model(it):
        with torch.no_grad():
            avg_train, avg_valid, avg_test = 0.0, 0.0, 0.0
            
            # evaluate on full dataset
            def get_avg(data):    
                avg = 0.0
                split_data = torch.split(data, ARGS.batch)
                for batch_data in split_data:
                    ld = log_density_fn(batch_data, model).item()
                    avg += ld
                avg = avg / data.shape[0]
                return avg
            
            avg_train = get_avg(train_data)
            avg_valid = get_avg(valid_data)
            avg_test = get_avg(test_data)

            print('step: %u, train-all: %f, valid-all: %f, test-all: %f' % (it, avg_train, avg_valid, avg_test) , flush=True)

        samples = model.sample(batch=ARGS.batch)
        avg_llh, std_llh = compute_parzen(test_data, samples, batch=ARGS.batch)
        print("parzen        : %.3f %.3f" % (avg_llh, std_llh))

        return avg_train, avg_valid, avg_test, avg_llh

    def sample_batch(data):
        batch_indices = np.random.choice(data.shape[0], size=min(data.shape[0],ARGS.batch), replace=False)
        return data[batch_indices]


    weight_decay = 0.0
    if ARGS.wd: weight_decay = ARGS.wd
    optimizer = optim.Adam( list(model.parameters()) , lr=ARGS.lr, weight_decay=weight_decay)

    infos = []
    TIMESTEPS, EVAL_PERIOD = timestep_config(ARGS.dataset)

    if ARGS.eval:
        avg_train, avg_valid, avg_test, avg_llh = eval_model(0)
        infos.append( (0, avg_train, avg_valid, avg_test, avg_llh) )
        return model, infos

    for i in range(TIMESTEPS+1):
        batch_train_data = sample_batch(train_data)

        optimizer.zero_grad()
        ld = log_density_fn(batch_train_data, model)
        (-ld).backward()
        optimizer.step()

        # log current progress
        if i % 10 == 0:
            batch_valid_data = sample_batch(valid_data)
            ld_valid = log_density_fn(batch_valid_data, model).item()

            batch_test_data = sample_batch(test_data)
            ld_test = log_density_fn(batch_test_data, model).item()

            avg_train = ld.item() / batch_train_data.shape[0]
            avg_valid = ld_valid / batch_valid_data.shape[0]
            avg_test = ld_test / batch_test_data.shape[0]

            print('step: %u, train: %f, valid: %f, test: %f' % (i, avg_train, avg_valid, avg_test) , flush=True)

        # eval on full dataset
        if i % EVAL_PERIOD == 0:
            avg_train, avg_valid, avg_test, avg_llh = eval_model(i)
            infos.append( (i, avg_train, avg_valid, avg_test, avg_llh) )

    return model, infos



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',        type=str,            required=True,                                   help="dataset name")
    parser.add_argument('--modeltype',      type=str,            required=True, choices=['hyperspn', 'spn'],      help="type of model")
    parser.add_argument('--run',            type=int,            default=1,                                       help="run id")
    parser.add_argument('--h',              type=int,            default=5,                                       help="embedding dimension")
    parser.add_argument('--N',              type=int,            default=5,                                       help="each sector has size N*N")
    parser.add_argument('--R',              type=int,            default=50,                                      help="number of regions")
    parser.add_argument('--lr',             type=float,          default=3e-4,                                    help="learning rate")
    parser.add_argument('--batch',          type=int,            default=100,                                     help="batch size")
    parser.add_argument('--wd',             type=float,          default=0.000,                                   help="weight decay")
    parser.add_argument('--eval',           action='store_true', default=False,                                   help="evaluate model on test set")
    ARGS = parser.parse_args()
    print(ARGS)

    train_data, valid_data, test_data = load_dataset(ARGS.dataset)

    savepath = 'output/%s_run=%u_h=%u_N=%u_%s_wd=%.5f' % (ARGS.dataset, ARGS.run, ARGS.h, ARGS.N, ARGS.modeltype, ARGS.wd)
    modelpath = '%s.pt' % (savepath)

    model = load_model(modelpath, train_data, ARGS, DEVICE)
    model, infos = run(train_data, valid_data, test_data, model=model, device=DEVICE)

    if not ARGS.eval:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, modelpath)
        print('Saved\n')

    with open('%s.csv' % savepath, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        for row in infos:
            writer.writerow(row)