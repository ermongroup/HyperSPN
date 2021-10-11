# HyperSPN

This repository contains code for the paper:

HyperSPNs: Compact and Expressive Probabilistic Circuits

```
"HyperSPNs: Compact and Expressive Probabilistic Circuits"
Andy Shih, Dorsa Sadigh, Stefano Ermon
In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS), 2021

@inproceedings{ShihSEneurips21,
  author    = {Andy Shih and Dorsa Sadigh and Stefano Ermon},
  title     = {HyperSPNs: Compact and Expressive Probabilistic Circuits},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  month     = {december},
  year      = {2021},
  keywords  = {conference}
}
```

## Installation

```
conda env create -f environment.yml
```

Optionally, for EinsumNetworks:
```
cd EinsumNetworks
pip3 install -r requirements.txt
```

## Datasets and Repos

The Twenty Datasets benchmark is from [here](https://github.com/arranger1044/DEBD).

The Amazon Baby Registries benchmark is from [here](https://github.com/cgartrel/LowRankDPP.jl/tree/master/data/Amazon-baby-registry). The dataset was converted from the set format into the binary format.

The Einsum Network repository is from [here](https://github.com/cambridge-mlg/EinsumNetworks).


## Commands

Experiments can be launched with the helper bash files
```
runid=0
bash bashfiles/run_hyperspn.bash ${runid} 5
bash bashfiles/run_hyperspn.bash ${runid} 10
bash bashfiles/run_hyperspn.bash ${runid} 20

bash bashfiles/run_spn.bash ${runid} 1e-3
bash bashfiles/run_spn.bash ${runid} 1e-4
bash bashfiles/run_spn.bash ${runid} 1e-5
```

```
cd EinsumNetworks/src/
python train_svhn_mixture.py --run=0
python train_svhn_mixture.py --nn --run=0
```