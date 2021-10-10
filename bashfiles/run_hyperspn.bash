#!/bin/bash

set -x

runid=$1
h=$2

for dataset in amzn_apparel amzn_bath amzn_bedding amzn_carseats amzn_diaper amzn_feeding amzn_furniture amzn_gear amzn_gifts amzn_health amzn_media amzn_moms amzn_safety amzn_strollers amzn_toys
do
    python run.py --run=${runid} --dataset=${dataset}    --batch=500 --N=5 --h=${h} --modeltype=hyperspn --lr=5e-3
done

for dataset in nltcs msnbc kdd plants baudio jester bnetflix accidents tretail pumsb_star dna kosarek msweb book tmovie cwebkb
do
    python run.py --run=${runid} --dataset=${dataset}    --batch=500 --N=5 --h=${h} --modeltype=hyperspn --lr=5e-3
done

for dataset in cr52 c20ng bbc ad
do
    python run.py --run=${runid} --dataset=${dataset}    --batch=50 --N=5 --h=${h} --modeltype=hyperspn --lr=5e-3
done

for dataset in toy
do
    python run.py --run=${runid} --dataset=${dataset}    --batch=500 --N=5 --h=${h} --modeltype=hyperspn --lr=5e-3
done