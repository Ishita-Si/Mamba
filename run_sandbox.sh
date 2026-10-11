#!/bin/sh
# Reduced CPU-sandbox versions of the experiments (the Colab versions use the defaults in train.py)
C="--task copy --d_model 32 --d_state 8 --L 32 --K 4 --bs 32 --steps 1500"
for arch in mamba mamba_lti transformer; do
  python3 train.py $C --arch $arch > results/log_copy_$arch.txt 2>&1
done
M="--task lm --d_model 32 --d_state 8 --L 64 --bs 32 --steps 1200 --lr 3e-3"
for arch in mamba transformer mamba_lti; do
  python3 train.py $M --arch $arch > results/log_lm_$arch.txt 2>&1
done
echo done > results/DONE
