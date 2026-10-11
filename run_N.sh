#!/bin/sh
for N in 1 4; do
python3 train.py --task copy --arch mamba --tag _N$N --d_model 32 --d_state $N --L 32 --K 4 --bs 32 --steps 1500 > results/log_copy_mamba_N$N.txt 2>&1
done
echo done > results/DONE_N
