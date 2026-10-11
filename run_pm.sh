#!/bin/sh
while [ ! -f results/DONE ]; do sleep 5; done
python3 train.py --task copy --arch transformer --tag _pm --mlp_mult 2 --d_model 32 --d_state 8 --L 32 --K 4 --bs 32 --steps 1500 > results/log_copy_transformer_pm.txt 2>&1
python3 train.py --task lm --arch transformer --tag _pm --mlp_mult 2 --d_model 32 --L 64 --bs 32 --steps 1200 --lr 3e-3 > results/log_lm_transformer_pm.txt 2>&1
echo done > results/DONE_PM
