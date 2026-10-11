# Mamba (selective SSM) from scratch — Deep Learning 2026, DL-061

Re-implementation of *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao, COLM 2024) in pure PyTorch.

## Contents
- `src/model.py` — MambaBlock (S6 selective scan, ZOH/Euler discretisation, causal conv, SiLU gating), sequential + parallel scan, LTI ablation switch, Transformer baseline.
- `src/data.py` — selective-copying generator, TinyShakespeare loader.
- `train.py` — training / evaluation (`--task copy|lm`, `--arch mamba|mamba_lti|transformer`).
- `tests_scan.py` — parallel scan == sequential scan; block is causal.
- `mamba_colab.ipynb` — runnable Colab notebook (T4). `make_report_assets.py` — table + plots.
- `results/` — JSON logs, curves, `results_table.md`. `report/report.pdf` — report.

## Setup / run
```
pip install torch matplotlib
python tests_scan.py
python train.py --task copy --arch mamba --d_model 32 --d_state 8 --L 32 --K 4 --bs 32 --steps 1500   # sandbox scale
./run_sandbox.sh && ./run_pm.sh                                                                          # all reduced runs
python make_report_assets.py
```
Seed 0 everywhere (`--seed`). Colab-scale commands are in the notebook.

## Results (reduced scale, CPU, 1 seed — see report for what changed vs the paper)
| Task | Model | Params | Metric | Value | Train time (s) |
|---|---|---|---|---|---|
| copy | Mamba (selective) | 17.2K | seq. accuracy | 0.986 | 201 |
| copy | Mamba-LTI (no selection) | 14.7K | seq. accuracy | 0.062 | 195 |
| copy | Transformer (4x MLP) | 33.8K | seq. accuracy | 0.998 | 30 |
| copy | Transformer (param-matched) | 18.5K | seq. accuracy | 0.930 | 27 |
| lm | Mamba (selective) | 18.9K | val bits/char | 2.915 | 296 |
| lm | Mamba-LTI (no selection) | 16.4K | val bits/char | 2.894 | 269 |
| lm | Transformer (4x MLP) | 35.6K | val bits/char | 3.438 | 38 |
| lm | Transformer (param-matched) | 21.4K | val bits/char | 3.540 | 34 |

## Notes
- The core scan is written from scratch; `state-spaces/mamba` (official repo) is cited only as the reference for initialisation choices (S4D-real A, dt in [1e-3,0.1]).
- No fused CUDA kernel: the scan is a Python loop / parallel Hillis-Steele scan, so speed is not comparable to the paper.
- Colab-scale numbers: replace the table above with your own Colab runs.

State-size ablation (Mamba, selective copy, seq. acc.): N=1: 0.889, N=4: 0.801, N=8: 0.986 (one seed, non-monotone => treat as noise). Run with `./run_N.sh`.
