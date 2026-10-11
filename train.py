"""Usage:
  python train.py --task copy --arch mamba        --steps 1500
  python train.py --task copy --arch mamba_lti    --steps 1500   (ablation: non-selective / LTI)
  python train.py --task copy --arch transformer  --steps 1500
  python train.py --task lm   --arch mamba        --steps 2000
"""
import argparse, json, math, os, time
import torch, torch.nn.functional as F
from src.model import LM, n_params
from src.data import selective_copy_batch, load_shakespeare, lm_batch

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["copy", "lm"], required=True)
p.add_argument("--arch", choices=["mamba", "mamba_lti", "transformer"], required=True)
p.add_argument("--d_model", type=int, default=64); p.add_argument("--layers", type=int, default=2)
p.add_argument("--d_state", type=int, default=16); p.add_argument("--mlp_mult", type=int, default=4)
p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--seed", type=int, default=0)
p.add_argument("--L", type=int, default=64); p.add_argument("--K", type=int, default=8)
p.add_argument("--out", default="results"); p.add_argument("--tag", default="")
a = p.parse_args()

torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
V = 8
if a.task == "copy":
    vocab = V + 2
else:
    train_d, val_d, vocab = load_shakespeare()

if a.arch == "transformer":
    model = LM(vocab, a.d_model, a.layers, "transformer", max_len=max(a.L + a.K, 256) if a.mlp_mult == 4 else a.L + a.K, mlp_mult=a.mlp_mult)
else:
    model = LM(vocab, a.d_model, a.layers, "mamba", d_state=a.d_state, selective=(a.arch == "mamba"))
model.to(dev)
print(f"{a.task}/{a.arch}: {n_params(model)/1e3:.1f}K params on {dev}")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)

def evaluate():
    model.eval(); gg = torch.Generator().manual_seed(999)
    with torch.no_grad():
        if a.task == "copy":
            x, y = selective_copy_batch(512, a.L, a.K, V, dev, gg)
            pred = model(x).argmax(-1); m = y != -100
            return {"acc_token": (pred[m] == y[m]).float().mean().item(),
                    "acc_seq": ((pred == y) | ~m).all(1).float().mean().item()}
        tot = 0
        for _ in range(20):
            x, y = lm_batch(val_d, 32, a.L, dev, gg)
            tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
        nll = tot / 20
        return {"val_nll": nll, "val_bpc": nll / math.log(2), "val_ppl": math.exp(nll)}

log, t0 = [], time.time()
for step in range(1, a.steps + 1):
    model.train()
    if a.task == "copy":
        x, y = selective_copy_batch(a.bs, a.L, a.K, V, dev, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
    else:
        x, y = lm_batch(train_d, a.bs, a.L, dev, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % max(a.steps // 15, 1) == 0 or step == a.steps:
        r = {"step": step, "loss": loss.item(), **evaluate(), "time_s": time.time() - t0}
        log.append(r); print(r, flush=True)

res = {"args": vars(a), "params": n_params(model), "final": log[-1], "log": log}
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/{a.task}_{a.arch}{a.tag}_s{a.seed}.json", "w"), indent=1)
