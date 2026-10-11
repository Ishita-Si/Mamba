import json, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = sys.argv[1] if len(sys.argv) > 1 else "results"
names = {"mamba": "Mamba (selective)", "mamba_lti": "Mamba-LTI (no selection)", "transformer": "Transformer (4x MLP)", "transformer_pm": "Transformer (param-matched)"}
def load(task):
    out = {}
    for arch in names:
        fs = sorted(glob.glob(f"{R}/{task}_{arch}_s*.json"))
        if fs: out[arch] = json.load(open(fs[0]))
    return out
rows = ["| Task | Model | Params | Metric | Value | Train time (s) |", "|---|---|---|---|---|---|"]
for task, key, label, fn in [("copy", "acc_seq", "seq. accuracy", "copy_curves.png"), ("lm", "val_bpc", "val bits/char", "lm_curves.png")]:
    res = load(task)
    if not res: continue
    fig, ax = plt.subplots(1, 2 if task == "copy" else 1, figsize=(9 if task == "copy" else 4.5, 3.2))
    ax = ax if task == "copy" else [ax]
    for arch, r in res.items():
        s = [l["step"] for l in r["log"]]
        ax[0].plot(s, [l[key] for l in r["log"]], label=names[arch])
        if task == "copy": ax[1].plot(s, [l["acc_token"] for l in r["log"]], label=names[arch])
        f = r["final"]
        rows.append(f"| {task} | {names[arch]} | {r['params']/1e3:.1f}K | {label} | {f[key]:.3f} | {f['time_s']:.0f} |")
    ax[0].set_xlabel("step"); ax[0].set_ylabel(label); ax[0].legend(fontsize=7)
    if task == "copy": ax[1].set_xlabel("step"); ax[1].set_ylabel("token accuracy")
    plt.tight_layout(); plt.savefig(f"{R}/{fn}", dpi=160); plt.close()
open(f"{R}/results_table.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
