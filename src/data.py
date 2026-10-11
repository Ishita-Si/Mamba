import os, urllib.request
import torch

# ---------------------------------------------------------------- selective copying
# Tokens: 0 = noise/blank, 1..V = data tokens, V+1 = "recall" marker.
# Input : L positions with K data tokens scattered at RANDOM positions (rest noise), then K recall markers.
# Target: at the K final positions, output the K data tokens in order. Loss only there.
# Requires content-aware selection: which positions to remember depends on the input itself.
def selective_copy_batch(bs, L=64, K=8, V=8, device="cpu", g=None):
    ids = torch.zeros(bs, L + K, dtype=torch.long)
    tgt = torch.full((bs, L + K), -100, dtype=torch.long)
    for i in range(bs):
        pos = torch.randperm(L, generator=g)[:K].sort().values
        vals = torch.randint(1, V + 1, (K,), generator=g)
        ids[i, pos] = vals
        tgt[i, L:] = vals
    ids[:, L:] = V + 1
    return ids.to(device), tgt.to(device)

# ---------------------------------------------------------------- TinyShakespeare
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_shakespeare(path="data/tinyshakespeare.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)

def lm_batch(data, bs, L, device="cpu", g=None):
    ix = torch.randint(0, len(data) - L - 1, (bs,), generator=g)
    x = torch.stack([data[i:i + L] for i in ix]); y = torch.stack([data[i + 1:i + L + 1] for i in ix])
    return x.to(device), y.to(device)
