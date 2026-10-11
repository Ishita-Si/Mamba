"""Minimal Mamba (S6) block in pure PyTorch + an equal-size Transformer baseline.

Paper: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023/COLM 2024).
Everything in MambaBlock / selective_scan is written from scratch for this project.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- scan
def selective_scan_seq(u, delta, A, B, C, D):
    """Reference sequential scan (Algorithm 2 of the paper), O(L) python loop.
    u, delta: (b,l,d)   A: (d,n)   B, C: (b,l,n)   D: (d,)  ->  (b,l,d)
    h_t = exp(delta_t A) * h_{t-1} + delta_t B_t u_t   (ZOH for A, Euler for B)
    y_t = C_t h_t + D u_t
    """
    b, l, d = u.shape
    dA = torch.exp(delta.unsqueeze(-1) * A)                        # (b,l,d,n)
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)   # (b,l,d,n)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(l):
        h = dA[:, t] * h + dBu[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1) + u * D


def selective_scan_par(u, delta, A, B, C, D):
    """Parallel (Hillis-Steele) scan of the same recurrence, O(L log L) work, log L steps.
    The recurrence h_t = a_t h_{t-1} + b_t is associative under
    (a1,b1) o (a2,b2) = (a1*a2, a2*b1 + b2), so it can be evaluated in parallel.
    """
    a = torch.exp(delta.unsqueeze(-1) * A)                          # (b,l,d,n)
    x = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)      # (b,l,d,n)
    l = u.shape[1]
    off = 1
    while off < l:
        a_prev = F.pad(a[:, :-off], (0, 0, 0, 0, off, 0), value=1.0)
        x_prev = F.pad(x[:, :-off], (0, 0, 0, 0, off, 0), value=0.0)
        x = a * x_prev + x
        a = a * a_prev
        off *= 2
    y = (x * C.unsqueeze(2)).sum(-1)
    return y + u * D


# --------------------------------------------------------------------------- block
class MambaBlock(nn.Module):
    """in_proj -> (x, z); x -> causal depthwise conv -> SiLU -> selective SSM; y * SiLU(z) -> out_proj.

    selective=False gives the LTI ablation (S4-style): delta, B, C become input-independent
    learned parameters, so the model is a linear time-invariant system.
    """

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_rank=None,
                 selective=True, scan="par"):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.selective = selective
        self.scan = selective_scan_par if scan == "par" else selective_scan_seq

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        inv_sp = dt + torch.log(-torch.expm1(-dt))                  # inverse softplus
        if selective:
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_sp)
                std = self.dt_rank ** -0.5
                nn.init.uniform_(self.dt_proj.weight, -std, std)
        else:
            self.dt_bias = nn.Parameter(inv_sp.clone())
            self.B_fix = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
            self.C_fix = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
        # S4D-real init: A = -(1..N), stored as log
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2)   # causal
        x = F.silu(x)
        A = -torch.exp(self.A_log)
        if self.selective:
            dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(dt))
        else:
            delta = F.softplus(self.dt_bias).expand(b, l, -1)
            B = self.B_fix.expand(b, l, -1)
            C = self.C_fix.expand(b, l, -1)
        y = self.scan(x, delta, A, B, C, self.D)
        return self.out_proj(y * F.silu(z))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# --------------------------------------------------------------------------- baseline
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, mlp_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_mult * d_model), nn.GELU(),
                                 nn.Linear(mlp_mult * d_model, d_model))
        self.n1, self.n2 = RMSNorm(d_model), RMSNorm(d_model)

    def forward(self, x):
        l = x.shape[1]
        mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), 1)
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    """Token embedding -> n_layers x (RMSNorm -> mixer, residual) -> RMSNorm -> tied head."""

    def __init__(self, vocab, d_model, n_layers, arch="mamba", max_len=1024, **kw):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(vocab, d_model)
        if arch == "transformer":
            self.pos = nn.Embedding(max_len, d_model)
            self.layers = nn.ModuleList([TransformerBlock(d_model, kw.get("n_heads", 4), kw.get("mlp_mult", 4)) for _ in range(n_layers)])
            self.norms = None
        else:
            self.layers = nn.ModuleList([MambaBlock(d_model, **{k: v for k, v in kw.items() if k not in ("n_heads", "mlp_mult")})
                                         for _ in range(n_layers)])
            self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        if self.arch == "transformer":
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
            for layer in self.layers:
                x = layer(x)
        else:
            for n, layer in zip(self.norms, self.layers):
                x = x + layer(n(x))
        return self.head(self.norm_f(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
