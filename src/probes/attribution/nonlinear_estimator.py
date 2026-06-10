"""Nonlinear (MLP) estimator of the residual shift ``δ = lora_resid − base_resid`` at a layer.

The linear ridge map lands *below* the lockstep recovery onset (cos≈0.61, R²≈0.31 at base's acting
site → 0.05 recovery), while the fidelity sweep shows the budget is reachable by t≈0.8. A learned
estimator ``f(a)≈δ`` needs no oracle access, so it is plain steering: inject ``a + α·f(a)`` every
step via :class:`NonlinearSteerHook` and score with ordinary KV-cache generation.
"""

from __future__ import annotations

import torch
from torch import nn


class DeltaMLP(nn.Module):
    """``a → δ`` MLP. Input LayerNorm absorbs the residual-norm scale; output is the raw shift."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


def _row_cosine(P: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> float:
    cos = (P * T).sum(1) / (P.norm(dim=1) * T.norm(dim=1) + eps)
    return float(cos.mean())


def _r2(P: torch.Tensor, T: torch.Tensor) -> float:
    ss_res = ((T - P) ** 2).sum()
    ss_tot = ((T - T.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_delta_mlp(a_tr, d_tr, a_val, d_val, hidden: int = 4096, epochs: int = 100,
                  lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 2048,
                  dropout: float = 0.1, device: str = "cpu", seed: int = 0,
                  patience: int = 12, verbose: bool = False):
    """Fit ``f(a)≈δ`` by AdamW/MSE with early stopping on validation cosine. Returns (mlp, metrics)."""
    torch.manual_seed(seed)
    a_tr, d_tr = a_tr.to(device).float(), d_tr.to(device).float()
    a_val, d_val = a_val.to(device).float(), d_val.to(device).float()
    mlp = DeltaMLP(a_tr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    n = a_tr.shape[0]
    best_cos, best_state, best_metrics, stale = -1e30, None, None, 0

    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((mlp(a_tr[idx]) - d_tr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            pv = mlp(a_val)
        cos, r2 = _row_cosine(pv, d_val), _r2(pv, d_val)
        if verbose:
            print(f"  epoch {ep:3d}: val cos={cos:+.3f} R²={r2:+.3f}", flush=True)
        if cos > best_cos:
            best_cos, best_metrics = cos, {"val_cosine": cos, "val_r2": r2, "epoch": ep}
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    mlp.load_state_dict(best_state)
    return mlp, best_metrics


class NonlinearSteerHook:
    """Inject ``α·f(hs)`` at one decoder layer every position (steering with an MLP map)."""

    def __init__(self, model, mlp: nn.Module, layer: int, alpha: float = 1.0):
        self.mlp = mlp.eval()
        self.layer, self.alpha, self.enabled, self._hooks = layer, alpha, True, []
        self._dtype = next(mlp.parameters()).dtype
        self._hooks.append(model.model.layers[layer].register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                delta = self.mlp(hs.to(self._dtype))
            hs = hs + self.alpha * delta.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
