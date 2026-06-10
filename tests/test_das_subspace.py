"""Tests for the DAS δ-subspace glue (CPU, no network).

Asserts the four crux properties in miniature: the learned basis stays orthonormal through QR; the
injection collapses to the full oracle at ``r=d``; a single optimizer step on a differentiable fake
forward lowers the loss while ``R`` remains orthonormal (the QR guarantees it every step); and the
LM-loss helper supervises the right CoT positions/targets.
"""

from __future__ import annotations

import torch
from torch import nn

from src.probes.attribution.das_subspace import OrthoSubspace, inject_value, subspace_lm_loss


def test_ortho_subspace_columns_orthonormal():
    R = OrthoSubspace(d=8, r=3, seed=1)()
    assert R.shape == (8, 3)
    eye = R.T @ R
    assert torch.allclose(eye, torch.eye(3), atol=1e-5)


def test_inject_value_full_rank_is_oracle():
    torch.manual_seed(0)
    a = torch.randn(5, 8)
    delta = torch.randn(5, 8)
    R = OrthoSubspace(d=8, r=8, seed=2)()        # full rank ⇒ span = ℝ^8
    out = inject_value(a, delta, R)
    assert torch.allclose(out, a + delta, atol=1e-5)   # exact oracle residual


def test_inject_value_is_projection_of_delta():
    torch.manual_seed(0)
    a = torch.randn(4, 6)
    delta = torch.randn(4, 6)
    R = OrthoSubspace(d=6, r=2, seed=3)()
    out = inject_value(a, delta, R)
    proj = (delta @ R) @ R.T
    assert torch.allclose(out - a, proj, atol=1e-6)
    # the injected shift lies entirely in span(R): re-projecting is a no-op
    assert torch.allclose((out - a) @ R @ R.T, out - a, atol=1e-5)


def test_training_step_lowers_loss_and_keeps_orthonormal():
    """One Adam step on a linear fake 'upper stack' must reduce CE and keep R orthonormal."""
    torch.manual_seed(0)
    d, r, vocab, seq = 8, 3, 10, 6
    a = torch.randn(seq, d)
    delta = torch.randn(seq, d)
    head = nn.Linear(d, vocab, bias=False)       # stand-in for base upper layers + lm_head
    target = torch.randint(0, vocab, (seq,))
    sub = OrthoSubspace(d, r, seed=4)
    opt = torch.optim.Adam(sub.parameters(), lr=0.1)

    def loss_fn():
        h = inject_value(a, delta, sub())        # (seq, d), differentiable in R
        logits = head(h).unsqueeze(0)            # (1, seq, vocab)
        return subspace_lm_loss(logits, target, plen=2)

    before = float(loss_fn().detach())
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    after = float(loss_fn().detach())
    assert after < before
    R = sub()
    assert torch.allclose(R.T @ R, torch.eye(r), atol=1e-5)


def test_subspace_lm_loss_positions_and_targets():
    """The loss must compare logits[plen-1:seq-1] against target_ids[plen:seq]."""
    seq, vocab, plen = 5, 7, 2
    logits = torch.full((1, seq, vocab), -10.0)
    target = torch.tensor([0, 0, 3, 5, 6])       # CoT targets at positions 2,3,4
    # make the supervised predictions perfect: logits[t] peaks at target[t+1] for t in [plen-1, seq-2]
    for t in range(plen - 1, seq - 1):
        logits[0, t, target[t + 1]] = 10.0
    loss = subspace_lm_loss(logits, target, plen)
    assert loss < 1e-3                           # near-zero when predictions match targets
    # corrupting one supervised position raises the loss
    logits[0, plen - 1, target[plen]] = -10.0
    assert subspace_lm_loss(logits, target, plen) > loss
