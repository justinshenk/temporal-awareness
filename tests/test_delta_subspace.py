"""Tests for projected_injection and PCA bands (pure, no model)."""

from __future__ import annotations

import torch

from src.probes.attribution.delta_subspace import energy_fraction, pca_bands
from src.probes.attribution.lockstep_oracle import projected_injection


def test_projected_injection_full_rank_is_oracle():
    torch.manual_seed(0)
    d = 6
    a = torch.randn(4, d)
    lora = torch.randn(4, d)
    V, _ = torch.linalg.qr(torch.randn(d, d))  # full orthonormal basis
    assert torch.allclose(projected_injection(a, lora, V), lora, atol=1e-5)


def test_projected_injection_empty_band_is_base():
    d = 5
    a = torch.randn(3, d)
    lora = torch.randn(3, d)
    V = torch.zeros(d, 0)  # no directions
    assert torch.allclose(projected_injection(a, lora, V), a, atol=1e-6)


def test_projected_injection_delta_lies_in_span():
    torch.manual_seed(1)
    d, k = 8, 3
    a = torch.randn(5, d)
    lora = torch.randn(5, d)
    V, _ = torch.linalg.qr(torch.randn(d, k))   # (d,3) orthonormal
    out = projected_injection(a, lora, V)
    injected = out - a
    assert torch.allclose(injected, (injected @ V) @ V.T, atol=1e-5)  # lies in span(V)


def test_pca_bands_recovers_dominant_direction():
    torch.manual_seed(0)
    n, d = 4000, 6
    u = torch.zeros(d); u[2] = 1.0                     # dominant energy along e2
    D = 5.0 * torch.randn(n, 1) * u + 0.1 * torch.randn(n, d)
    V, lam = pca_bands(D)
    assert lam[0] > lam[1]
    assert abs(abs(float(V[:, 0] @ u.double())) - 1.0) < 0.05  # top eigvec ≈ ±e2


def test_energy_fraction_monotone_and_bounded():
    lam = torch.tensor([10.0, 5.0, 1.0, 0.0])
    assert energy_fraction(lam, 1) < energy_fraction(lam, 2)
    assert abs(energy_fraction(lam, 4) - 1.0) < 1e-9
