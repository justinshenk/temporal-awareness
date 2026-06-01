"""Integration test for the per-layer comparison used by the report script."""

import numpy as np

from scripts.lora_icl.run_subspace_comparison import compare_layer


def test_aligned_shifts_score_high():
    rng = np.random.default_rng(0)
    d, n = 256, 30
    direction = rng.normal(size=d)
    # Both shift sets point along the same direction with small per-example noise.
    icl = direction + 0.05 * rng.normal(size=(n, d))
    lora = 0.7 * direction + 0.05 * rng.normal(size=(n, d))
    res = compare_layer(icl, lora, layer=14, k=5)
    assert res.layer == 14
    assert res.mean_cosine > 0.9
    assert res.n_examples == n and res.hidden_dim == d


def test_orthogonal_shifts_score_near_zero():
    rng = np.random.default_rng(1)
    d, n = 256, 30
    a = np.zeros(d)
    a[0] = 1.0
    b = np.zeros(d)
    b[1] = 1.0
    icl = a + 0.01 * rng.normal(size=(n, d))
    lora = b + 0.01 * rng.normal(size=(n, d))
    res = compare_layer(icl, lora, layer=0, k=5)
    assert abs(res.mean_cosine) < 0.1
