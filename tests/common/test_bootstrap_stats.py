"""Bootstrap intervals, and the clustering that keeps a token-level gap from overstating itself.

The clustered helper exists because the gold-token lens (P4) reports a difference over ~20k
tokens drawn from only ~300 problems. These tests pin the two properties the verdict depends on:
the point estimate is the plain pooled difference (clustering must not move it), and the interval
widens as dependence concentrates inside clusters.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.bootstrap_stats import (
    bootstrap_interval,
    clustered_rate_gap,
    pooled_rate_gap,
)


def test_bootstrap_interval_is_seeded_and_brackets_the_estimate():
    values = np.random.default_rng(0).normal(5.0, 1.0, 400)
    first = bootstrap_interval(values, np.mean)
    again = bootstrap_interval(values, np.mean)
    assert (first.lo, first.estimate, first.hi) == (again.lo, again.estimate, again.hi)
    assert first.lo < first.estimate < first.hi
    assert first.n == 400


def test_pooled_rate_gap_weighs_clusters_by_size():
    """Pooled, not per-cluster-averaged: 3/4 vs 1/4 over two rows is 0.75 - 0.25."""
    rows = np.array([[2.0, 2.0, 2.0, 1.0],
                     [2.0, 1.0, 2.0, 0.0]])
    assert pooled_rate_gap(rows) == pytest.approx(0.75 - 0.25)


def test_pooled_rate_gap_is_nan_when_a_side_is_empty():
    assert np.isnan(pooled_rate_gap(np.array([[0.0, 0.0, 3.0, 1.0]])))


def test_clustered_estimate_equals_the_flat_pooled_difference():
    rng = np.random.default_rng(1)
    counts_a = np.stack([np.full(50, 10.0), rng.integers(0, 11, 50).astype(float)], axis=1)
    counts_b = np.stack([np.full(50, 10.0), rng.integers(0, 11, 50).astype(float)], axis=1)
    iv = clustered_rate_gap(counts_a, counts_b)
    flat = counts_a[:, 1].sum() / counts_a[:, 0].sum() - counts_b[:, 1].sum() / counts_b[:, 0].sum()
    assert iv.estimate == pytest.approx(flat)
    assert iv.n == 50                                   # clusters, not the 1000 observations


def test_clustering_widens_the_interval_when_outcomes_are_within_cluster_correlated():
    """Same 400 observations, same point estimate; perfectly correlated clusters carry less n.

    ``spread`` gives every cluster the same mixed outcome (tokens independent in effect);
    ``lumpy`` makes each cluster all-hits or all-misses (one problem = one effective
    observation). The interval must be materially wider in the lumpy case, which is exactly the
    overstatement a per-token interval would hide.
    """
    n_clusters, per = 40, 10
    b = np.stack([np.full(n_clusters, float(per)), np.full(n_clusters, per / 2.0)], axis=1)

    spread = np.stack([np.full(n_clusters, float(per)), np.full(n_clusters, 7.0)], axis=1)
    lumpy_hits = np.array([float(per) if i % 10 < 7 else 0.0 for i in range(n_clusters)])
    lumpy = np.stack([np.full(n_clusters, float(per)), lumpy_hits], axis=1)

    iv_spread, iv_lumpy = clustered_rate_gap(spread, b), clustered_rate_gap(lumpy, b)
    assert iv_spread.estimate == pytest.approx(iv_lumpy.estimate)
    assert (iv_lumpy.hi - iv_lumpy.lo) > 2 * (iv_spread.hi - iv_spread.lo)


def test_clustered_rate_gap_rejects_misaligned_or_wrong_shaped_counts():
    ok = np.ones((3, 2))
    with pytest.raises(ValueError, match="aligned"):
        clustered_rate_gap(ok, np.ones((4, 2)))
    with pytest.raises(ValueError, match="aligned"):
        clustered_rate_gap(np.ones((3, 3)), np.ones((3, 3)))


def test_excludes_zero_and_render_report_the_sign():
    iv = clustered_rate_gap(np.stack([np.full(30, 10.0), np.full(30, 9.0)], axis=1),
                            np.stack([np.full(30, 10.0), np.full(30, 3.0)], axis=1))
    assert iv.excludes_zero()
    assert iv.render().startswith("+0.600 [+")
