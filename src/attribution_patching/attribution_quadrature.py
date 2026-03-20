"""Quadrature methods for numerical integration in attribution patching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class QuadratureMethod(str, Enum):
    """Quadrature methods for numerical integration."""

    MIDPOINT = "midpoint"
    LEGENDRE = "gauss-legendre"
    CHEBYSHEV = "gauss-chebyshev"


@dataclass
class QuadratureNodes:
    """Quadrature nodes and weights for numerical integration."""

    nodes: np.ndarray  # Alpha values where to evaluate
    weights: np.ndarray  # Weights for each node


def get_quadrature(
    n_steps: int,
    method: QuadratureMethod = QuadratureMethod.MIDPOINT,
    a: float = 0.0,
    b: float = 1.0,
) -> QuadratureNodes:
    """Get quadrature nodes and weights for numerical integration.

    Args:
        n_steps: Number of integration points
        method: Quadrature method to use
        a: Lower bound of integration interval
        b: Upper bound of integration interval

    Returns:
        QuadratureNodes with nodes (alpha values) and weights
    """
    if method == QuadratureMethod.MIDPOINT:
        return _midpoint_quadrature(n_steps, a, b)
    elif method == QuadratureMethod.LEGENDRE:
        return _gauss_legendre_quadrature(n_steps, a, b)
    elif method == QuadratureMethod.CHEBYSHEV:
        return _gauss_chebyshev_quadrature(n_steps, a, b)
    else:
        raise ValueError(f"Unknown quadrature method: {method}")


def _midpoint_quadrature(n_steps: int, a: float, b: float) -> QuadratureNodes:
    """Midpoint rule quadrature."""
    interval_width = (b - a) / n_steps
    nodes = np.array([(i + 0.5) * interval_width + a for i in range(n_steps)])
    weights = np.full(n_steps, interval_width)
    return QuadratureNodes(nodes=nodes, weights=weights)


def _gauss_legendre_quadrature(n_steps: int, a: float, b: float) -> QuadratureNodes:
    """Gauss-Legendre quadrature, transformed to [a, b]."""
    # Get standard nodes/weights on [-1, 1]
    std_nodes, std_weights = np.polynomial.legendre.leggauss(n_steps)

    # Transform to [a, b]: x = (b-a)/2 * t + (a+b)/2
    half_width = (b - a) / 2
    midpoint = (a + b) / 2
    nodes = half_width * std_nodes + midpoint
    weights = half_width * std_weights

    return QuadratureNodes(nodes=nodes, weights=weights)


def _gauss_chebyshev_quadrature(n_steps: int, a: float, b: float) -> QuadratureNodes:
    """Gauss-Chebyshev quadrature of the first kind, transformed to [a, b].

    Standard Chebyshev quadrature integrates f(x)/sqrt(1-x^2) on [-1,1].
    To integrate f(x) directly, we absorb the weight function into the
    quadrature weights by multiplying by sqrt(1-x_k^2).

    The modified weights are normalized to sum to (b-a) for exact integration
    of constant functions.
    """
    # Chebyshev nodes of the first kind on [-1, 1]
    k = np.arange(1, n_steps + 1)
    std_nodes = np.cos((2 * k - 1) * np.pi / (2 * n_steps))

    # Modified weights to integrate f(x) directly (not f(x)/sqrt(1-x^2))
    # w_k = (π/n) * sqrt(1 - x_k^2) = (π/n) * sin((2k-1)π/(2n))
    raw_weights = np.sin((2 * k - 1) * np.pi / (2 * n_steps))

    # Normalize weights to sum to 2 on [-1, 1] (interval length)
    raw_weights = raw_weights * (2.0 / np.sum(raw_weights))

    # Transform to [a, b]
    half_width = (b - a) / 2
    midpoint = (a + b) / 2
    nodes = half_width * std_nodes + midpoint
    weights = half_width * raw_weights

    return QuadratureNodes(nodes=nodes, weights=weights)
