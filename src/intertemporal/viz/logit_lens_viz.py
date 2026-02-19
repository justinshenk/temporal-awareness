"""Logit lens visualization for tracking predictions across layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ...binary_choice import BinaryChoiceRunner
from ...common.contrastive_pair import ContrastivePair
from ...viz.layer_position_heatmaps import _finalize_plot


def visualize_logit_lens(
    pairs: list[ContrastivePair],
    runner: BinaryChoiceRunner,
    output_dir: Path,
    max_pairs: int = 3,
) -> None:
    """Visualize logit lens for contrastive pairs.

    Shows how the model's prediction (choice token logits) evolves
    across layers, similar to the logit lens technique.

    Args:
        pairs: List of ContrastivePair objects
        runner: BinaryChoiceRunner with model access
        output_dir: Directory to save plots
        max_pairs: Maximum number of pairs to visualize
    """
    if not pairs:
        print("[viz] No pairs for logit lens visualization")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs[:max_pairs]):
        try:
            _plot_logit_lens_pair(pair, runner, output_dir / f"logit_lens_pair_{i}.png")
        except Exception as e:
            print(f"[viz] Failed to plot logit lens for pair {i}: {e}")

    print(f"[viz] Logit lens plots saved to {output_dir}")


def _plot_logit_lens_pair(
    pair: ContrastivePair,
    runner: BinaryChoiceRunner,
    save_path: Path,
) -> None:
    """Plot logit lens for a single contrastive pair.

    Shows logit difference, probability, and rank evolution across layers
    for both short and long trajectories.
    """
    # Get the model and check backend
    model = runner._model
    if model is None:
        print("[viz] No model available for logit lens")
        return

    # Get choice token IDs
    short_label = pair.short_label or "A"
    long_label = pair.long_label or "B"

    short_token_id = _get_first_token_id(runner, short_label)
    long_token_id = _get_first_token_id(runner, long_label)

    if short_token_id is None or long_token_id is None:
        print(f"[viz] Could not find token IDs for labels: {short_label}, {long_label}")
        return

    # Compute per-layer logits for both trajectories
    short_results = _compute_per_layer_logits(
        pair.short_traj, model, runner, short_token_id, long_token_id
    )
    long_results = _compute_per_layer_logits(
        pair.long_traj, model, runner, short_token_id, long_token_id
    )

    if short_results is None or long_results is None:
        print("[viz] Could not compute per-layer logits")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot short trajectory
    _plot_layer_evolution(
        axes[0],
        short_results,
        f"Short Trajectory ({short_label})",
        short_label,
        long_label,
    )

    # Plot long trajectory
    _plot_layer_evolution(
        axes[1],
        long_results,
        f"Long Trajectory ({long_label})",
        short_label,
        long_label,
    )

    fig.suptitle(
        f"Logit Lens: {short_label} vs {long_label}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    _finalize_plot(save_path)


def _get_first_token_id(runner: Any, label: str) -> int | None:
    """Get the first token ID for a label."""
    # Encode without special tokens
    token_ids = runner.encode_ids(label, add_special_tokens=False)
    return token_ids[0] if token_ids else None


def _compute_per_layer_logits(
    traj: Any,
    model: Any,
    runner: Any,
    token_a_id: int,
    token_b_id: int,
) -> dict | None:
    """Compute logits for choice tokens at each layer.

    Uses the residual stream activations and applies unembed to get
    per-layer predictions.
    """
    if not traj.has_internals():
        return None

    cache = traj.internals
    available_layers = []

    # Find available layers with resid_post
    for hook in cache.keys():
        if "hook_resid_post" in hook:
            try:
                layer = int(hook.split(".")[1])
                available_layers.append(layer)
            except (IndexError, ValueError):
                pass

    if not available_layers:
        return None

    available_layers = sorted(available_layers)

    # Get unembed matrix and layer norm based on model type
    W_U = None
    ln_final = None

    try:
        # TransformerLens model
        if hasattr(model, "W_U"):
            W_U = model.W_U  # [d_model, d_vocab]
            ln_final = model.ln_final
        # HuggingFace models (Qwen, Llama, etc.)
        elif hasattr(model, "lm_head"):
            W_U = (
                model.lm_head.weight.T
            )  # lm_head is [vocab, d_model], need [d_model, vocab]
            # Get layer norm - varies by model architecture
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                ln_final = model.model.norm  # Qwen/Llama style
            elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                ln_final = model.transformer.ln_f  # GPT-2 style
        else:
            print(f"[viz] Model type {type(model).__name__} - cannot find unembed")
            return None

        if W_U is None:
            print(f"[viz] Could not find unembedding matrix")
            return None

    except Exception as e:
        print(f"[viz] Error accessing unembed: {e}")
        return None

    # Last position for prediction
    last_pos = -1

    results = {
        "layers": available_layers,
        "logit_diff": [],  # logit(A) - logit(B)
        "logprob_a": [],
        "prob_a": [],
        "rank_a": [],
    }

    # Determine device
    device = W_U.device if hasattr(W_U, "device") else "cpu"

    with torch.no_grad():
        for layer in available_layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            resid = cache.get(hook_name)

            if resid is None:
                continue

            # Get last position
            if isinstance(resid, torch.Tensor):
                resid_last = resid[0, last_pos, :].to(device)
            else:
                resid_last = torch.tensor(resid[0, last_pos, :], device=device)

            # Apply layer norm if available
            resid_input = resid_last.unsqueeze(0)
            if ln_final is not None:
                # Match dtype of ln_final
                ln_dtype = next(ln_final.parameters()).dtype
                resid_normed = ln_final(resid_input.to(ln_dtype))
            else:
                resid_normed = resid_input

            # Get logits - ensure dtype matches W_U
            resid_normed = resid_normed.to(W_U.dtype)
            logits = resid_normed @ W_U  # [1, d_vocab]
            logits = logits.squeeze(0).float()  # Convert back to float for calculations

            # Extract metrics
            logit_a = logits[token_a_id].item()
            logit_b = logits[token_b_id].item()

            # Compute probabilities
            probs = torch.softmax(logits, dim=-1)
            prob_a = probs[token_a_id].item()

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            logprob_a = log_probs[token_a_id].item()

            # Compute rank (1-indexed, lower is better)
            sorted_indices = torch.argsort(logits, descending=True)
            rank_a = (sorted_indices == token_a_id).nonzero(as_tuple=True)[0].item() + 1

            results["logit_diff"].append(logit_a - logit_b)
            results["logprob_a"].append(logprob_a)
            results["prob_a"].append(prob_a)
            results["rank_a"].append(rank_a)

    return results


def _plot_layer_evolution(
    ax: plt.Axes,
    results: dict,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    """Plot logit/prob evolution across layers."""
    layers = results["layers"]
    logit_diff = results["logit_diff"]
    logprob_a = results["logprob_a"]
    prob_a = results["prob_a"]
    rank_a = results["rank_a"]

    # Create twin axis for probability/rank
    ax2 = ax.twinx()

    # Plot logit difference and logprob on primary axis
    line1 = ax.plot(
        layers,
        logit_diff,
        "b-",
        linewidth=2,
        label=f"Logit({label_a}) - Logit({label_b})",
    )
    line2 = ax.plot(layers, logprob_a, "g-", linewidth=2, label=f"Logprob({label_a})")

    # Plot probability and reciprocal rank on secondary axis
    line3 = ax2.plot(
        layers,
        prob_a,
        "-",
        color="orange",
        linewidth=2,
        label=f"Probability({label_a})",
    )

    # Reciprocal rank (1/rank)
    reciprocal_rank = [1.0 / r for r in rank_a]
    line4 = ax2.plot(
        layers,
        reciprocal_rank,
        "--",
        color="brown",
        linewidth=2,
        label=f"Reciprocal rank({label_a})",
    )

    # Labels
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Logit difference / Logprob", fontsize=11)
    ax2.set_ylabel("Probability / Reciprocal rank", fontsize=11)

    ax.set_title(title, fontsize=11, fontweight="bold")

    # Grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=8)

    # Set y-axis limits
    ax2.set_ylim(0, 1.05)
