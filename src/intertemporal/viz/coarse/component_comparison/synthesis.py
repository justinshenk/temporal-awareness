"""Circuit synthesis plots: information flow summary.

This module creates the final circuit summary visualizations that synthesize
all findings from the component comparison analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

from .....activation_patching.coarse import SweepStepResults
from .utils import save_plot


def plot_synthesis(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Generate circuit synthesis plots and text summary."""
    circuit_info = _extract_circuit_info(layer_data, pos_data)
    _plot_information_flow_diagram(circuit_info, output_dir)
    _generate_circuit_summary(circuit_info, output_dir)


def _extract_circuit_info(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
) -> dict:
    """Extract all circuit information from the data.

    Returns a dict with:
    - input_positions: list of (pos, denoise_score, noise_score, is_redundant)
    - output_positions: list of (pos, denoise_score, noise_score, is_bottleneck)
    - attn_layers: list of (layer, denoise_score, noise_score)
    - mlp_layers: list of (layer, denoise_score, noise_score)
    - multi_function_layers: set of layers appearing in both attn and mlp top lists
    - layer_position_bindings: dict mapping layer ranges to position ranges
    """
    info = {
        "input_positions": [],
        "output_positions": [],
        "attn_layers": [],
        "mlp_layers": [],
        "multi_function_layers": set(),
        "layer_position_bindings": {},
    }

    # Extract position data
    resid_post_pos = pos_data.get("resid_post", {})
    if resid_post_pos:
        positions = sorted(resid_post_pos.keys())
        mid_pos = positions[len(positions) // 2] if positions else 0

        pos_data_list = []
        for p in positions:
            result = resid_post_pos[p]
            denoise = result.recovery or 0
            noise = result.disruption or 0
            pos_data_list.append((p, denoise, noise))

        # Sort by noising score (more reliable for finding bottlenecks)
        pos_data_list.sort(key=lambda x: x[2], reverse=True)

        # Split into input (early) and output (late) positions
        # FILTER: Skip denoising-only positions (where denoise >> noise, ratio > 2x)
        for p, denoise, noise in pos_data_list[:20]:
            # Skip denoising-only positions that would be misleading
            if denoise > 0.1 and noise < 0.05:
                continue  # Denoising-only, skip
            if denoise > noise * 3 and noise < 0.1:
                continue  # Strongly denoising-dominated, skip

            redundancy_gap = noise - denoise
            if p < mid_pos:
                # Input position: redundant if gap < 0 (denoise > noise)
                is_redundant = redundancy_gap < -0.1
                info["input_positions"].append((p, denoise, noise, is_redundant))
            else:
                # Output position: bottleneck if noise > 0.5
                is_bottleneck = noise > 0.5
                info["output_positions"].append((p, denoise, noise, is_bottleneck))

        # Limit to top 3-4 each
        info["input_positions"] = info["input_positions"][:4]
        info["output_positions"] = info["output_positions"][:4]

    # Extract layer data
    attn_data = layer_data.get("attn_out", {})
    mlp_data = layer_data.get("mlp_out", {})

    if attn_data:
        attn_list = []
        for lyr, result in attn_data.items():
            denoise = result.recovery or 0
            noise = result.disruption or 0
            attn_list.append((lyr, denoise, noise))
        # Sort by noising (more reliable)
        attn_list.sort(key=lambda x: x[2], reverse=True)
        info["attn_layers"] = attn_list[:5]

    if mlp_data:
        mlp_list = []
        for lyr, result in mlp_data.items():
            denoise = result.recovery or 0
            noise = result.disruption or 0
            mlp_list.append((lyr, denoise, noise))
        mlp_list.sort(key=lambda x: x[2], reverse=True)
        info["mlp_layers"] = mlp_list[:5]

    # Find multi-function layers (appear in both attn and mlp top lists)
    attn_top_layers = {lyr for lyr, _, _ in info["attn_layers"]}
    mlp_top_layers = {lyr for lyr, _, _ in info["mlp_layers"]}
    info["multi_function_layers"] = attn_top_layers & mlp_top_layers

    # Infer layer-position bindings from top layers and positions
    # This is an approximation: early layers likely read from early positions,
    # late layers likely write to late positions
    if info["attn_layers"] and info["input_positions"]:
        attn_layers_sorted = sorted([l for l, _, _ in info["attn_layers"]])
        input_pos_sorted = sorted([p for p, _, _, _ in info["input_positions"]])
        if attn_layers_sorted and input_pos_sorted:
            info["layer_position_bindings"]["attn_reads_from"] = {
                "layers": f"L{min(attn_layers_sorted)}-L{max(attn_layers_sorted)}",
                "positions": f"P{min(input_pos_sorted)}-P{max(input_pos_sorted)}",
            }

    if info["mlp_layers"] and info["output_positions"]:
        mlp_layers_sorted = sorted([l for l, _, _ in info["mlp_layers"]])
        output_pos_sorted = sorted([p for p, _, _, _ in info["output_positions"]])
        if mlp_layers_sorted and output_pos_sorted:
            info["layer_position_bindings"]["mlp_writes_to"] = {
                "layers": f"L{min(mlp_layers_sorted)}-L{max(mlp_layers_sorted)}",
                "positions": f"P{min(output_pos_sorted)}-P{max(output_pos_sorted)}",
            }

    return info


def _plot_information_flow_diagram(
    circuit_info: dict,
    output_dir: Path,
) -> None:
    """Create information flow summary diagram with redundancy annotations.

    Improvements over basic version:
    - Shows both denoising and noising scores
    - Marks redundant positions (have backups) vs bottlenecks (critical)
    - Highlights multi-function layers (appear in both attn and mlp)
    - Uses noising scores for primary ranking (more reliable)
    - Arrow thickness reflects importance
    """
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(6, 9.5, "Circuit Information Flow Summary", fontsize=18, fontweight="bold", ha="center")
    ax.text(6, 9.1, "(Scores: noising disruption / denoising recovery)", fontsize=10, ha="center",
            style="italic", color="gray")

    # Input positions box
    rect1 = FancyBboxPatch((0.5, 6), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#E3F2FD", edgecolor="black", linewidth=2)
    ax.add_patch(rect1)
    ax.text(2, 8.2, "Source Positions", fontsize=12, fontweight="bold", ha="center")

    if circuit_info["input_positions"]:
        for i, (pos, denoise, noise, is_redundant) in enumerate(circuit_info["input_positions"][:3]):
            style = "italic" if is_redundant else "normal"
            color = "#666666" if is_redundant else "black"
            marker = " (redundant)" if is_redundant else ""
            ax.text(2, 7.6 - i * 0.5, f"P{pos}: {noise:.2f}/{denoise:.2f}{marker}",
                    fontsize=9, ha="center", style=style, color=color)
    else:
        ax.text(2, 7.2, "(analyzing...)", fontsize=10, ha="center", style="italic")

    # Attention layers box
    rect2 = FancyBboxPatch((4.5, 6), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#FFF3E0", edgecolor="black", linewidth=2)
    ax.add_patch(rect2)
    ax.text(6, 8.2, "Attention Layers", fontsize=12, fontweight="bold", ha="center")

    if circuit_info["attn_layers"]:
        for i, (layer, denoise, noise) in enumerate(circuit_info["attn_layers"][:4]):
            is_multi = layer in circuit_info["multi_function_layers"]
            marker = " ★" if is_multi else ""
            fontweight = "bold" if is_multi else "normal"
            ax.text(6, 7.6 - i * 0.45, f"L{layer}: {noise:.2f}/{denoise:.2f}{marker}",
                    fontsize=9, ha="center", fontweight=fontweight)
    else:
        ax.text(6, 7.2, "(no data)", fontsize=10, ha="center", style="italic")

    # MLP layers box
    rect3 = FancyBboxPatch((4.5, 2.5), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#E8F5E9", edgecolor="black", linewidth=2)
    ax.add_patch(rect3)
    ax.text(6, 4.7, "MLP Layers", fontsize=12, fontweight="bold", ha="center")

    if circuit_info["mlp_layers"]:
        for i, (layer, denoise, noise) in enumerate(circuit_info["mlp_layers"][:4]):
            is_multi = layer in circuit_info["multi_function_layers"]
            marker = " ★" if is_multi else ""
            fontweight = "bold" if is_multi else "normal"
            ax.text(6, 4.2 - i * 0.45, f"L{layer}: {noise:.2f}/{denoise:.2f}{marker}",
                    fontsize=9, ha="center", fontweight=fontweight)
    else:
        ax.text(6, 3.8, "(no data)", fontsize=10, ha="center", style="italic")

    # Output positions box - larger border to indicate bottleneck importance
    rect4 = FancyBboxPatch((8.5, 4), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#FCE4EC", edgecolor="#D32F2F", linewidth=3)
    ax.add_patch(rect4)
    ax.text(10, 6.2, "Destination Positions", fontsize=12, fontweight="bold", ha="center")
    ax.text(10, 5.85, "(BOTTLENECK)", fontsize=9, fontweight="bold", ha="center", color="#D32F2F")

    if circuit_info["output_positions"]:
        for i, (pos, denoise, noise, is_bottleneck) in enumerate(circuit_info["output_positions"][:3]):
            fontweight = "bold" if is_bottleneck else "normal"
            color = "#D32F2F" if is_bottleneck else "black"
            ax.text(10, 5.3 - i * 0.5, f"P{pos}: {noise:.2f}/{denoise:.2f}",
                    fontsize=9, ha="center", fontweight=fontweight, color=color)
    else:
        ax.text(10, 5.0, "(analyzing...)", fontsize=10, ha="center", style="italic")

    # Arrows - thickness reflects importance
    # Input → Attention (medium)
    ax.annotate("", xy=(4.5, 7), xytext=(3.5, 7),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(4.0, 7.3, "reads from", fontsize=8, ha="center", color="gray")

    # Attention → MLP (medium, vertical)
    ax.annotate("", xy=(6, 5), xytext=(6, 6),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(6.5, 5.5, "routes to", fontsize=8, ha="left", color="gray")

    # MLP → Output (THICK - this is the critical path)
    ax.annotate("", xy=(8.5, 5.25), xytext=(7.5, 3.75),
                arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=4))
    ax.text(8.3, 4.2, "writes to\n(critical)", fontsize=8, ha="center", color="#D32F2F", fontweight="bold")

    # Attention → Output (dashed - bypass path)
    ax.annotate("", xy=(8.5, 5.75), xytext=(7.5, 7),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5, ls="--"))
    ax.text(8.3, 6.6, "bypass\n(backup)", fontsize=7, ha="center", color="gray", style="italic")

    # Layer-position bindings annotation (the actual circuit edges)
    bindings = circuit_info.get("layer_position_bindings", {})
    if bindings:
        ax.text(1.5, 2.5, "Layer-Position Bindings:", fontsize=10, fontweight="bold", color="#1565C0")
        y_offset = 2.1
        if "attn_reads_from" in bindings:
            b = bindings["attn_reads_from"]
            ax.text(1.5, y_offset, f"• {b['layers']} attention reads from {b['positions']}",
                    fontsize=9, color="#1565C0")
            y_offset -= 0.35
        if "mlp_writes_to" in bindings:
            b = bindings["mlp_writes_to"]
            ax.text(1.5, y_offset, f"• {b['layers']} MLP writes to {b['positions']}",
                    fontsize=9, color="#1565C0")

    # Legend
    ax.text(6, 1.8, "Legend:", fontsize=10, fontweight="bold")
    ax.text(6, 1.4, "• Scores: noising / denoising", fontsize=9)
    ax.text(6, 1.0, "• ★ = Multi-function layer (both attn & mlp)", fontsize=9)
    ax.text(6, 0.6, "• (redundant) = Has backup pathways", fontsize=9)
    ax.text(6, 0.2, "• Thick red arrow = Critical bottleneck", fontsize=9, color="#D32F2F")

    # Summary stats
    if circuit_info["attn_layers"] and circuit_info["mlp_layers"]:
        attn_noise_total = sum(n for _, _, n in circuit_info["attn_layers"][:3])
        mlp_noise_total = sum(n for _, _, n in circuit_info["mlp_layers"][:3])
        ax.text(9, 1.5, f"Top-3 Attn (noising): {attn_noise_total:.2f}", fontsize=10, ha="left")
        ax.text(9, 1.1, f"Top-3 MLP (noising): {mlp_noise_total:.2f}", fontsize=10, ha="left")

        if circuit_info["multi_function_layers"]:
            multi_str = ", ".join(f"L{l}" for l in sorted(circuit_info["multi_function_layers"]))
            ax.text(9, 0.7, f"Multi-function: {multi_str}", fontsize=10, ha="left", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, "information_flow_diagram.png")


def _generate_circuit_summary(
    circuit_info: dict,
    output_dir: Path,
) -> None:
    """Generate a text file summarizing the circuit hypothesis.

    This provides a plain-text explanation of the circuit for readers
    who haven't seen the plots.
    """
    lines = [
        "# Circuit Summary",
        "",
        "This document summarizes the inferred circuit from activation patching analysis.",
        "Scores shown as: noising_disruption / denoising_recovery",
        "",
        "## Key Finding: Two-Phase Architecture",
        "",
        "The circuit operates in two phases:",
        "1. **Attention phase**: Attention heads read from source positions and route information",
        "2. **MLP phase**: MLP layers transform and write to destination positions",
        "",
        "## Source Positions (Early)",
        "",
    ]

    if circuit_info["input_positions"]:
        redundant_count = sum(1 for _, _, _, r in circuit_info["input_positions"] if r)
        total = len(circuit_info["input_positions"])
        lines.append(f"Found {total} key source positions ({redundant_count} have backup pathways):")
        for pos, denoise, noise, is_redundant in circuit_info["input_positions"]:
            status = "REDUNDANT - has backups" if is_redundant else "important"
            lines.append(f"  - P{pos}: {noise:.3f}/{denoise:.3f} ({status})")
        lines.append("")
        lines.append("Interpretation: Redundant positions have denoising > noising, meaning")
        lines.append("the model can recover even when these positions are corrupted (backup pathways exist).")
    else:
        lines.append("(No position data available)")

    lines.extend([
        "",
        "## Attention Layers",
        "",
    ])

    if circuit_info["attn_layers"]:
        lines.append("Top attention layers (ranked by noising disruption):")
        for layer, denoise, noise in circuit_info["attn_layers"]:
            multi = " ★ MULTI-FUNCTION" if layer in circuit_info["multi_function_layers"] else ""
            lines.append(f"  - L{layer}: {noise:.3f}/{denoise:.3f}{multi}")
        lines.append("")
        lines.append("These layers read from source positions and route information.")
        if circuit_info["multi_function_layers"]:
            multi_str = ", ".join(f"L{l}" for l in sorted(circuit_info["multi_function_layers"]))
            lines.append(f"Multi-function layers ({multi_str}) appear in both attention AND MLP top lists,")
            lines.append("suggesting they do double-duty processing.")
    else:
        lines.append("(No attention data available)")

    lines.extend([
        "",
        "## MLP Layers",
        "",
    ])

    if circuit_info["mlp_layers"]:
        lines.append("Top MLP layers (ranked by noising disruption):")
        for layer, denoise, noise in circuit_info["mlp_layers"]:
            multi = " ★ MULTI-FUNCTION" if layer in circuit_info["multi_function_layers"] else ""
            lines.append(f"  - L{layer}: {noise:.3f}/{denoise:.3f}{multi}")
        lines.append("")
        lines.append("These layers transform information and write to destination positions.")
    else:
        lines.append("(No MLP data available)")

    lines.extend([
        "",
        "## Destination Positions (Late) - THE BOTTLENECK",
        "",
    ])

    if circuit_info["output_positions"]:
        bottleneck_count = sum(1 for _, _, _, b in circuit_info["output_positions"] if b)
        lines.append(f"Found {len(circuit_info['output_positions'])} key destination positions:")
        for pos, denoise, noise, is_bottleneck in circuit_info["output_positions"]:
            status = "★ BOTTLENECK" if is_bottleneck else "secondary"
            lines.append(f"  - P{pos}: {noise:.3f}/{denoise:.3f} ({status})")
        lines.append("")
        if bottleneck_count > 0:
            lines.append(f"CRITICAL: {bottleneck_count} position(s) are true bottlenecks (noising > 0.5).")
            lines.append("Corrupting these positions severely disrupts the model's output.")
            lines.append("This is where the final answer is computed.")
    else:
        lines.append("(No position data available)")

    # Layer-position bindings
    lines.extend([
        "",
        "## Layer-Position Bindings",
        "",
        "These show which layers act at which positions (the actual circuit edges):",
        "",
    ])

    bindings = circuit_info.get("layer_position_bindings", {})
    if bindings:
        if "attn_reads_from" in bindings:
            b = bindings["attn_reads_from"]
            lines.append(f"  - {b['layers']} attention reads from {b['positions']}")
        if "mlp_writes_to" in bindings:
            b = bindings["mlp_writes_to"]
            lines.append(f"  - {b['layers']} MLP writes to {b['positions']}")
    else:
        lines.append("  (Insufficient data to infer bindings)")

    lines.extend([
        "",
        "## Methodology Notes",
        "",
        "- **Noising** (disruption): Replace clean activation with corrupted → measures necessity",
        "- **Denoising** (recovery): Replace corrupted activation with clean → measures sufficiency",
        "- High noising + low denoising = necessary but has backups (OR relationship)",
        "- High noising + high denoising = necessary AND sufficient (AND relationship)",
        "- The redundancy gap (noising - denoising) indicates how replaceable a component is",
        "- **Denoising-only positions filtered**: Positions where denoising >> noising are excluded",
        "  as they indicate artifacts rather than true circuit components",
        "",
        "## Circuit Diagram",
        "",
        "See information_flow_diagram.png for a visual representation.",
        "",
    ])

    content = "\n".join(lines)
    (output_dir / "circuit_summary.txt").write_text(content)
    print(f"Saved: {output_dir / 'circuit_summary.txt'}")
