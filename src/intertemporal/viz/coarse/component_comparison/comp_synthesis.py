"""Circuit synthesis plots: information flow summary.

This module creates the final circuit summary visualizations that synthesize
all findings from the component comparison analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .....activation_patching.coarse import SweepStepResults
from .....intertemporal.experiments.processing import (
    CircuitHypothesis,
    ComponentComparisonResults,
    extract_circuit_hypothesis,
)
from .comp_utils import save_plot


def plot_synthesis(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    processed_results: ComponentComparisonResults | None = None,
) -> None:
    """Generate circuit synthesis plots and text summary."""
    # Use pre-computed circuit hypothesis if available
    if processed_results and processed_results.circuit:
        circuit = processed_results.circuit
    else:
        # Fallback to computing inline (for backwards compatibility)
        circuit = extract_circuit_hypothesis(layer_data, pos_data)

    _plot_information_flow_diagram(circuit, output_dir)
    _generate_circuit_summary(circuit, output_dir)


def _plot_information_flow_diagram(
    circuit: CircuitHypothesis,
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

    if circuit.source_positions:
        for i, pos_info in enumerate(circuit.source_positions[:3]):
            style = "italic" if pos_info.is_redundant else "normal"
            color = "#666666" if pos_info.is_redundant else "black"
            marker = " (redundant)" if pos_info.is_redundant else ""
            ax.text(2, 7.6 - i * 0.5, f"P{pos_info.position}: {pos_info.disruption:.2f}/{pos_info.recovery:.2f}{marker}",
                    fontsize=9, ha="center", style=style, color=color)
    else:
        ax.text(2, 7.2, "(analyzing...)", fontsize=10, ha="center", style="italic")

    # Attention layers box
    rect2 = FancyBboxPatch((4.5, 6), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#FFF3E0", edgecolor="black", linewidth=2)
    ax.add_patch(rect2)
    ax.text(6, 8.2, "Attention Layers", fontsize=12, fontweight="bold", ha="center")

    multi_func_set = set(circuit.multi_function_layers)
    if circuit.attn_layers:
        for i, layer_info in enumerate(circuit.attn_layers[:4]):
            is_multi = layer_info.layer in multi_func_set
            marker = " ★" if is_multi else ""
            fontweight = "bold" if is_multi else "normal"
            ax.text(6, 7.6 - i * 0.45, f"L{layer_info.layer}: {layer_info.disruption:.2f}/{layer_info.recovery:.2f}{marker}",
                    fontsize=9, ha="center", fontweight=fontweight)
    else:
        ax.text(6, 7.2, "(no data)", fontsize=10, ha="center", style="italic")

    # MLP layers box
    rect3 = FancyBboxPatch((4.5, 2.5), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#E8F5E9", edgecolor="black", linewidth=2)
    ax.add_patch(rect3)
    ax.text(6, 4.7, "MLP Layers", fontsize=12, fontweight="bold", ha="center")

    if circuit.mlp_layers:
        for i, layer_info in enumerate(circuit.mlp_layers[:4]):
            is_multi = layer_info.layer in multi_func_set
            marker = " ★" if is_multi else ""
            fontweight = "bold" if is_multi else "normal"
            ax.text(6, 4.2 - i * 0.45, f"L{layer_info.layer}: {layer_info.disruption:.2f}/{layer_info.recovery:.2f}{marker}",
                    fontsize=9, ha="center", fontweight=fontweight)
    else:
        ax.text(6, 3.8, "(no data)", fontsize=10, ha="center", style="italic")

    # Output positions box - larger border to indicate bottleneck importance
    rect4 = FancyBboxPatch((8.5, 4), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor="#FCE4EC", edgecolor="#D32F2F", linewidth=3)
    ax.add_patch(rect4)
    ax.text(10, 6.2, "Destination Positions", fontsize=12, fontweight="bold", ha="center")
    ax.text(10, 5.85, "(BOTTLENECK)", fontsize=9, fontweight="bold", ha="center", color="#D32F2F")

    if circuit.destination_positions:
        for i, pos_info in enumerate(circuit.destination_positions[:3]):
            fontweight = "bold" if pos_info.is_bottleneck else "normal"
            color = "#D32F2F" if pos_info.is_bottleneck else "black"
            ax.text(10, 5.3 - i * 0.5, f"P{pos_info.position}: {pos_info.disruption:.2f}/{pos_info.recovery:.2f}",
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
    if circuit.layer_position_bindings:
        ax.text(1.5, 2.5, "Layer-Position Bindings:", fontsize=10, fontweight="bold", color="#1565C0")
        y_offset = 2.1
        for binding in circuit.layer_position_bindings:
            if binding.binding_type == "attn_reads_from":
                ax.text(1.5, y_offset, f"• {binding.layers} attention reads from {binding.positions}",
                        fontsize=9, color="#1565C0")
                y_offset -= 0.35
            elif binding.binding_type == "mlp_writes_to":
                ax.text(1.5, y_offset, f"• {binding.layers} MLP writes to {binding.positions}",
                        fontsize=9, color="#1565C0")
                y_offset -= 0.35

    # Legend
    ax.text(6, 1.8, "Legend:", fontsize=10, fontweight="bold")
    ax.text(6, 1.4, "• Scores: noising / denoising", fontsize=9)
    ax.text(6, 1.0, "• ★ = Multi-function layer (both attn & mlp)", fontsize=9)
    ax.text(6, 0.6, "• (redundant) = Has backup pathways", fontsize=9)
    ax.text(6, 0.2, "• Thick red arrow = Critical bottleneck", fontsize=9, color="#D32F2F")

    # Summary stats
    if circuit.attn_layers and circuit.mlp_layers:
        ax.text(9, 1.5, f"Top-3 Attn (noising): {circuit.top3_attn_noising:.2f}", fontsize=10, ha="left")
        ax.text(9, 1.1, f"Top-3 MLP (noising): {circuit.top3_mlp_noising:.2f}", fontsize=10, ha="left")

        if circuit.multi_function_layers:
            multi_str = ", ".join(f"L{l}" for l in sorted(circuit.multi_function_layers))
            ax.text(9, 0.7, f"Multi-function: {multi_str}", fontsize=10, ha="left", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, "information_flow_diagram.png")


def _generate_circuit_summary(
    circuit: CircuitHypothesis,
    output_dir: Path,
) -> None:
    """Generate a text file summarizing the circuit hypothesis.

    This provides a plain-text explanation of the circuit for readers
    who haven't seen the plots.
    """
    multi_func_set = set(circuit.multi_function_layers)

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

    if circuit.source_positions:
        redundant_count = sum(1 for p in circuit.source_positions if p.is_redundant)
        total = len(circuit.source_positions)
        lines.append(f"Found {total} key source positions ({redundant_count} have backup pathways):")
        for pos_info in circuit.source_positions:
            status = "REDUNDANT - has backups" if pos_info.is_redundant else "important"
            lines.append(f"  - P{pos_info.position}: {pos_info.disruption:.3f}/{pos_info.recovery:.3f} ({status})")
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

    if circuit.attn_layers:
        lines.append("Top attention layers (ranked by noising disruption):")
        for layer_info in circuit.attn_layers:
            multi = " ★ MULTI-FUNCTION" if layer_info.layer in multi_func_set else ""
            lines.append(f"  - L{layer_info.layer}: {layer_info.disruption:.3f}/{layer_info.recovery:.3f}{multi}")
        lines.append("")
        lines.append("These layers read from source positions and route information.")
        if circuit.multi_function_layers:
            multi_str = ", ".join(f"L{l}" for l in sorted(circuit.multi_function_layers))
            lines.append(f"Multi-function layers ({multi_str}) appear in both attention AND MLP top lists,")
            lines.append("suggesting they do double-duty processing.")
    else:
        lines.append("(No attention data available)")

    lines.extend([
        "",
        "## MLP Layers",
        "",
    ])

    if circuit.mlp_layers:
        lines.append("Top MLP layers (ranked by noising disruption):")
        for layer_info in circuit.mlp_layers:
            multi = " ★ MULTI-FUNCTION" if layer_info.layer in multi_func_set else ""
            lines.append(f"  - L{layer_info.layer}: {layer_info.disruption:.3f}/{layer_info.recovery:.3f}{multi}")
        lines.append("")
        lines.append("These layers transform information and write to destination positions.")
    else:
        lines.append("(No MLP data available)")

    lines.extend([
        "",
        "## Destination Positions (Late) - THE BOTTLENECK",
        "",
    ])

    if circuit.destination_positions:
        bottleneck_count = sum(1 for p in circuit.destination_positions if p.is_bottleneck)
        lines.append(f"Found {len(circuit.destination_positions)} key destination positions:")
        for pos_info in circuit.destination_positions:
            status = "★ BOTTLENECK" if pos_info.is_bottleneck else "secondary"
            lines.append(f"  - P{pos_info.position}: {pos_info.disruption:.3f}/{pos_info.recovery:.3f} ({status})")
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

    if circuit.layer_position_bindings:
        for binding in circuit.layer_position_bindings:
            if binding.binding_type == "attn_reads_from":
                lines.append(f"  - {binding.layers} attention reads from {binding.positions}")
            elif binding.binding_type == "mlp_writes_to":
                lines.append(f"  - {binding.layers} MLP writes to {binding.positions}")
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
