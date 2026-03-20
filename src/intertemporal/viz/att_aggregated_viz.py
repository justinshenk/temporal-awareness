"""Aggregated visualization for attribution patching results.

Creates heatmaps organized by:
- Analysis slice (all, same_labels, etc.)
- Mode (denoising, noising)
- Component (resid_post, attn_out, mlp_out)
- Method (standard, eap, eap_ig)
- Gradient point (clean, corrupted)

Also generates analysis plots:
- Layer attribution line plots
- Score histograms
- Component comparison plots
- Top scores plots
"""

from __future__ import annotations

from pathlib import Path

from ...attribution_patching import AttrPatchAggregatedResults, AttributionSummary
from ...common import profile
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_multi_metric_heatmap,
    plot_patching_heatmap,
)
from .analysis_plots.main import generate_analysis_plots
from .att_patching_viz import _build_result_key, _discover_axes, parse_result_key
from .viz_config import (
    ATT_DEFAULT_COMPONENT,
    ATT_DEFAULT_GRAD_AT,
    ATT_DEFAULT_METHOD,
    ATT_DEFAULT_QUADRATURE,
    ATT_VIZ_AXES,
    ATT_VIZ_COMPARISON,
    CORE_SLICES,
    GENERATE_ALL_SLICES,
)


@profile
def visualize_att_aggregated(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
    slice_name: str = "all",
) -> None:
    """Visualize aggregated attribution patching results.

    Creates:
        output_dir/
          denoising/
            by_component/resid_post/{method}_{grad_at}.png
            by_method/eap_ig/{component}_{grad_at}.png
            comparison.png
          noising/
            ...

    Args:
        agg: Aggregated attribution results
        output_dir: Directory to save plots (e.g., agg/all/att_patching/)
        slice_name: Analysis slice name for titles
    """
    output_dir = Path(output_dir)

    # Denoising visualization
    if agg.denoising_agg:
        denoising_dir = output_dir / "denoising"
        _visualize_summary_structured(
            agg.denoising_agg,
            denoising_dir,
            mode="denoising",
            n_pairs=len(agg.denoising),
            slice_name=slice_name,
        )

    # Noising visualization
    if agg.noising_agg:
        noising_dir = output_dir / "noising"
        _visualize_summary_structured(
            agg.noising_agg,
            noising_dir,
            mode="noising",
            n_pairs=len(agg.noising),
            slice_name=slice_name,
        )

    # Analysis plots (layer line plots, histograms, comparisons, top scores)
    analysis_dir = output_dir / "analysis"
    generate_analysis_plots(agg, analysis_dir, title_prefix=f"{slice_name} | ")


def _visualize_summary_structured(
    summary: AttributionSummary,
    output_dir: Path,
    mode: str,
    n_pairs: int,
    slice_name: str,
) -> None:
    """Visualize a single AttributionSummary using 'vary one axis' approach.

    Instead of generating all combinations, uses the first value in each axis
    as the default, then generates plots varying ONE axis at a time.

    Creates:
        output_dir/
          by_method/
            standard.png, eap.png, eap_ig.png  (varying method, others default)
          by_component/
            resid_post.png, attn_out.png, mlp_out.png  (varying component)
          by_grad_at/
            clean.png, corrupted.png  (varying grad_at)
          by_quadrature/
            midpoint.png, gauss-legendre.png, ...  (varying quadrature)
          comparison.png

    Args:
        summary: Attribution summary to visualize
        output_dir: Directory to save plots
        mode: "denoising" or "noising"
        n_pairs: Number of pairs aggregated
        slice_name: Analysis slice name for titles
    """
    if not summary.results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common dimensions
    first_result = next(iter(summary.results.values()))
    layers = first_result.layers
    n_positions = first_result.n_positions
    pos_labels = [f"p{i}" for i in range(n_positions)]

    mode_label = "Denoise" if mode == "denoising" else "Noise"
    title_base = f"{mode_label} | {slice_name} | n={n_pairs}"

    # Discover axes
    methods, components, grad_ats, quadratures = _discover_axes(summary.results)

    # Use configured defaults, fallback to first available if not present
    default_method = ATT_DEFAULT_METHOD if ATT_DEFAULT_METHOD in methods else (methods[0] if methods else "standard")
    default_component = ATT_DEFAULT_COMPONENT if ATT_DEFAULT_COMPONENT in components else (components[0] if components else "resid_post")
    default_grad_at = ATT_DEFAULT_GRAD_AT if ATT_DEFAULT_GRAD_AT in grad_ats else (grad_ats[0] if grad_ats else "clean")
    default_quadrature = ATT_DEFAULT_QUADRATURE if ATT_DEFAULT_QUADRATURE in quadratures else (quadratures[0] if quadratures else None)

    def get_result(method: str, component: str, grad_at: str, quadrature: str | None):
        key = _build_result_key(method, component, grad_at, quadrature)
        return summary.results.get(key)

    def find_component_for_method(method: str) -> str | None:
        """Find a valid component for the given method."""
        # Try default first
        if get_result(method, default_component, default_grad_at, default_quadrature):
            return default_component
        # Try other components
        for comp in components:
            if get_result(method, comp, default_grad_at, default_quadrature):
                return comp
        return None

    def find_method_for_component(component: str) -> str | None:
        """Find a valid method for the given component."""
        # Try default first
        if get_result(default_method, component, default_grad_at, default_quadrature):
            return default_method
        # Try other methods (standard has all components, EAP only has attn/mlp)
        for method in methods:
            if get_result(method, component, default_grad_at, default_quadrature):
                return method
        return None

    def plot_single(attr_result, save_path: Path, title: str):
        if attr_result.scores.size == 0:
            return
        config = PatchingHeatmapConfig(
            title=f"{title} | {title_base}",
            subtitle=f"{attr_result.n_layers} layers, {n_positions} positions",
            cbar_label="Attribution Score",
            cmap="RdBu_r",
        )
        plot_patching_heatmap(
            attr_result.scores,
            layers,
            pos_labels,
            config=config,
            save_path=save_path,
        )

    # 1. By method (varying method, others default)
    # Note: EAP methods don't have resid_post, so find a valid component for each method
    if "method" in ATT_VIZ_AXES and len(methods) > 1:
        by_method_dir = output_dir / "by_method"
        by_method_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            comp = find_component_for_method(method)
            if comp:
                attr_result = get_result(method, comp, default_grad_at, default_quadrature)
                if attr_result:
                    title = f"{method} | {comp}"
                    plot_single(attr_result, by_method_dir / f"{method}.png", title)

    # 2. By component (varying component, others default)
    # Note: Standard has resid_post but EAP doesn't, so find a valid method for each component
    if "component" in ATT_VIZ_AXES and len(components) > 1:
        by_comp_dir = output_dir / "by_component"
        by_comp_dir.mkdir(parents=True, exist_ok=True)
        for component in components:
            method = find_method_for_component(component)
            if method:
                attr_result = get_result(method, component, default_grad_at, default_quadrature)
                if attr_result:
                    title = f"{method} | {component}"
                    plot_single(attr_result, by_comp_dir / f"{component}.png", title)

    # 3. By grad_at (varying grad_at, others default)
    if "grad_at" in ATT_VIZ_AXES and len(grad_ats) > 1:
        by_grad_dir = output_dir / "by_grad_at"
        by_grad_dir.mkdir(parents=True, exist_ok=True)
        for grad_at in grad_ats:
            attr_result = get_result(default_method, default_component, grad_at, default_quadrature)
            if attr_result:
                title = f"{default_method} | grad@{grad_at}"
                plot_single(attr_result, by_grad_dir / f"{grad_at}.png", title)

    # 4. By quadrature (varying quadrature, others default)
    if "quadrature" in ATT_VIZ_AXES and len(quadratures) > 1:
        by_quad_dir = output_dir / "by_quadrature"
        by_quad_dir.mkdir(parents=True, exist_ok=True)
        for quadrature in quadratures:
            attr_result = get_result(default_method, default_component, default_grad_at, quadrature)
            if attr_result:
                title = f"{default_method} | {quadrature}"
                plot_single(attr_result, by_quad_dir / f"{quadrature}.png", title)

    # 5. Multi-method comparison (limited to default component/grad_at/quadrature)
    if ATT_VIZ_COMPARISON and len(methods) > 1:
        matrices = {}
        for method in methods:
            attr_result = get_result(method, default_component, default_grad_at, default_quadrature)
            if attr_result and attr_result.scores.size > 0:
                matrices[method] = attr_result.scores

        if matrices:
            plot_multi_metric_heatmap(
                matrices,
                layers,
                pos_labels,
                save_path=output_dir / "comparison.png",
            )

    print(f"[viz] Attribution aggregated ({mode}) plots saved to {output_dir}")


@profile
def visualize_all_att_aggregated_slices(
    agg: AttrPatchAggregatedResults,
    output_dir: Path,
) -> None:
    """Visualize aggregated attribution results for all analysis slices.

    Creates:
        output_dir/
          all/att_patching/
            denoising/by_component/..., by_method/..., comparison.png
            noising/...
          same_labels/att_patching/...
          ... (other slices)

    Args:
        agg: Aggregated attribution results
        output_dir: Base output directory (typically agg/)
    """
    from .coarse.aggregated.analysis_slices import ANALYSIS_SLICES

    output_dir = Path(output_dir)

    # Determine which slices to generate
    n_samples = len(agg.denoising) or len(agg.noising)
    if not GENERATE_ALL_SLICES or n_samples <= 2:
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name in CORE_SLICES]
    else:
        slices_to_generate = ANALYSIS_SLICES

    # Generate plots for each analysis slice
    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name

        visualize_att_aggregated(agg, slice_dir, slice_name)

    print(f"[viz] All attribution aggregated slices saved to {output_dir}")
