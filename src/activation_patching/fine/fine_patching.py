"""Main orchestration for fine-grained activation patching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...common.contrastive_pair import ContrastivePair
from ...common.device_utils import clear_gpu_memory
from ...common.profiler import profile
from ...common.patching_types import PatchingMode

from .fine_config import FineConfig, DEFAULT_FINE_CONFIG
from .fine_results import FinePatchingResults
from .head_patching import run_head_patching
from .mlp_analysis import run_mlp_neuron_analysis
from .attention_analysis import analyze_attention_patterns

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner


@profile
def run_fine_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    config: FineConfig | None = None,
    skip_head_patching: bool = False,
    skip_mlp_analysis: bool = False,
    skip_attention_analysis: bool = False,
) -> FinePatchingResults:
    """Run all fine-grained analyses on a contrastive pair.

    Performs:
    1. Head-level patching at key attention layers (L24, L21, L19, L29, L30)
    2. MLP neuron analysis at key MLP layers (L31, L24, L28)
    3. Attention pattern analysis for top heads found in step 1

    Args:
        runner: Model runner
        pair: Contrastive pair
        config: Fine patching configuration
        skip_head_patching: Skip head-level analysis
        skip_mlp_analysis: Skip MLP neuron analysis
        skip_attention_analysis: Skip attention pattern analysis

    Returns:
        FinePatchingResults with all analysis results
    """
    if config is None:
        config = DEFAULT_FINE_CONFIG

    results = FinePatchingResults(
        sample_id=pair.sample_id,
        n_layers=runner.n_layers,
        n_heads=runner._backend.get_n_heads(),
        d_head=runner._backend.get_d_head(),
        d_mlp=runner._backend.get_d_mlp(),
    )

    # Run both denoising and noising modes
    for mode in ["denoising", "noising"]:
        mode_suffix = f"_{mode}"

        # 1. Head-level patching
        if not skip_head_patching:
            head_results = run_head_patching(runner, pair, mode, config)
            # Store with mode suffix to differentiate
            for layer, layer_results in head_results.items():
                # Merge with existing results or store new
                if layer not in results.head_results:
                    results.head_results[layer] = layer_results
                else:
                    # Average scores across modes
                    for hr, existing_hr in zip(
                        layer_results.head_results,
                        results.head_results[layer].head_results
                    ):
                        if mode == "denoising":
                            existing_hr.denoising_score = hr.score
                        else:
                            existing_hr.noising_score = hr.score
                        # Update combined score
                        existing_hr.score = (
                            existing_hr.denoising_score + existing_hr.noising_score
                        ) / 2

        # 2. MLP neuron analysis
        if not skip_mlp_analysis:
            mlp_results = run_mlp_neuron_analysis(runner, pair, mode, config)
            # Store MLP results (similar merging logic could be added)
            for layer, layer_results in mlp_results.items():
                if layer not in results.mlp_results:
                    results.mlp_results[layer] = layer_results

    # 3. Attention pattern analysis (uses top heads from head patching)
    if not skip_attention_analysis and not skip_head_patching:
        top_heads = results.get_top_heads_all_layers(config.n_top_heads)
        attention_patterns = analyze_attention_patterns(
            runner, pair, top_heads, config
        )
        results.attention_patterns = attention_patterns

    clear_gpu_memory()
    return results


@profile
def run_fine_patching_batch(
    runner: "BinaryChoiceRunner",
    pairs: list[ContrastivePair],
    config: FineConfig | None = None,
) -> list[FinePatchingResults]:
    """Run fine-grained patching on multiple pairs.

    Args:
        runner: Model runner
        pairs: List of contrastive pairs
        config: Fine patching configuration

    Returns:
        List of FinePatchingResults, one per pair
    """
    if config is None:
        config = DEFAULT_FINE_CONFIG

    results = []
    for pair in pairs:
        result = run_fine_patching(runner, pair, config)
        results.append(result)

    return results
