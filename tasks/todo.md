# Fine-Grained Patching Visualization Implementation

## Overview
Implement comprehensive fine-grained patching analysis and visualizations (Plots 17-26).

## Phase 1: Data Structures and Configuration

- [x] Create `src/intertemporal/experiments/fine_grained/` module
- [x] Define `FineGrainedConfig` dataclass with all parameters
- [x] Define result dataclasses:
  - [x] `HeadPatchingResult` (denoising/noising per head)
  - [x] `PositionPatchingResult` (per position per head)
  - [x] `PathPatchingResult` (head-to-component connectivity)
  - [x] `MultiSiteResult` (interaction effects)
  - [x] `NeuronPatchingResult` (per neuron at target layer)
  - [x] `FineGrainedResults` (aggregates all above)

## Phase 2: Analysis Functions

- [x] `run_head_patching_sweep()` - patch each head individually, both modes
- [x] `run_position_patching()` - for top heads, patch at each position
- [x] `run_path_patching()` - patch source, measure effect at destination
- [x] `run_multi_site_patching()` - joint patching for interaction effects
- [x] `run_neuron_patching()` - ablate individual neurons at target layer
- [x] `run_fine_grained_analysis()` - orchestrates all above

## Phase 3: Visualization Functions

- [x] Plot 17: Head-level heatmap (layers x heads, denoising/noising versions)
- [x] Plot 18: Head-level ranked bar chart with cumulative line
- [x] Plot 19: Head scatter (denoising vs noising)
- [x] Plot 20: Head x position heatmap for top heads
- [x] Plot 21: Path patching head-to-MLP matrix
- [x] Plot 22: Path patching head-to-head matrix
- [x] Plot 23: Multi-site interaction heatmap
- [x] Plot 24: Neuron-level ranked bar chart at L31
- [x] Plot 25: Cumulative neuron contribution curve
- [x] Plot 26: Layer-position fine heatmap (attn_out and mlp_out)

## Phase 4: Integration

- [x] Add `FINE_GRAINED` config to `experiment_config.py`
- [x] Add `--fine_grained` flag to CLI
- [x] Add `step_fine_grained_patching()` to experiment pipeline
- [x] Integrate visualizations into `generate_viz()`
- [x] Add save/load methods to `ExperimentContext`

## Phase 5: Testing

- [ ] Run with `--fine_grained '{}'` flag
- [ ] Verify all PNGs generated
- [ ] Check plot quality and labels

## Files Created/Modified

### New Files:
- `src/intertemporal/experiments/fine_grained/__init__.py`
- `src/intertemporal/experiments/fine_grained/fine_grained_config.py`
- `src/intertemporal/experiments/fine_grained/fine_grained_results.py`
- `src/intertemporal/experiments/fine_grained/fine_grained_analysis.py`
- `src/intertemporal/experiments/fine_grained/fine_grained_viz.py`

### Modified Files:
- `src/intertemporal/experiments/experiment_config.py` - Added FINE_GRAINED config
- `src/intertemporal/experiments/experiment_context.py` - Added fine_grained storage/load/save
- `src/intertemporal/experiments/intertemporal_experiment.py` - Added step_fine_grained_patching
- `src/intertemporal/experiments/intertemporal_viz.py` - Added fine_grained visualization
- `scripts/intertemporal/run_intertemporal_experiment.py` - Added --fine_grained flag

## Notes
- Extends existing fine patching infrastructure
- Uses existing `patch_for_choice()` for interventions
- Uses existing visualization helpers (finalize_plot, etc.)
- Numpy arrays are properly serialized to JSON via custom to_dict/from_dict
