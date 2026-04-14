# v6: Clean→Fatigued Activation Patching Experiments

Tests whether substituting clean-context activations into fatigued-context prediction sites can move accuracy, confidence, or calibration gap. v6-1 patches only the within_task_fraction direction; v6-2 patches the full L0 residual at each prediction site.

## Phase A: Clean baseline (single-case forward passes)

- Total test cases run (clean context, no ICL): **90**
- Correct predictions: **45** (50.0%)

Note: clean accuracy is expected to be lower than fatigued accuracy (no ICL context). Fatigued no_patch accuracy is shown in the global table below.

## Global summary table

| condition     |   n |   accuracy |   mean_confidence |   calibration_gap |
|:--------------|----:|-----------:|------------------:|------------------:|
| no_patch      |  90 |     0.6667 |            0.927  |            0.2603 |
| patch_wf_L0   |  90 |     0.6667 |            0.9294 |            0.2628 |
| patch_wf_L10  |  90 |     0.6667 |            0.9275 |            0.2609 |
| patch_L0_full |  90 |     0.6667 |            0.9282 |            0.2616 |

## Flip stats vs no_patch

| condition     |   wrong_to_right |   right_to_wrong |   net_flips |
|:--------------|-----------------:|-----------------:|------------:|
| patch_wf_L0   |                0 |                0 |           0 |
| patch_wf_L10  |                0 |                0 |           0 |
| patch_L0_full |                0 |                0 |           0 |

## A4-style binned table (by within_task_fraction readout, no_patch wf)

### no_patch

| condition   |   wf_bin |   n |   accuracy |   mean_confidence |   calibration_gap |
|:------------|---------:|----:|-----------:|------------------:|------------------:|
| no_patch    |        0 |  18 |     0.8333 |            0.8986 |            0.0652 |
| no_patch    |        1 |  18 |     0.5556 |            0.87   |            0.3144 |
| no_patch    |        2 |  18 |     0.6111 |            0.9819 |            0.3708 |
| no_patch    |        3 |  18 |     0.6111 |            0.9159 |            0.3048 |
| no_patch    |        4 |  18 |     0.7222 |            0.9685 |            0.2463 |

### patch_wf_L0

| condition   |   wf_bin |   n |   accuracy |   mean_confidence |   calibration_gap |
|:------------|---------:|----:|-----------:|------------------:|------------------:|
| patch_wf_L0 |        0 |  18 |     0.8333 |            0.9039 |            0.0705 |
| patch_wf_L0 |        1 |  18 |     0.5556 |            0.8691 |            0.3136 |
| patch_wf_L0 |        2 |  18 |     0.6111 |            0.9826 |            0.3714 |
| patch_wf_L0 |        3 |  18 |     0.6111 |            0.9203 |            0.3092 |
| patch_wf_L0 |        4 |  18 |     0.7222 |            0.9714 |            0.2491 |

### patch_wf_L10

| condition    |   wf_bin |   n |   accuracy |   mean_confidence |   calibration_gap |
|:-------------|---------:|----:|-----------:|------------------:|------------------:|
| patch_wf_L10 |        0 |  18 |     0.8333 |            0.8912 |            0.0579 |
| patch_wf_L10 |        1 |  18 |     0.5556 |            0.8724 |            0.3168 |
| patch_wf_L10 |        2 |  18 |     0.6111 |            0.9825 |            0.3714 |
| patch_wf_L10 |        3 |  18 |     0.6111 |            0.9193 |            0.3082 |
| patch_wf_L10 |        4 |  18 |     0.7222 |            0.9722 |            0.25   |

### patch_L0_full

| condition     |   wf_bin |   n |   accuracy |   mean_confidence |   calibration_gap |
|:--------------|---------:|----:|-----------:|------------------:|------------------:|
| patch_L0_full |        0 |  18 |     0.8333 |            0.903  |            0.0697 |
| patch_L0_full |        1 |  18 |     0.5556 |            0.8663 |            0.3107 |
| patch_L0_full |        2 |  18 |     0.6111 |            0.9821 |            0.371  |
| patch_L0_full |        3 |  18 |     0.6111 |            0.9199 |            0.3087 |
| patch_L0_full |        4 |  18 |     0.7222 |            0.9699 |            0.2477 |

## v6-2 sanity slice (token logits after patched position)

Top-5 predicted tokens at positions prediction_site+1, +2, +3 (first test trace, first 3 cases). If output is degenerate (e.g., all mass on 'Based'), L0 full patching corrupts downstream computation.

|   case_index |   prediction_site |   offset | top5_tokens                                        |
|-------------:|------------------:|---------:|:---------------------------------------------------|
|            0 |               211 |        1 | [')', '.', ':', ' is', ' -']                       |
|            0 |               211 |        2 | ['\n\n', ' **', '**', '\n\n\n', '\n']              |
|            0 |               211 |        3 | ['**', 'An', '<eos>', '\\', ' **']                 |
|            1 |               516 |        1 | [')', '\n\n\n', ' ', '.', ' -']                    |
|            1 |               516 |        2 | ['\n\n', '<eos>', '\n\n\n', '\n', '<end_of_turn>'] |
|            1 |               516 |        3 | ['<eos>', ' ', '<end_of_turn>', '\n\n', 'The']     |
|            2 |               709 |        1 | [' ', ')', '\n\n\n', '  ', '.']                    |
|            2 |               709 |        2 | ['<eos>', '\n', '\n\n', ' ', '\n\n\n']             |
|            2 |               709 |        3 | ['<eos>', ' ', '<end_of_turn>', '*', 'Gu']         |

## Interpretation

- **patch_wf_L0**: accuracy Δ=+0.0000, calibration_gap Δ=+0.0025
- **patch_wf_L10**: accuracy Δ=+0.0000, calibration_gap Δ=+0.0006
- **patch_L0_full**: accuracy Δ=+0.0000, calibration_gap Δ=+0.0013

If neither v6-1 nor v6-2 moves accuracy or calibration gap beyond noise (|Δ| < 0.02), the patching result is consistent with the steering and ablation nulls from v2–v4: the within_task_fraction direction is a readout of context length, not a causal lever. The model's calibration gap is driven by mechanisms not localizable to a single layer's residual stream direction.

If v6-2 (full L0 patch) also produces near-zero effects, the result strengthens the conclusion that even the *full* L0 state — which encodes absolute position through RoPE at the attention level — cannot transplant the 'freshness' of a clean context into a fatigued one.
