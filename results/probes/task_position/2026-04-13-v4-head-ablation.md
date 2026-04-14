# Head-level ablation experiment (v4)

Tests whether the L20 heads that drive `att_current_case` are causally upstream of the v1 calibration gap. Phase 1 ranks the 16 L20 heads by their per-head `att_current_case` correlation with the L10 within_task_fraction probe readout. Phase 2 ablates top/bottom/random heads by zeroing the corresponding input columns of `o_proj.weight`. Phase 3 measures per-case accuracy and confidence under each ablation. Phase 4 compares the global calibration gap and the A4-style binned gap pattern across conditions.

A flattening of the gap-vs-bin slope under ablation is strong evidence that the ablated heads are causally upstream of the A4 pattern. A null result (slope unchanged) means the ablation removed the probe-predictive feature but not the calibration gap — meaning the gap is driven by something else entirely.

## Phase 1: per-head L20 ranking

|   rank |   head |   mean_correlation |
|-------:|-------:|-------------------:|
|      0 |     13 |           0.516288 |
|      1 |      3 |           0.487076 |
|      2 |      6 |           0.420169 |
|      3 |     14 |           0.408966 |
|      4 |      8 |           0.333972 |
|      5 |     15 |           0.330181 |
|      6 |      7 |           0.304548 |
|      7 |      2 |           0.295766 |
|      8 |     11 |           0.288366 |
|      9 |     12 |           0.269975 |
|     10 |      0 |           0.219183 |
|     11 |      9 |           0.214085 |
|     12 |      5 |           0.205685 |
|     13 |      1 |           0.199725 |
|     14 |     10 |           0.185456 |
|     15 |      4 |           0.12563  |

Ablation sets: top1=[13], top3=[13, 3, 6], top5=[13, 3, 6, 14, 8], bottom3=[1, 10, 4], random3=[0, 3, 11]

## Global summary per condition

| condition      |   n |   accuracy |   mean_confidence |   calibration_gap |
|:---------------|----:|-----------:|------------------:|------------------:|
| no_ablation    |  90 |   0.666667 |          0.926968 |          0.260302 |
| ablate_top1    |  90 |   0.677778 |          0.923897 |          0.246119 |
| ablate_top3    |  90 |   0.677778 |          0.935621 |          0.257843 |
| ablate_top5    |  90 |   0.677778 |          0.929986 |          0.252209 |
| ablate_bottom3 |  90 |   0.666667 |          0.932765 |          0.266098 |
| ablate_random3 |  90 |   0.655556 |          0.934482 |          0.278926 |

## Gap-vs-bin slope per condition

| condition      |   gap_slope |
|:---------------|------------:|
| no_ablation    |   0.0352442 |
| ablate_top1    |   0.0451855 |
| ablate_top3    |   0.0669987 |
| ablate_top5    |   0.0502688 |
| ablate_bottom3 |   0.0160447 |
| ablate_random3 |   0.0183962 |

Baseline slope is the expected A4 monotone pattern on this test set. Ablation conditions that drop the slope toward 0 have attacked the mechanism producing the pattern.

## A4-style binned calibration by condition

| condition      |   bin |   n |   confidence |   accuracy |         gap |
|:---------------|------:|----:|-------------:|-----------:|------------:|
| no_ablation    |     0 |  18 |     0.898581 |   0.833333 |  0.0652477  |
| no_ablation    |     1 |  18 |     0.86997  |   0.555556 |  0.314415   |
| no_ablation    |     2 |  18 |     0.981878 |   0.611111 |  0.370767   |
| no_ablation    |     4 |  18 |     0.968496 |   0.722222 |  0.246274   |
| no_ablation    |     3 |  18 |     0.915916 |   0.611111 |  0.304805   |
| ablate_top1    |     0 |  18 |     0.887057 |   0.888889 | -0.00183163 |
| ablate_top1    |     1 |  18 |     0.881224 |   0.555556 |  0.325669   |
| ablate_top1    |     2 |  18 |     0.979023 |   0.611111 |  0.367912   |
| ablate_top1    |     4 |  18 |     0.957235 |   0.722222 |  0.235013   |
| ablate_top1    |     3 |  18 |     0.914946 |   0.611111 |  0.303834   |
| ablate_top3    |     0 |  18 |     0.890132 |   0.944444 | -0.0543121  |
| ablate_top3    |     1 |  18 |     0.894959 |   0.555556 |  0.339404   |
| ablate_top3    |     2 |  18 |     0.976495 |   0.611111 |  0.365384   |
| ablate_top3    |     4 |  18 |     0.984249 |   0.722222 |  0.262027   |
| ablate_top3    |     3 |  18 |     0.932269 |   0.555556 |  0.376713   |
| ablate_top5    |     0 |  18 |     0.900046 |   0.944444 | -0.0443985  |
| ablate_top5    |     1 |  18 |     0.88575  |   0.5      |  0.38575    |
| ablate_top5    |     2 |  18 |     0.986025 |   0.611111 |  0.374914   |
| ablate_top5    |     4 |  18 |     0.977085 |   0.722222 |  0.254863   |
| ablate_top5    |     3 |  18 |     0.901026 |   0.611111 |  0.289915   |
| ablate_bottom3 |     0 |  18 |     0.904182 |   0.777778 |  0.126404   |
| ablate_bottom3 |     1 |  18 |     0.918321 |   0.555556 |  0.362765   |
| ablate_bottom3 |     2 |  18 |     0.965663 |   0.666667 |  0.298997   |
| ablate_bottom3 |     4 |  18 |     0.955917 |   0.722222 |  0.233695   |
| ablate_bottom3 |     3 |  18 |     0.919742 |   0.611111 |  0.308631   |
| ablate_random3 |     0 |  18 |     0.916328 |   0.777778 |  0.13855    |
| ablate_random3 |     1 |  18 |     0.906063 |   0.555556 |  0.350508   |
| ablate_random3 |     2 |  18 |     0.958561 |   0.611111 |  0.34745    |
| ablate_random3 |     4 |  18 |     0.975668 |   0.722222 |  0.253445   |
| ablate_random3 |     3 |  18 |     0.91579  |   0.611111 |  0.304678   |

## Flip stats vs no_ablation

| condition      |   wrong_to_right |   right_to_wrong |   net_flips |
|:---------------|-----------------:|-----------------:|------------:|
| ablate_top1    |                2 |                1 |           1 |
| ablate_top3    |                2 |                1 |           1 |
| ablate_top5    |                3 |                2 |           1 |
| ablate_bottom3 |                2 |                2 |           0 |
| ablate_random3 |                0 |                1 |          -1 |

## Interpretation

Read this in order:
1. Does the global calibration gap change under ablation? If no change at top5 either, the heads are not carrying the gap signal.
2. Does the gap-vs-bin slope flatten under ablation? If yes (and not for bottom3/random3), the ablation targets are causally upstream of the A4 pattern. If no, the A4 pattern survives the ablation and is driven by something the ablation didn't touch.
3. Do the targeted ablations (top1/3/5) differ meaningfully from the random3 control? If they behave identically, the effect is non-specific.
