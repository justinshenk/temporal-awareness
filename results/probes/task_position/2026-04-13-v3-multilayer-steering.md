# Multi-layer causal steering experiment (v3-1)

Extends the v2 single-layer steering experiment by hooking the within_task_fraction probe at multiple layers simultaneously. The v2 single-layer null could have been an artifact of downstream-layer compensation; v3-1 tests whether broader-stack steering punches through.

Each condition steers all listed layers using their own trained within_task_fraction probe direction, pushing every token's readout to the target value at every steered layer simultaneously.

## Per-condition summary

| condition         |   n |   accuracy |   mean_confidence |   calibration_gap |
|:------------------|----:|-----------:|------------------:|------------------:|
| no_steer          |  90 |   0.666667 |          0.933096 |          0.266429 |
| early_L10         |  90 |   0.666667 |          0.933685 |          0.267018 |
| late_L10          |  90 |   0.677778 |          0.93496  |          0.257182 |
| early_L10_L20     |  90 |   0.655556 |          0.933457 |          0.277902 |
| late_L10_L20      |  90 |   0.666667 |          0.934395 |          0.267728 |
| early_L10_L20_L30 |  90 |   0.666667 |          0.931725 |          0.265058 |
| late_L10_L20_L30  |  90 |   0.666667 |          0.934837 |          0.268171 |
| early_all_late    |  90 |   0.644444 |          0.932638 |          0.288194 |
| late_all_late     |  90 |   0.666667 |          0.934911 |          0.268245 |

## Confidence shift vs no_steer (mean per case)

| condition         |   mean_confidence_shift |
|:------------------|------------------------:|
| early_L10         |             0.000589485 |
| late_L10          |             0.00186444  |
| early_L10_L20     |             0.000361818 |
| late_L10_L20      |             0.00129914  |
| early_L10_L20_L30 |            -0.00137097  |
| late_L10_L20_L30  |             0.00174184  |
| early_all_late    |            -0.000457461 |
| late_all_late     |             0.00181564  |

## Case-level flips vs no_steer

| condition         |   wrong_to_right |   right_to_wrong |   net_flips |
|:------------------|-----------------:|-----------------:|------------:|
| early_L10         |                0 |                0 |           0 |
| late_L10          |                1 |                0 |           1 |
| early_L10_L20     |                0 |                1 |          -1 |
| late_L10_L20      |                0 |                0 |           0 |
| early_L10_L20_L30 |                0 |                0 |           0 |
| late_L10_L20_L30  |                0 |                0 |           0 |
| early_all_late    |                0 |                2 |          -2 |
| late_all_late     |                0 |                0 |           0 |

## Interpretation

Compare each early/late condition to no_steer. If the steering effect scales with the number of steered layers (e.g., L10 alone is null, L10+L20 has small effect, all-layer has large effect), then the v2 null was a single-layer compensation artifact and the direction IS causal — just hard to perturb in isolation.

If even the full-stack steering produces near-zero effects, the v2 conclusion stands: the within_task_fraction direction at L10 is a *readout* of something that drives the calibration gap, not the *cause* of it. The probe captures a real internal signal but not a lever.
