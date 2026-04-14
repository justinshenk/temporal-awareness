# Task-Position Probes v1 Results (Qwen-2.5-7B-Instruct, DDXPlus)

Multi-case DDXPlus traces accumulated to ~90% of Qwen-2.5-7B-Instruct's 8k context. Per-token residual streams captured at layers [0, 2, 4, 6, 8, 10, 12, 14, 18, 22, 27]. Twenty traces split 80/20 by trace id (train=[0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 16, 17, 18, 19], test=[7, 9, 14, 15], seed=42).

## Headline numbers (best layer per target, from training run)

| target                |   layer |   metric |   baseline_raw_pos |      delta |
|:----------------------|--------:|---------:|-------------------:|-----------:|
| task_index            |       2 | 0.993022 |        0.991162    | 0.00186056 |
| tokens_until_boundary |       4 | 0.916546 |       -0.000912513 | 0.917458   |
| within_task_fraction  |       2 | 0.95977  |        0.00245391  | 0.957317   |

Metric is Spearman ρ for `task_index` (ordinal target) and R² for the other two (the latter trained in `log1p` space). The baseline is a single-feature Ridge on raw token position; if a residual-stream probe doesn't beat it, the probe is just reading positional encoding.

## A1: Orthogonality between task-position and raw-position directions

For each layer we fit a ridge probe to predict raw token position from the residual stream, then compute the cosine similarity between that direction and each target probe's direction. Strong alignment = 'this target is just raw position'. Near-zero = 'this target encodes a new signal'.

|   layer | target                |   cosine_with_raw_pos |   raw_dir_norm |   target_dir_norm |
|--------:|:----------------------|----------------------:|---------------:|------------------:|
|       0 | task_index            |            0.980178   |      79943.4   |       228.07      |
|       0 | within_task_fraction  |           -0.0671427  |      79943.4   |         7.39238   |
|       0 | tokens_until_boundary |            0.0440632  |      79943.4   |        26.4147    |
|       2 | task_index            |            0.863515   |      14396.5   |        45.1346    |
|       2 | within_task_fraction  |            0.0129384  |      14396.5   |         2.41182   |
|       2 | tokens_until_boundary |           -0.00306938 |      14396.5   |        10.3667    |
|       4 | task_index            |            0.863036   |       6700.17  |        20.2882    |
|       4 | within_task_fraction  |            0.035801   |       6700.17  |         1.039     |
|       4 | tokens_until_boundary |            0.00536998 |       6700.17  |         4.60725   |
|       6 | task_index            |            0.914563   |       4635.48  |        13.943     |
|       6 | within_task_fraction  |           -0.00657417 |       4635.48  |         0.568948  |
|       6 | tokens_until_boundary |            0.0115543  |       4635.48  |         2.5516    |
|       8 | task_index            |            0.942638   |       3388.98  |        10.1393    |
|       8 | within_task_fraction  |           -0.0844014  |       3388.98  |         0.35839   |
|       8 | tokens_until_boundary |            0.0595324  |       3388.98  |         1.52291   |
|      10 | task_index            |            0.947284   |       3297.3   |         9.68803   |
|      10 | within_task_fraction  |           -0.0624162  |       3297.3   |         0.361998  |
|      10 | tokens_until_boundary |            0.0641428  |       3297.3   |         1.55197   |
|      12 | task_index            |            0.952876   |       2733.95  |         8.07072   |
|      12 | within_task_fraction  |           -0.0484895  |       2733.95  |         0.286138  |
|      12 | tokens_until_boundary |            0.0575851  |       2733.95  |         1.20863   |
|      14 | task_index            |            0.952513   |       2348.6   |         6.8552    |
|      14 | within_task_fraction  |           -0.0476151  |       2348.6   |         0.243865  |
|      14 | tokens_until_boundary |            0.065197   |       2348.6   |         1.02943   |
|      18 | task_index            |            0.965422   |       1716.78  |         4.94963   |
|      18 | within_task_fraction  |           -0.0341792  |       1716.78  |         0.155873  |
|      18 | tokens_until_boundary |            0.0631635  |       1716.78  |         0.656351  |
|      22 | task_index            |            0.976423   |       1159.69  |         3.33037   |
|      22 | within_task_fraction  |            0.00597564 |       1159.69  |         0.0860924 |
|      22 | tokens_until_boundary |            0.032498   |       1159.69  |         0.348568  |
|      27 | task_index            |            0.98675    |        566.908 |         1.61967   |
|      27 | within_task_fraction  |           -0.0171011  |        566.908 |         0.0462426 |
|      27 | tokens_until_boundary |            0.0484166  |        566.908 |         0.143012  |

**Mean |cosine| with raw position** — `task_index` 0.940, `within_task_fraction` 0.038, `tokens_until_boundary` 0.041.

## A2: Upcoming failure prediction

Logistic regression trained on TRAIN traces, scored on TEST. Baseline: residual stream only. Treatment: residual stream + `[task_index, within_task_fraction, log1p(tokens_until_boundary)]` at the case's last prompt token. Target to beat: 0.67–0.68 (Qwen-7B baseline from context-fatigue writeup Section 4.1).

|   layer |   baseline_auc |   with_task_position_auc |        delta |   n_train |   n_test |
|--------:|---------------:|-------------------------:|-------------:|----------:|---------:|
|       0 |       0.504582 |                 0.477628 | -0.0269542   |       341 |       88 |
|       2 |       0.504582 |                 0.486792 | -0.0177898   |       341 |       88 |
|       4 |       0.474933 |                 0.459838 | -0.0150943   |       341 |       88 |
|       6 |       0.531536 |                 0.525067 | -0.006469    |       341 |       88 |
|       8 |       0.607008 |                 0.604313 | -0.00269542  |       341 |       88 |
|      10 |       0.552022 |                 0.547709 | -0.00431267  |       341 |       88 |
|      12 |       0.504582 |                 0.489488 | -0.0150943   |       341 |       88 |
|      14 |       0.484097 |                 0.466846 | -0.0172507   |       341 |       88 |
|      18 |       0.569811 |                 0.569811 |  0           |       341 |       88 |
|      22 |       0.519137 |                 0.519677 |  0.000539084 |       341 |       88 |
|      27 |       0.461995 |                 0.462534 |  0.000539084 |       341 |       88 |

## A4: Calibration gap vs within_task_fraction readout (L10)

We train a within_task_fraction probe on TRAIN activations and apply it to the prediction-site activation of each TEST case. We then bin cases two ways: (a) by the probe readout (the model's *subjective* sense of task-lateness), and (b) by raw token position (the *objective* depth into the trace). For each bin we report mean confidence (`option_probs[pred]`), mean accuracy (0/1), and the calibration gap = confidence - accuracy. The hypothesis is that calibration degrades more sharply with subjective lateness than with raw position.

| binning                      |   bin |   confidence |   accuracy |   n |      gap |
|:-----------------------------|------:|-------------:|-----------:|----:|---------:|
| within_task_fraction_readout |     0 |     0.925564 |   0.611111 |  18 | 0.314453 |
| within_task_fraction_readout |     1 |     0.907858 |   0.647059 |  17 | 0.2608   |
| within_task_fraction_readout |     2 |     0.955512 |   0.722222 |  18 | 0.23329  |
| within_task_fraction_readout |     3 |     0.892923 |   0.529412 |  17 | 0.363511 |
| within_task_fraction_readout |     4 |     0.96441  |   0.5      |  18 | 0.46441  |
| raw_position                 |     0 |     0.937934 |   0.555556 |  18 | 0.382378 |
| raw_position                 |     1 |     0.91659  |   0.705882 |  17 | 0.210708 |
| raw_position                 |     2 |     0.894097 |   0.444444 |  18 | 0.449653 |
| raw_position                 |     3 |     0.931066 |   0.705882 |  17 | 0.225184 |
| raw_position                 |     4 |     0.969184 |   0.611111 |  18 | 0.358073 |

## Verdict against v1 success criteria

**Criterion 1** (signal exists): at least one task-position probe beats both baselines on test split for at least one target.

  - Status: **MET**

**Criterion 2** (interpretable orthogonality picture): A1 produces a clear alignment story.

  - Status: **MET**

**Criterion 3** (calibration figure exists): A4 produces a binned table either supporting or refuting the calibration-gap hypothesis.

  - Status: **MET**
