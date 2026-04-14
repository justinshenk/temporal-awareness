# v5: dense early-layer probe sweep (Qwen-2.5-7B-Instruct, DDXPlus)

Trained ridge probes at layers [0, 2, 4, 6, 8, 10, 12, 14, 18, 22, 27] using the same 20-trace extraction and the same train/test split (16/4, seed 42) as v1. Goal: localize the earliest layer at which the within_task_fraction and tokens_until_boundary signals become linearly decodable.

## Peak layer per target

| target                |   peak_layer |   peak_metric |   peak_delta |
|:----------------------|-------------:|--------------:|-------------:|
| task_index            |            2 |      0.993022 |   0.00186056 |
| within_task_fraction  |            2 |      0.95977  |   0.957317   |
| tokens_until_boundary |            4 |      0.916546 |   0.917458   |

## Per-layer probe metrics

| target                |   layer |   metric |   baseline_raw_pos |        delta |
|:----------------------|--------:|---------:|-------------------:|-------------:|
| task_index            |       0 | 0.963829 |        0.991162    | -0.0273329   |
| task_index            |       2 | 0.993022 |        0.991162    |  0.00186056  |
| task_index            |       4 | 0.991763 |        0.991162    |  0.000601241 |
| task_index            |       6 | 0.987845 |        0.991162    | -0.00331713  |
| task_index            |       8 | 0.981331 |        0.991162    | -0.00983099  |
| task_index            |      10 | 0.978309 |        0.991162    | -0.0128532   |
| task_index            |      12 | 0.975057 |        0.991162    | -0.0161048   |
| task_index            |      14 | 0.978255 |        0.991162    | -0.0129072   |
| task_index            |      18 | 0.96906  |        0.991162    | -0.0221014   |
| task_index            |      22 | 0.950921 |        0.991162    | -0.0402407   |
| task_index            |      27 | 0.916621 |        0.991162    | -0.074541    |
| within_task_fraction  |       0 | 0.926349 |        0.00245391  |  0.923895    |
| within_task_fraction  |       2 | 0.95977  |        0.00245391  |  0.957317    |
| within_task_fraction  |       4 | 0.959071 |        0.00245391  |  0.956617    |
| within_task_fraction  |       6 | 0.955797 |        0.00245391  |  0.953344    |
| within_task_fraction  |       8 | 0.952676 |        0.00245391  |  0.950222    |
| within_task_fraction  |      10 | 0.950273 |        0.00245391  |  0.947819    |
| within_task_fraction  |      12 | 0.946846 |        0.00245391  |  0.944392    |
| within_task_fraction  |      14 | 0.944516 |        0.00245391  |  0.942062    |
| within_task_fraction  |      18 | 0.943935 |        0.00245391  |  0.941481    |
| within_task_fraction  |      22 | 0.943483 |        0.00245391  |  0.941029    |
| within_task_fraction  |      27 | 0.915846 |        0.00245391  |  0.913392    |
| tokens_until_boundary |       0 | 0.889898 |       -0.000912513 |  0.890811    |
| tokens_until_boundary |       2 | 0.915983 |       -0.000912513 |  0.916895    |
| tokens_until_boundary |       4 | 0.916546 |       -0.000912513 |  0.917458    |
| tokens_until_boundary |       6 | 0.909859 |       -0.000912513 |  0.910772    |
| tokens_until_boundary |       8 | 0.908633 |       -0.000912513 |  0.909546    |
| tokens_until_boundary |      10 | 0.90373  |       -0.000912513 |  0.904642    |
| tokens_until_boundary |      12 | 0.900528 |       -0.000912513 |  0.901441    |
| tokens_until_boundary |      14 | 0.897967 |       -0.000912513 |  0.89888     |
| tokens_until_boundary |      18 | 0.899265 |       -0.000912513 |  0.900177    |
| tokens_until_boundary |      22 | 0.902852 |       -0.000912513 |  0.903764    |
| tokens_until_boundary |      27 | 0.891778 |       -0.000912513 |  0.89269     |

## Per-layer A1 orthogonality (cosine with raw-position direction)

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

## Interpretation

Read the per-layer R² curve for `within_task_fraction` and `tokens_until_boundary`. Three regimes to distinguish:

- **Early saturation (R² ≥ 0.8 at L2 or L4):** The signal is present almost immediately after token embeddings. The mechanism is upstream of almost every interventional knob — essentially baked into the embedding + first few attention operations. Any steering or ablation at L10+ is intervening on a downstream copy.
- **Gradual climb (R² climbs from ~0 at L2 to ~0.9 at L10):** The signal is *constructed* across the early stack. The layer at which R² first exceeds ~0.5 is the earliest layer where an intervention could plausibly move the downstream calibration gap.
- **Late emergence (R² still near baseline at L8):** The signal is built in a narrow window around L10. Our v2/v3/v4 interventions at L10+ were in the right place but the wrong modality (residual perturbation can't match the QK attention pattern that constructs the signal).

The A1 orthogonality table tells a complementary story: at which layer does the probe direction first become orthogonal to raw position? Early orthogonality means the signal is *not* positional encoding at that layer.
