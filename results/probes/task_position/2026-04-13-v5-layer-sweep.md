# v5: dense early-layer probe sweep (Gemma-9B-IT, DDXPlus)

Trained ridge probes at layers [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 41] using the same 20-trace extraction and the same train/test split (16/4, seed 42) as v1. Goal: localize the earliest layer at which the within_task_fraction and tokens_until_boundary signals become linearly decodable.

## Peak layer per target

| target                |   peak_layer |   peak_metric |   peak_delta |
|:----------------------|-------------:|--------------:|-------------:|
| task_index            |            6 |      0.990156 |  -0.00058858 |
| within_task_fraction  |            6 |      0.957224 |   0.954899   |
| tokens_until_boundary |            6 |      0.911337 |   0.911778   |

## Per-layer probe metrics

| target                |   layer |   metric |   baseline_raw_pos |        delta |
|:----------------------|--------:|---------:|-------------------:|-------------:|
| task_index            |       0 | 0.895409 |        0.990745    | -0.0953361   |
| task_index            |       2 | 0.989851 |        0.990745    | -0.000893754 |
| task_index            |       4 | 0.989314 |        0.990745    | -0.00143042  |
| task_index            |       6 | 0.990156 |        0.990745    | -0.00058858  |
| task_index            |       8 | 0.988211 |        0.990745    | -0.0025341   |
| task_index            |      10 | 0.987938 |        0.990745    | -0.00280623  |
| task_index            |      15 | 0.987826 |        0.990745    | -0.00291903  |
| task_index            |      20 | 0.98407  |        0.990745    | -0.00667496  |
| task_index            |      25 | 0.977553 |        0.990745    | -0.0131915   |
| task_index            |      30 | 0.964884 |        0.990745    | -0.025861    |
| task_index            |      35 | 0.94703  |        0.990745    | -0.0437142   |
| task_index            |      41 | 0.960075 |        0.990745    | -0.03067     |
| within_task_fraction  |       0 | 0.898945 |        0.00232513  |  0.89662     |
| within_task_fraction  |       2 | 0.940582 |        0.00232513  |  0.938257    |
| within_task_fraction  |       4 | 0.95502  |        0.00232513  |  0.952694    |
| within_task_fraction  |       6 | 0.957224 |        0.00232513  |  0.954899    |
| within_task_fraction  |       8 | 0.956249 |        0.00232513  |  0.953924    |
| within_task_fraction  |      10 | 0.95437  |        0.00232513  |  0.952045    |
| within_task_fraction  |      15 | 0.94924  |        0.00232513  |  0.946915    |
| within_task_fraction  |      20 | 0.942801 |        0.00232513  |  0.940476    |
| within_task_fraction  |      25 | 0.935578 |        0.00232513  |  0.933253    |
| within_task_fraction  |      30 | 0.927215 |        0.00232513  |  0.92489     |
| within_task_fraction  |      35 | 0.92301  |        0.00232513  |  0.920685    |
| within_task_fraction  |      41 | 0.920436 |        0.00232513  |  0.918111    |
| tokens_until_boundary |       0 | 0.8713   |       -0.000441468 |  0.871742    |
| tokens_until_boundary |       2 | 0.897137 |       -0.000441468 |  0.897578    |
| tokens_until_boundary |       4 | 0.908598 |       -0.000441468 |  0.909039    |
| tokens_until_boundary |       6 | 0.911337 |       -0.000441468 |  0.911778    |
| tokens_until_boundary |       8 | 0.909191 |       -0.000441468 |  0.909633    |
| tokens_until_boundary |      10 | 0.906793 |       -0.000441468 |  0.907235    |
| tokens_until_boundary |      15 | 0.900777 |       -0.000441468 |  0.901218    |
| tokens_until_boundary |      20 | 0.895521 |       -0.000441468 |  0.895962    |
| tokens_until_boundary |      25 | 0.888559 |       -0.000441468 |  0.889       |
| tokens_until_boundary |      30 | 0.879352 |       -0.000441468 |  0.879794    |
| tokens_until_boundary |      35 | 0.877391 |       -0.000441468 |  0.877832    |
| tokens_until_boundary |      41 | 0.884362 |       -0.000441468 |  0.884803    |

## Per-layer A1 orthogonality (cosine with raw-position direction)

|   layer | target                |   cosine_with_raw_pos |   raw_dir_norm |   target_dir_norm |
|--------:|:----------------------|----------------------:|---------------:|------------------:|
|       0 | task_index            |           0.97207     |      42623.1   |       122.573     |
|       0 | within_task_fraction  |          -0.0212172   |      42623.1   |         5.06264   |
|       0 | tokens_until_boundary |           0.000393599 |      42623.1   |        17.5893    |
|       2 | task_index            |           0.877317    |       6882.06  |        22.0092    |
|       2 | within_task_fraction  |           0.0229525   |       6882.06  |         1.39632   |
|       2 | tokens_until_boundary |           0.0206697   |       6882.06  |         5.27354   |
|       4 | task_index            |           0.860282    |       3893.79  |        12.3724    |
|       4 | within_task_fraction  |          -0.00235123  |       3893.79  |         0.662009  |
|       4 | tokens_until_boundary |           0.024414    |       3893.79  |         2.85801   |
|       6 | task_index            |           0.842772    |       2328.3   |         7.54911   |
|       6 | within_task_fraction  |           0.0175548   |       2328.3   |         0.388583  |
|       6 | tokens_until_boundary |          -0.00385887  |       2328.3   |         1.74711   |
|       8 | task_index            |           0.871823    |       1600.91  |         5.24619   |
|       8 | within_task_fraction  |          -0.000586451 |       1600.91  |         0.251752  |
|       8 | tokens_until_boundary |          -0.00372412  |       1600.91  |         1.11872   |
|      10 | task_index            |           0.860367    |       1104.19  |         3.60164   |
|      10 | within_task_fraction  |          -0.0174973   |       1104.19  |         0.183136  |
|      10 | tokens_until_boundary |           0.0107829   |       1104.19  |         0.790062  |
|      15 | task_index            |           0.88368     |        506.882 |         1.70786   |
|      15 | within_task_fraction  |           0.0355148   |        506.882 |         0.0878771 |
|      15 | tokens_until_boundary |          -0.0133618   |        506.882 |         0.367757  |
|      20 | task_index            |           0.921888    |        388.352 |         1.18514   |
|      20 | within_task_fraction  |           0.0227673   |        388.352 |         0.0517857 |
|      20 | tokens_until_boundary |           0.00975356  |        388.352 |         0.207286  |
|      25 | task_index            |           0.943202    |        411.126 |         1.20849   |
|      25 | within_task_fraction  |           0.0413229   |        411.126 |         0.0468626 |
|      25 | tokens_until_boundary |           0.00839956  |        411.126 |         0.181007  |
|      30 | task_index            |           0.963174    |        404.12  |         1.19058   |
|      30 | within_task_fraction  |          -0.00118474  |        404.12  |         0.0380749 |
|      30 | tokens_until_boundary |           0.0506497   |        404.12  |         0.14985   |
|      35 | task_index            |           0.975634    |        347.245 |         1.01612   |
|      35 | within_task_fraction  |           0.0106042   |        347.245 |         0.0275497 |
|      35 | tokens_until_boundary |           0.0456442   |        347.245 |         0.106142  |
|      41 | task_index            |           0.970598    |        227.101 |         0.663689  |
|      41 | within_task_fraction  |           0.0458675   |        227.101 |         0.0240656 |
|      41 | tokens_until_boundary |           0.00800319  |        227.101 |         0.0771829 |

## Interpretation

Read the per-layer R² curve for `within_task_fraction` and `tokens_until_boundary`. Three regimes to distinguish:

- **Early saturation (R² ≥ 0.8 at L2 or L4):** The signal is present almost immediately after token embeddings. The mechanism is upstream of almost every interventional knob — essentially baked into the embedding + first few attention operations. Any steering or ablation at L10+ is intervening on a downstream copy.
- **Gradual climb (R² climbs from ~0 at L2 to ~0.9 at L10):** The signal is *constructed* across the early stack. The layer at which R² first exceeds ~0.5 is the earliest layer where an intervention could plausibly move the downstream calibration gap.
- **Late emergence (R² still near baseline at L8):** The signal is built in a narrow window around L10. Our v2/v3/v4 interventions at L10+ were in the right place but the wrong modality (residual perturbation can't match the QK attention pattern that constructs the signal).

The A1 orthogonality table tells a complementary story: at which layer does the probe direction first become orthogonal to raw position? Early orthogonality means the signal is *not* positional encoding at that layer.
