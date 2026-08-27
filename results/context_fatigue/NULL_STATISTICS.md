# Interval estimates for the context-fatigue nulls

## 1. Accuracy vs context fill (the null, with power)

| stream | n | accuracy | corr(correct, fill) 95% CI | upper-half minus lower-half | decline excluded above |
|---|---:|---:|---|---|---:|
| coherent | 1126 | 0.818 | +0.008 [-0.051, +0.066] | +0.008 [-0.038, +0.053] | 0.038 |
| random | 817 | 0.597 | -0.051 [-0.119, +0.018] | -0.023 [-0.090, +0.045] | 0.090 |

The final column is the equivalence bound: accuracy declines larger than this (in proportion points, upper vs lower half of the context) are excluded at 95%.

### Scoped to the first 80% of context (the paper's flat claim)

| stream | n | accuracy | corr(correct, fill) 95% CI | upper-half minus lower-half | decline excluded above |
|---|---:|---:|---|---|---:|
| coherent | 988 | 0.822 | +0.032 [-0.030, +0.095] | +0.020 [-0.029, +0.068] | 0.029 |
| random | 728 | 0.614 | +0.002 [-0.071, +0.074] | +0.017 [-0.055, +0.090] | 0.055 |

The flat claim is scoped to fill < 0.8; the ≥0.8 region is quantified in §2 below.

## 2. The top-fill-bin dip — **WITHDRAWN 2026-08-19**

**Do not quote the random-stream row below.** `E2B_DIP_RESCUE.md` reproduced the committed run item-for-item (per-item agreement 1.000 over 344 shared items) and then located the whole effect inside a single trough: 0.80–0.85 = 0.625 (n=40), **0.85–0.88 = 0.419 (n=31)**, 0.88–0.93 = 0.703 (n=37). The artifact this table is built from has a maximum fill of 0.8784, so its entire top bin *is* that trough. Extending the same sessions to 0.93 gives −0.097 (n.s.); 14 fresh sessions give **+0.090** (opposite sign); all 26 pooled give **+0.005 [−0.105, +0.092]**. The row is retained only so the withdrawn number stays traceable.

| stream | n top bin | acc top bin | acc rest | difference 95% CI | significant |
|---|---:|---:|---:|---|---|
| coherent | 138 | 0.790 | 0.822 | -0.032 [-0.104, +0.038] | no |
| random | 89 | 0.461 | 0.614 | -0.153 [-0.263, -0.042] | ~~yes~~ **artifact** |

## 3. Per-case inversion with intervals (instruct, layer 24)

| fill bin | n cases | correct | wrong | delta (wrong-correct) 95% CI | significant |
|---|---:|---:|---:|---|---|
| 0-25% | 36 | 0.349 | 0.377 | +0.028 [-0.076, +0.129] | no |
| 25-50% | 32 | 0.193 | 0.197 | +0.005 [-0.008, +0.017] | no |
| 50-75% | 31 | 0.168 | 0.179 | +0.010 [-0.001, +0.021] | no |
| 75-100% | 16 | 0.153 | 0.182 | +0.029 [+0.013, +0.046] | yes |
| **pooled** | 115 | | | **+0.045 [-0.003, +0.097]** | no |

1 of 4 individual bins reach significance; the pooled estimate is the claim to lead with.

### Layer generality

| layer | pooled delta 95% CI | significant |
|---|---|---|
| 8 | +0.025 [-0.021, +0.073] | no |
| 16 | +0.003 [-0.020, +0.026] | no |
| 24 | +0.045 [-0.003, +0.097] | no |
| 31 | +0.034 [-0.012, +0.083] | no |

## 4. The confidently-wrong gap

Pooled DDXPlus MCQ streams, n=154 cases (llama8b=49, llama8b_reversed=43, qwen7b_verbose=32, qwen7b_reversed=30).

| fill bin | n | accuracy | confidence | gap | confidence on wrong answers 95% CI |
|---|---:|---:|---:|---:|---|
| 0-25% | 41 | 0.317 | 0.849 | +0.532 | 0.858 [0.837, 0.878] (n=28) |
| 25-50% | 40 | 0.225 | 0.908 | +0.683 | 0.906 [0.892, 0.919] (n=31) |
| 50-75% | 42 | 0.452 | 0.939 | +0.486 | 0.937 [0.924, 0.950] (n=23) |
| 75-100% | 31 | 0.387 | 0.947 | +0.560 | 0.945 [0.935, 0.955] (n=19) |

corr(confidence, fill) = +0.722 [+0.637, +0.790], n=154

corr(confidence, fill) on **wrong answers only** = +0.687 [+0.568, +0.778], n=101
