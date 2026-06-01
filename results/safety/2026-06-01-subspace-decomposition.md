# Decomposition — is finetuning's safety axis in the non-shared part?

Finetune (LoRA) shift split into the ICL-aligned `par` (shared task direction) and the orthogonal `perp` (finetune-only). Magnitude-normalized cosine with the refusal direction r isolates *direction* (negative = toward compliance), robust to large shift magnitudes.

## On medical (DDXPlus) inputs — where the shared component is large

| Layer | cos(shared `par`, r) | cos(finetune-only `perp`, r) | shift frac in shared | cos(shared dir, r) |
|------:|---------------------:|-----------------------------:|---------------------:|-------------------:|
| 0 | -0.123 | +0.085 | 0.139 | +0.123 |
| 7 | +0.057 | -0.112 | 0.022 | +0.057 |
| 14 | -0.113 | -0.024 | 0.275 | -0.113 |
| 21 | -0.037 | -0.075 | 0.657 | -0.037 |
| 28 | -0.037 | -0.104 | 0.739 | -0.037 |
| 35 | -0.083 | -0.028 | 0.807 | -0.083 |
| 41 | +0.049 | +0.322 | 0.744 | +0.049 |

Late-layer (≥21): shared `par`·r̂ = -0.027, finetune-only `perp`·r̂ = +0.029, shared fraction ≈ 0.74, cos(shared dir, r) = -0.027.

## On harmful inputs — where safety erosion manifests

| Layer | cos(shared `par`, r) | cos(finetune-only `perp`, r) | shift frac in shared | cos(shared dir, r) |
|------:|---------------------:|-----------------------------:|---------------------:|-------------------:|
| 0 | -0.320 | -0.085 | 0.473 | +0.320 |
| 7 | +0.069 | -0.036 | 0.044 | -0.069 |
| 14 | -0.247 | -0.091 | 0.043 | -0.247 |
| 21 | -0.175 | -0.294 | 0.172 | -0.175 |
| 28 | -0.016 | -0.512 | 0.027 | +0.016 |
| 35 | +0.008 | -0.593 | 0.058 | -0.008 |
| 41 | -0.266 | -0.201 | 0.021 | -0.266 |

Late-layer (≥21): shared `par`·r̂ = -0.112, finetune-only `perp`·r̂ = -0.400, shared fraction ≈ 0.07.

## Reading

- **Shared part is safety-neutral.** On medical inputs the shared task direction is the bulk of the finetune shift (≈74% of it late) yet is ~orthogonal to the refusal axis (cos(shared dir, r) ≈ -0.03). The part ICL reproduces does not point along refusal.
- **Harm is finetune-only.** On harmful inputs the finetune shift is almost entirely the orthogonal `perp` (shared fraction ≈ 0.07), and that `perp` carries the toward-compliance drift (cos(perp, r) = -0.40).
- **Verdict:** supports "shared subspace = beneficial task adaptation" — the ICL-shared component is safety-neutral and the compliance drift lives in the finetune-only direction. Refinement: that harmful direction is input-gated (near-zero refusal-axis content on benign medical inputs), so finetuning installs a weight change whose harm is triggered by harmful input rather than a static always-on compliance vector.
