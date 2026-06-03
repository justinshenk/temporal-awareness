# Three steering methods compared on gemma-2-9b-it (mean vs singular-vector vs linear map)

Cross-family comparison of the steering families on DDXPlus → refusal, gemma-2-9b-it
(offline). Split: base task 0.381, refusal 0.96. n_eval=40, n_harmful=25, n_fit=100,
all-position, layers [0,7,14,21,28,35,41].

## Results

| method | α=0.5 task | α=0.5 refusal | α=1.0 task | α=1.0 refusal |
|---|--:|--:|--:|--:|
| (1) **mean vector** | 0.575 | 0.96 | 0.475 | 0.00 |
| (3) **singular-vector** k=4 | 0.600 | 0.96 | 0.450 | 0.00 |
| (3) **singular-vector** k=16 | 0.600 | 0.96 | 0.475 | 0.00 |
| (2) **linear map** λ=1 | **0.650** | 0.96 | 0.654 | 0.00 |
| (2) **linear map** λ=3 | 0.625 | 0.96 | 0.621 | 0.00 |

## Ordering (at α=0.5, all refusal-safe at 0.96)
**linear map (0.65) > singular-vector (0.60) > mean (0.575).** Both richer methods beat the
raw mean, and the order is mechanistically sensible:

- **(3) Singular-vector steering** (project the mean shift onto the top-k right singular
  directions of the shift set — a *fixed, denoised* low-rank mean) gives a **modest** gain
  over the raw mean (0.60 vs 0.575). It **saturates by k=4** (k=16 identical), so a few
  principal shift directions carry the task; the rest of the mean is noise. But it is still a
  *fixed* vector, so it can't adapt per input.
- **(2) Linear map** (input-conditional `α·W·a`, closed-form ridge) is **best** (0.65) — and
  also the most over-steer-robust: at α=1 it *holds* task 0.65 while mean and singular-vector
  collapse (0.45–0.48). The extra win over (3) is exactly the **input-conditioning** the fixed
  methods lack.

So the gap mean→singular-vector is the *denoising* benefit; the larger gap
singular-vector→linear-map is the *input-conditioning* benefit. This **replicates the Qwen-7B
result** (conditioning beats the mean) cross-family and places singular-vector steering
cleanly between the two.

## Caveats
gemma's Goldilocks band is narrow — all methods erode refusal at α=1 (0.00); the comparison
is at α=0.5. Single split (absolute task numbers are split-sensitive; the within-run ordering
is the signal), n_eval=40, n_harmful=25, single seed.

## Reproduce
```bash
HF_HUB_OFFLINE=1 uv run python -m scripts.safety.run_conditional_steer \
    --config configs/safety/route_safety_gemma.yaml --modes all \
    --lambdas 1,3 --sv-ks 4,16 --alphas 0.5,1.0 --n-fit 100 --n-eval 40 --n-harmful 25
```
`LinearConditionalSteerHook` (linear map) + singular-vector branch in
`run_conditional_steer.py`. JSON: `results/safety/gemma_sweep/conditional_steer.json`.
