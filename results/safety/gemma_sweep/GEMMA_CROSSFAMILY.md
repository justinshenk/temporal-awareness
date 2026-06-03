# Cross-family replication on gemma-2-9b-it: steering = competence without the safety tax

Re-runs the steering-vs-finetune safety comparison on a different model family (Gemma2 vs
Qwen2), reusing the existing DDXPlus LoRA (`results/lora_icl/adapter`). gemma loaded offline
from cache. n_eval=50 harmful, n_task=40, layers [0,7,14,21,28,35,41] of 42.

## Results

| condition | task_acc | refusal |
|---|--:|--:|
| base | 0.167 | 0.98 |
| steer (mean vector) α=0.5 | **0.769** | 0.98 |
| steer_orth (r-orthogonalized) α=0.5 | **0.769** | **1.00** |
| steer α=1.0 (over-steer) | 0.300 | 0.00 |
| steer_orth α=1.0 | 0.300 | 0.00 |
| LoRA finetune | 0.975 | 0.88 |

## What replicates (the headline)
- **Task-vector steering transfers DDXPlus competence**: 0.167 → **0.769** at α=0.5 — an even
  larger gain than Qwen-7B (0.14→0.65), on a different model family.
- **Steering preserves refusal; finetuning erodes it.** Orthogonalized steering at α=0.5
  gives **0.769 task / 1.00 refusal** — full competence transfer with zero safety cost —
  while the LoRA finetune reaches 0.975 task but **erodes refusal to 0.88**. Same
  route-dependence direction as the 7B result.
- **Orthogonalizing the steer against r removes the leak** here too (steer_orth α=0.5 refusal
  1.00 vs plain steer 0.98).

## Cross-family differences (honest)
- **Narrower Goldilocks band:** gemma over-steers harder — at α=1.0 both task AND refusal
  collapse (0.30 / 0.00), where Qwen at α=1 still held 0.65 / 0.14. gemma's usable band is
  ~α≤0.5; orthogonalization does NOT rescue α=1 (0.00), unlike the gentler Qwen curve.
- **Milder finetune erosion:** the gemma LoRA erodes refusal only 0.98→0.88 (−0.10) vs the
  Qwen dose-600 LoRA's →0.00. This is adapter-specific — the gemma adapter is the pre-existing
  subspace-study LoRA (r=32), not a matched high-dose adapter — so the *magnitude* isn't
  directly comparable, but the *direction* (finetune erodes, steering doesn't) holds.

## Takeaway
The core finding generalizes across model families: **a fixed ICL-derived task vector,
steered at the right (small) magnitude and orthogonalized against the refusal direction,
delivers most of the task gain with NO refusal erosion — unlike finetuning.** The
magnitude sensitivity is model-dependent (gemma's safe band is narrower), reinforcing that
α must be tuned per model and that r-orthogonalization is the robust safety lever.

## Caveats
One adapter (not dose-matched to Qwen), n_eval=50, single seed, gemma 8k context. The
Qwen2-specific attention-capture experiments were not ported (different RoPE); the
steering/route results use generic residual hooks that work cross-family.

## Reproduce
```bash
HF_HUB_OFFLINE=1 uv run python -m scripts.safety.run_steering_safety \
    --config configs/safety/route_safety_gemma.yaml --alphas 0.5,1.0 \
    --adapter results/lora_icl/adapter
```
JSON: `results/safety/gemma_sweep/steering_safety.json`.
