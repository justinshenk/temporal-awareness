# The linear route-map W: Δh_L (LoRA shift) → Δh_B (ICL shift) strips the refusal component

For harmful prompts, per layer, fit `W Δh_L ≈ Δh_B` by ridge (dual form) on a fit split and
evaluate held-out:
- `Δh_B = resid(base, prompt+ICL) − resid(base, prompt)`  (activation-route shift)
- `Δh_L = resid(LoRA, prompt) − resid(base, prompt)`      (weight-route shift)

Qwen2.5-7B + DDXPlus LoRA (dose 600), 16-shot ICL, n_fit=80, n_eval=40, λ=1.

## Results (held-out)

| layer | cos(Δh_L,Δh_B) raw | cos(W·Δh_L,Δh_B) | residual | cos(Δh_L,r̂) **in** | cos(W·Δh_L,r̂) **out** |
|---|--:|--:|--:|--:|--:|
| 0  | −0.18 | **+0.97** | 0.23 | −0.05 | −0.16 |
| 7  | +0.23 | **+0.97** | 0.26 | −0.18 | −0.17 |
| 14 | +0.22 | +0.93 | 0.37 | −0.24 | −0.24 |
| 21 | +0.06 | +0.88 | 0.47 | **−0.50** | **+0.01** |
| 27 | +0.04 | +0.80 | 0.59 | **−0.45** | **−0.06** |

## Two findings
1. **The routes are linearly related off-task.** On harmful prompts the raw LoRA and ICL
   shifts are ~orthogonal (cos ≈ 0), but a held-out linear map lifts the alignment to
   **0.80–0.97** (cleaner early, looser late). So `W Δh_L ≈ Δh_B` genuinely holds out of
   sample — a strong strengthening of the on-task "shared task direction" (cos 0.74).
2. **W strips the refusal component, at the layers where it lives.** At L21/L27 (where the
   LoRA-specific direction ŵ aligns most with r), the LoRA shift enters with a strong
   refusal-axis component (cos(Δh_L,r̂) = −0.50 / −0.45) and exits with it removed
   (+0.01 / −0.06) — while still mapping to the real ICL shift (fidelity 0.80–0.88). This
   formalizes the central claim: **the difference between the weight route and the activation
   route IS the refusal component** — W is the linear LoRA→ICL converter, and what it does is
   project out r.

## Method note (a flawed test, corrected)
A first attempt measured `‖W·r̂‖` and found it 4–15× a random-direction baseline, which looks
like W *amplifies* r. That was the wrong quantity: `r̂` is in W's active input subspace (the
LoRA shifts have a big r-component), so feeding `r̂` in yields a large output — but that says
nothing about whether W *removes* r from the shifts. The correct test is the r-component of
the mapped shifts (cos in vs out), which gives the clean −0.50 → 0. "Is r in the null space"
≠ "does the map remove r from the data."

## Caveats
One adapter (dose 600, strongly eroded), n_eval=40, single seed, λ=1 (not tuned). The map is
rank ≤ n_fit; residual grows with depth (0.23→0.59), so the linear relation is approximate,
strongest at early/mid layers.

## Reproduce
`scripts/safety/run_route_map.py` with `configs/safety/route_safety_qwen.yaml` and
`--adapter results/safety/qwen_sweep/adapter_d600`. JSON: `route_map.json`.
