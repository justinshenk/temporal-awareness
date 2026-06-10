# DAS: task-loss-trained δ-subspace vs PCA bands (oracle-subspace contrast)

## 1. Problem statement
PCA bands pick the injected δ-subspace by **variance**; the capability is invisible to that
objective (top-64 = 0% recovery at 55% of δ-energy; recovery only switches on at top-512 = 0.45).
Complement test settled it: δ−top64 (`tail64`, 4032 dirs, 45% energy) = **0.00** → the low-variance
tail is not causally sufficient; it's a fidelity/wide-band requirement, and PCA-by-variance can't
*find* the causal directions. Question DAS answers: **are the causal directions low-RANK but
low-VARIANCE — findable by a task-loss subspace search at a rank where PCA recovers nothing?**

## 2. Approach — the clean one-variable contrast
The DAS intervention is *identical* to the PCA-band injection `a + Π_R(δ_true)` at L20 every decode
step (literally `projected_injection` with `V = R`). Only the subspace differs:
- **PCA**: `V` = top-r eigenvectors of δδᵀ (variance; already run: top8/64/256/512 = 0/0/0.05/0.45).
- **DAS**: `R` = orthonormal d×r learned by **gradient descent on a behavioral loss** — make
  base+`Π_R(δ)` reproduce LoRA's per-step greedy next-token decisions on the base trajectory.

Same injection mechanism, same oracle δ, same closed-loop lockstep eval → DAS-R@r vs PCA-top-r at
matched rank isolates **subspace-selection objective**. If DAS-R@64 ≫ PCA-top64 (=0), the directions
are low-rank, just low-variance → task-loss/whitened fit is the salvage with a real prior on success.
If DAS-R needs r≈512 too, it's a genuine wide-band fidelity floor → the LoReFT map (learn the value,
exp. 2) is the honest next move.

**Decisions (locked):** oracle-subspace contrast (not full map yet); ranks **8, 64, 256, 512**
(match PCA bands); layer L20; PCA anchor = existing `lockstep_pca_band_L20.json` (same train-split δ,
n=100, so directly comparable).

## 3. Training signal (matches the eval distribution)
Per train problem (train split, disjoint from test contrast set): generate **base** CoT (adapter off)
→ `full_ids, plen` (same trajectory dist as eval). Teacher-force base → `a_L (seq,d)`; teacher-force
LoRA → `lora_L (seq,d)` + `lora_logits`. `δ_L = lora_L − a_L`; **target[t] = argmax(lora_logits[t])**
(LoRA's greedy next-token given *base* context — exactly what the oracle injects to reproduce).
Train R: inject `a + Π_R(δ)` at L20 (all positions), base upper layers → patched logits, CE vs target
over CoT positions. Only R trains (orthonormal via reduced-QR of a raw param); δ cached/detached.
Eval = closed-loop lockstep recovery on the 34-problem contrast set (honest re: closed-loop drift).

## 4. Files
- NEW `src/probes/attribution/das_subspace.py` — `OrthoSubspace` (QR-orthonormal R), `inject_value`
  (`a+Π_R(δ)`, differentiable), `subspace_lm_loss` (CE over CoT positions). Pure/testable.
- NEW `scripts/attribution/das_subspace_gsm8k.py` — CLI: build train cache, `train_subspace` per rank,
  eval via `eval_band` (reused), load PCA anchor, write JSON + table.
- NEW `tests/attribution/test_das_subspace.py` — fake CPU model: R orthonormal; r=d ⇒ inject=lora_resid;
  trainer reduces loss + R stays orthonormal; loss indexing correct.
- REUSE `lockstep_oracle.{OverwriteResidualHook,projected_injection,lockstep_generate}`,
  `lockstep_pca_band.eval_band`, `nonlinear_delta_gsm8k.collect_base_traj`/`load_contrast`,
  `cot_collection`, `delta_subspace.pca_bands`.

## 5. Non-goals
No edits to oracle/PCA modules or existing scripts. Not learning the injected *value* (LoReFT = exp. 2,
gated on this). Single layer L20. No heavy deps (hand-rolled QR, not geotorch).

## 6. Constraints
Single GPU 7B bf16; R is d×r f32 (tiny). Train teacher-forced batch=1 (hooks assert batch=1),
full base forward with grad-enabled overwrite hook (upper-layer activations only in grad path).
Eval = slow lockstep (~7 min/rank). Deterministic seed. Measure epoch time on rank-8 first; cap
epochs/n-train if too slow — log any narrowing.

## 7. Acceptance criteria
- AC1: r=d (full rotation) ⇒ eval recovery ≈ full-oracle (~0.75) — wiring/alignment check.
- AC2: per-rank table {DAS-R recovery vs PCA-top-r recovery} over 8/64/256/512.
- AC3: trained R verified orthonormal (RᵀR≈I) post-train; result JSON has both curves + train loss.
- AC4: fake-model unit tests green.

## 8. TDD (test-forward)
Fake model with `.model.layers` + `lm_head`: (a) `OrthoSubspace().forward()` orthonormal; (b)
`inject_value` with R=I (r=d) equals `a+δ`; (c) one Adam step on a differentiable fake forward lowers
loss and R stays orthonormal; (d) `subspace_lm_loss` selects the right CoT positions/targets. GPU
(script, not pytest): AC1 r=d ≈ oracle before trusting any rank number.

## Prior result (this branch, uncommitted) — PCA band sweep @L20
top8/64/256/512/1024/full recovery = 0/0/0.05/0.45/0.65/0.75; energy 0.36/0.55/0.71/0.80/0.88/1.0.
δ−top64 (tail64) = 0.00 @ 45% energy; δ−top8 (tail8) = 0.00 @ 64% energy → fidelity threshold (strong
Reading B). Figure: results/figures/pca_band_recovery_L20.png. Writeup: 2026-06-10-pca-band-complement.md.

## RESULT (2026-06-10) — DAS subspace vs PCA: salvage REFUTED
DAS-R recovery @ r=8/64/256/512 = 0/0/0/0 (PCA top-r = 0/0/0.05/0.45) despite teacher-forced CE
2.67/0.96/0.16/0.038. Task-loss subspace selection is strictly ≤ variance and collapses to 0 closed-loop
even at near-zero CE. Mechanism: teacher-forced CE is anti-aligned with closed-loop δ-fidelity — DAS
overfits the on-policy trajectory / picks a low-energy 512-subspace; PCA captures 80% of δ-energy and
generalizes. Wall = VALUE-fidelity in a wide variance band, not subspace choice. Subspace search ruled
out as salvage. Files: src/probes/attribution/das_subspace.py, scripts/attribution/{das_subspace_gsm8k,
plot_das_subspace}.py, tests/test_das_subspace.py (5 passing). Figure: results/figures/das_vs_pca_L20.png.
Writeup: 2026-06-10-das-subspace.md. Next fork: (a) energy-capture diagnostic (confirm mechanism) or
(b) LoReFT value-map on-policy (the upper bound; note DAgger map already failed 2026-06-07).
