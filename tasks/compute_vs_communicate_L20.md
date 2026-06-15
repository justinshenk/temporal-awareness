# Does the L20 oracle install a capability, or just transport the answer? (E1 logit-lens + E3 lesion)

## 1. Problem statement
The lockstep oracle recovers GSM8K accuracy (full δ = 0.75, top-512 = 0.45) by overwriting base's L20
residual with the LoRA's L20 residual every decode step on the base-fails/LoRA-solves contrast set
(`2026-06-10-pca-band-complement.md`). A successful patch cannot distinguish two readings:

- **H_compute** — base layers 21–31 perform the genuine downstream reasoning *once given a corrected
  L20 state*; the capability is latent in base and L20 was the missing input.
- **H_communicate** — the LoRA's L20 residual already encodes the answer / next reasoning tokens, and
  base's 21–31 merely transcribe it; the capability lives in the LoRA's ≤20 stack and base is a readout.

L20-of-32 is suspicious: patching deeper is *more* communication by construction. Prior nulls (ridge /
MLP / DAgger maps fitting `δ = f(h_base)` all ≈0; only the true per-step δ works) lean toward
H_communicate but are confounded by "the map is hard to learn" (DAS/PCA: wide-band value-fidelity wall
+ exposure bias). This task resolves it with two experiments that look *where the answer crystallizes*
(E1) and *whether base's downstream is load-bearing* (E3).

## 2. Agreed solution approach
Both reuse the existing full-δ oracle (= `projected_injection` with full-rank `V`) on the same 20
contrast problems; only the readout/intervention is added.

**E1 — logit-lens at the patch site.** Run the base+L20-full-δ oracle. During the injected base
forward, capture post-injection residuals at readout layers {20,22,24,26,28,30,31}. Project each
through `final_norm`+`lm_head` (logit lens) and, for the token the oracle actually emits that step,
record its rank/logit at each layer. Compare against the LoRA-natural run and the base-only run on the
same tokens. Restrict the headline metric to **answer-bearing tokens** (digit tokens; the final
`#### N` span), not format/language tokens.
- H_communicate ⇒ the emitted answer token is already top-1 at L20; rank/logit barely move L20→L31.
- H_compute ⇒ it is buried at L20 and climbs to top-1 only after several base layers.

**E3 — downstream-lesion necessity.** Apply the L20 full-δ patch, then *identity-ablate* base layers
21→31 cumulatively (top-down: {31}, {30,31}, …, {21..31}) — a decoder block made identity returns its
input hidden state, removing that block's computation. Measure recovery vs #layers ablated.
- H_communicate ⇒ recovery stays high while several downstream layers are ablated (little compute left
  to do).
- H_compute ⇒ recovery collapses with each ablated layer (21–31 are load-bearing).

**Control: LoRA-natural under the same ablation** (replaces the base-solvable control — base solves
≈nothing on GSM8K, so a base-solvable set is empty). Run the *full LoRA* (capability natively held,
not transplanted) on the same contrast set and apply the identical cumulative ablation. This is the
purest form of the question: does the model that actually solves these still need 21→31?
- If LoRA-natural is robust to ablating 21→31 ⇒ the answer is already at *its* L20 ⇒ 21–31 are a
  readout even for the LoRA ⇒ base+patch recovery is communication.
- If LoRA-natural collapses too ⇒ the answer is *not* present at L20 in the model that has the
  capability ⇒ base+patch recovery requires base's 21–31 to do analogous computation (H_compute), and
  the matched curves also rule out "ablation just breaks generic fluency."

**Honest caveat baked into E1.** "Logit-lens top-1 at L20" is *sufficient but not necessary* for
communication — base's 21–31 are a nonlinear decoder, so a token can be determined at L20 without being
linear-lens-top-1. So E1's "already decodable" is a conservative lower bound on communication; the
causal complement is E3. The two together adjudicate; neither alone is decisive.

## 3. Concrete traces
- Contrast problem 0 (base-fails/LoRA-solves), oracle emits the digit `"42"` at some step t.
  - E1 records: rank of `"42"` under logit-lens at L20, L22, …, L31 in (a) base+patch, (b) LoRA-natural,
    (c) base-only. Output row: `{step:t, token:"42", is_answer_bearing:true, rank_by_layer:{20:317,
    24:12, 28:1, 31:1}, logit_by_layer:{...}}`. Communicate ⇒ `rank_by_layer[20]==1`; compute ⇒ it
    descends to 1 only by ~L28.
  - E3 records: `recovery_patch(k)` over the 20 problems at ablation level k=0 (== full oracle ≈0.75),
    k=1 ({31} ablated), …, k=11 ({21..31} ablated), and `recovery_lora(k)` = LoRA-natural on the same
    problems under the same ablation (k=0 ⇒ 1.0 by construction).

## 4. Files
New (no edits to committed lockstep behavior):
- `src/probes/attribution/logit_lens.py` — `LogitLens` (final-norm + unembed projection; per-layer
  rank/logit of a target token id). Multi-word name.
- `src/probes/attribution/layer_ablation.py` — `IdentityAblationHook` (identity-ablate decoder layers;
  batch=1, mirrors `OverwriteResidualHook` conventions).
- `scripts/attribution/logit_lens_patch_gsm8k.py` — E1 driver (oracle run + per-layer lens, contrast +
  controls → JSON).
- `scripts/attribution/downstream_lesion_gsm8k.py` — E3 driver (oracle + cumulative ablation sweep +
  base-solvable control → JSON).
- `scripts/attribution/plot_logit_lens.py`, `scripts/attribution/plot_downstream_lesion.py`.
- `tests/test_logit_lens.py`, `tests/test_layer_ablation.py`.
- `results/attribution/2026-06-13-compute-vs-communicate-L20.md` (existing writeup format).

Reused unchanged: `lockstep_oracle` (`OverwriteResidualHook`, `lockstep_generate`,
`projected_injection`), `PerTokenResidualCapture`, `nonlinear_delta_gsm8k.load_contrast`,
`gsm8k_prompts` (`extract_pred_number`, `numeric_match`), `attribution_common`
(`load_base_and_lora`, `prompt_token_ids`), config `metamath_llama2_gsm8k.yaml`.

## 5. Non-goals
- No tuned-lens (affine per-layer probe) — optional follow-up if logit-lens is too blunt.
- No retraining; no changes to `lockstep_oracle.py`/`lockstep_pca_band.py` semantics (additive modules).
- E2 (patch-layer sweep) and E4 (counterfactual δ) deferred.
- Not the full GSM8K test set — the 20-problem contrast set only (E3 control is LoRA-natural on the
  same set, not a separate base-solvable set, which is ≈empty).
- No changes to the crossmodel/affine-bridge work.

## 6. Operational constraints
- GPU, bf16, Llama-2-7B + metamath-LoRA; lockstep is a dual forward per step; max_new 256; n_contrast 20.
- E1 adds capture at ~7 layers (cheap). E3 runs ≤12 ablation levels × 20 problems × dual-forward
  (heavier) — allow `--ablation-levels` subsetting and `--n-contrast`.
- Cloud GPU per project convention; only CPU tests run locally.
- Lens must use the model's real `final_norm` + `lm_head` (tied weights) — not a re-derived unembed.

## 7. TDD test-forward expectations
- `test_logit_lens`: on a tiny CausalLM, the lens applied to the **final** layer's residual equals the
  model's actual logits (atol ~1e-4) — proves norm+unembed wiring. Rank/logit extraction matches a
  hand-set logits vector. Target-token rank is in [0, vocab).
- `test_layer_ablation`: with the hook on layer `li`, that layer's output hidden == its input hidden
  (atol ~1e-5, i.e. a no-op); ablating *all* layers makes the final pre-norm hidden equal the
  post-embedding hidden; batch>1 raises (matches `OverwriteResidualHook`).
- Continuity check (GPU, not a unit test): E3 at ablation level k=0 reproduces the full-δ oracle on the
  same 20 problems (≈0.75, matching `lockstep_pca_band_L20.json` `full`).

## 8. Acceptance criteria
- CPU tests pass (lens==model logits at final layer; identity ablation is a no-op).
- E1 yields, for answer-bearing tokens: the crystallization-layer distribution and the L20→L31 rank/
  logit movement for base+patch vs LoRA-natural vs base-only.
- E3 yields `recovery_patch(k)` for k=0..11 alongside the `recovery_lora(k)` control curve, so a
  recovery drop is read against whether the natively-capable LoRA also needs those layers (separating
  "answer already at L20" from "21–31 are load-bearing computation").
- Writeup adjudicates H_compute vs H_communicate, explicitly noting E1's lower-bound caveat and E3's
  control, in the established results format. JSON + figure per experiment.
