# Qwen E7 — counterfactual format-instruction patching on Qwen2.5-7B-Instruct

**Verdict: the counterfactual instruction state is delivered and perfectly readable, and on
neutral filler it carries a LARGE causal effect — but its behavioral expression is inverted
relative to the donor's instruction, and it is channel-asymmetric: the clinical/letter
productions move, the JSON channel never does. The precedent (mmlu) cell is a null once the
unrelated-fact control is subtracted.**

Run 2026-08-24 · `Qwen/Qwen2.5-7B-Instruct` · seed 42 · eager attention · driver
`run_format_patch.py` · artifacts `results/context_fatigue/qwen_e7_format_patch_{mmlu,code}/`
(+ `qwen_e7_format_patch_preflight/`). mmlu arm ran with `--filler-letter-only` (see design
note), n = 40; code arm depth 15, n = 39 (1 overflow skip; fill from artifacts: mmlu 0.890,
code 0.916).

## Design notes forced by Qwen

1. **Filler-mode pin.** Under System A, Qwen answers mmlu *filler* in the clinical format
   ('B\nSUPPORTING: …'), so the unmodified "precedent" arm would demonstrate the instructed
   format rather than counter-demonstrate it. `--filler-letter-only` stores each mmlu filler
   reply as its extracted bare letter, restoring the OLMo cell's semantics (verified with a
   6-turn reproduction of the accumulation stream before running).
2. **Closure degeneracy, exact on Qwen.** Under scale-0 closure the recipient's system span
   is unreadable, so the recipient's identity drops out: `pure_a ≡ pure_b` bit-identically,
   `patch_ab ≡ patch_aa`, `patch_ba ≡ patch_bb`, and ΔΔ_B→A = −ΔΔ_A→B by construction — the
   two directions are ONE measurement mirrored, not two confirmations. (This also holds on
   OLMo up to a one-token boundary leak; its Stage-1 report's two directions should be read
   the same way.)
3. **The pure-state delivery probe is vacuous under exact closure** — with `pure_a` and
   `pure_b` states identical, it has no trainable direction (AUC 0.500 by construction).
   Delivery is instead tested on the patched states directly: donor identity decodes at
   **AUC 1.000** (5-fold CV, layer 28, code arm).

## Results (paired case-resampled bootstrap, 10,000 draws)

| arm | ΔΔ_A→B (vs self-patch) | unrelated-fact control | instruction-specific (DiD) |
|---|---|---|---|
| mmlu (precedent, depth 42) | −0.214 [−0.490, +0.065] ns | −0.226 [−0.358, −0.094] sig | **+0.011 [−0.276, +0.299] ns** |
| code (neutral, depth 15) | **−2.488 [−2.777, −2.197]** | −0.108 [−0.230, +0.024] ns | **−2.379 [−2.671, −2.082]** |

- **mmlu**: everything the patch does is generic donor perturbation — the unrelated-fact
  swap (one city word) moves the contrast as much as swapping the entire instruction. No
  instruction-specific carried signal; noise floor ~5× OLMo's.
- **code**: a 2.4-nat instruction-specific effect, ~40× OLMo's (+0.056) and **sign-inverted**:
  A-donor states depress the A-format productions relative to B-donor states.

**Channel decomposition (code arm, A-donor − B-donor):** S_A ("ANSWER:") −2.47; S_P (bare
letter) −3.01; S_B (JSON prefix) **+0.02, inert**. The JSON instruction's content is never
behaviorally transported — Qwen produces no JSON in any condition (S_B ≈ −46 nats vs S_A ≈
−33; all 45 graded generations are free prose under closure). What moves is the *clinical*
family of productions, and it moves opposite to the donor that was instructed to produce it.

## Reading, cautiously

The states demonstrably carry which instruction was given (AUC 1.000, control-clean 2.4-nat
behavioral difference). But "carry" does not cash out as transport of the donor's format
disposition on Qwen: with the direct route closed, states formed under the *other*
instruction leave the clinical attractor stronger than states formed under the clinical
instruction itself. A candidate account — states formed under System A encode the
instruction as already-being-handled, and transplanting them satisfies rather than drives
the production — is speculative; distinguishing it needs Stage-2 (which positions/layers
carry the inversion) and an open-route arm. Cross-family, the E7 conclusion that survives
both models is: **the instruction's presence in context states is causally potent and
linearly readable, while its behavioral force is set by what the context demonstrates —
OLMo's precedent inverts a small carried signal; Qwen's neutral cell inverts a large one.**

## Controls

A→A no-op exact (preflight assert, max|Δlogits| = 0.0); rendered-length twin matching
asserted per probe; unrelated-fact pair token-matched; self-patch baselines throughout.
Generation agreement: vacuous under closure on Qwen (no canonical mode in any of 90 graded
replies across both arms) — the estimand is prefix-based.
