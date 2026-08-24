# E7 — counterfactual format-instruction patching, Stage 1 (maximal patch)

**Verdict: the intervening-context states causally carry which instruction was given — and
applicable precedent inverts, not merely suppresses, that carried signal.** Under neutral
(code) filler the maximal patch flips the format contrast toward the donor's instruction in
both directions; under mmlu precedent the same patch flips it *away* from the donor in both
directions. Delivery is confirmed in both arms (transfer AUC ≥ 0.97 on the patched states),
so the precedent-cell reversal is a fact about behavioral control, not a failed transplant.

Run 2026-08-24 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · eager attention · artifacts
`results/context_fatigue/e7_format_patch_{mmlu,code}/` (`turns.csv`, `summary.json`,
`delivery_states.npz`) · driver `scripts/context_fatigue/run_format_patch.py` · instrument
`src/probes/context_fatigue/activation_patch.py` (28 pinned tests, both tiny families).

## Design as executed (deviations from the brief flagged)

System A = the clinical template; System B = a JSON answer shape, trimmed to render to the
identical transcript token count (48 isolated tokens proved insufficient — the trim's final
word merges with template tokens, so the twin is matched on **rendered** length; caught by
the §5 alignment assertion in preflight). Same accumulated filler tokens under both systems;
patch = all intervening positions (system span excluded), all 32 layers. Readouts
teacher-forced with the system span's attention closed (scale=0) in every scored forward;
donor states captured open.

**Baseline deviation (forced by a preflight finding):** under closure, a pure run's context
states form closed while donor states are captured open, so pure-run baselines confound
"received open-captured states" with donor identity — the unrelated-fact control moved as
much as the counterfactual patch (−0.81 vs −0.49, preflight n=1). Every patched condition is
therefore compared to a **self-patch baseline** (the recipient's own open-captured states
through the identical procedure): ΔΔ = (S_A−S_B)_patch_donor − (S_A−S_B)_patch_self. With
this baseline the unrelated-fact control is null in both arms.

A→A no-op: asserted `max|Δlogits| = 0.0` in preflight (exact, per the instrument's tests).
Random-position control: not applicable at Stage 1 (the maximal patch uses every intervening
position); it belongs to Stage 2's subsets.

## Results (n = 40 probes per arm, paired case-resampled bootstrap, 10,000 draws)

Prefixes: S_A = "ANSWER:", S_B = `{"answer":`, S_P = log Σ P(first token is a letter).

| arm | direction | ΔΔ (S_A−S_B) | 95% CI | sig |
|---|---|---|---|---|
| code (neutral, depth 15, fill ≈0.6) | A→B | **+0.056** | [+0.016, +0.095] | yes |
| code | B→A | **−0.071** | [−0.108, −0.033] | yes |
| code | unrelated-fact | −0.011 | [−0.039, +0.016] | no |
| mmlu (precedent, depth 42, fill ≈0.87) | A→B | **−0.076** | [−0.124, −0.025] | yes |
| mmlu | B→A | **+0.066** | [+0.013, +0.117] | yes |
| mmlu | unrelated-fact | −0.031 | [−0.065, +0.002] | no |

Positive A→B / negative B→A = movement toward the donor's instruction. The code arm matches
the pre-registered prediction (decode-time steering installs the mode; here the carrier
arrives through the states with the direct route closed). The precedent cell was
pre-registered open: the answer is a symmetric **sign inversion** — donor-A states push the
precedent-laden recipient *away* from A. Effects are ~1–1.5% of the standing (S_A−S_B)
contrast (≈+5 to +6.6 nats everywhere: "ANSWER:" always beats the JSON prefix; at these
depths the dominant behavior is S_P's bare letter / free prose).

S_P moves consistently in the mmlu arm (A→B −0.014 [−0.024, −0.004]; B→A +0.029 [+0.015,
+0.044]); ns in the code arm.

## Controls and acceptance criteria (§6)

- A→A bit-identity: asserted, exact. Token-count equality: asserted on rendered transcripts,
  per probe, both pairs.
- Unrelated-fact donor: null in both arms (table above).
- Delivery probe: logistic probe trained pure-A vs pure-B on final-position states, tested on
  patch_ab vs patch_bb: **AUC 0.997 (code, L16), 0.972 (mmlu, L12)** — the transplanted
  instruction is linearly readable in the patched runs; the precedent cell is "delivered but
  overruled (inverted)", not a failed transplant.
- Generation agreement: every reply with a canonical mode matches the top prefix score
  (33/33, mmlu arm — all bare-letter, S_P top in all). Under closure the code arm generates
  free prose with no canonical mode (consistent with E5's closure collapse), so the ≥90%
  criterion is met where defined but vacuous there; the prefix scores, not generations, carry
  the estimand.

## Interpretation, briefly

(a) Yes: with direct system-prompt attention closed, transplanting intervening-context
states moves the model's format disposition toward the donor's instruction — the states
causally carry it. (b) Under applicable demonstrated precedent the carried signal's
behavioral force is not just suppressed to zero but slightly inverted, in both directions —
as if the precedent-laden context treats the carried instruction as marked-against. The
delivery probe rules out transplant failure. Stage 2 (position/layer bisection) can now ask
*where* the carrier lives; the inversion makes the precedent cell the more interesting
bisection target.

## Provenance

Preflight transcripts and replies read before the full runs
(`e7_format_patch_preflight/`). Voided intermediate design (pure-run baselines) never ran at
n>1; no artifacts to void. 0 probes skipped in either arm.
