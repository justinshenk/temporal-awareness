# Counterfactual format-instruction patching (E7) — execution brief (2026-08-24)

## 1. Problem statement

§4.5's chain "read, represented, overruled" is causally incomplete: "represented" rests on
decoding (Probe 1, transfer AUC 1.000) and "overruled" on behavior, with nothing showing the
represented instruction is *causally recoverable* from context states. The carrier of the mode
is unidentified (four erase-nulls; the probe reads a correlate). Dongre et al.'s
distributed-encoding conjecture is untested by us. This experiment transplants hidden states
between counterfactual instruction conditions and asks: (a) do intervening-context states
causally carry *which instruction was given*, with direct system-prompt attention closed, and
(b) does applicable demonstrated precedent suppress that carrier's behavioral control?

## 2. Agreed solution approach

**Counterfactual pair.** Two contradictory format instructions, drafted to identical token
counts (Probe 1's matched-twin method): System A = the clinical template (`ANSWER:` +
`SUPPORTING:`), System B = a JSON answer shape (e.g. `{"answer": "<letter>"}`). B must be
lexically distinct from both A and from what any filler demonstrates, so three attractors are
distinguishable: A-format / B-format / demonstrated-format.

**Conditions (2×2 core).** Filler ∈ {code (inapplicable), mmlu bare-letter (applicable
precedent)} × patch ∈ {A→B, unpatched}. Plus direction reversal (B→A) and controls. Deep
context (mmlu at its collapse depth; code fill-matched). System span attention closed
(SpanAttentionClamp scale=0) in recipient AND donor for the primary arms; donor-open as a
secondary arm.

**Patch.** New instrument `SpanActivationPatch`: substitute hidden states at chosen
(layer-set, position-set) during a full-sequence forward, donor states harvested from the
donor forward. Stage 1: maximal patch — all intervening positions (system span EXCLUDED),
all layers. Stage 2 bisection: position sets {prev assistant turns, prev user turns, last
1/2/4 turns, all} × layer blocks {0–7, 8–15, 16–23, 24–31, all}, both directions.

**Metric.** Teacher-forced prefix log-probs of 2–4-token canonical prefixes per mode
(S_A, S_B; S_P = log Σ_letters P, summing bare + space-prefixed token variants), prefixes
chosen EMPIRICALLY from preflight generations of each pure condition. Estimand:
ΔΔ = (S_A − S_B)_patched − (S_A − S_B)_unpatched per item (and the S_P contrasts likewise) —
offsets from prefix length/base-rate cancel. Secondary: generate + grade n≈40 at the most
informative patch sites to confirm the prefix score predicts reply shape.

**Controls.** (i) A→A no-op patch: ΔΔ must be exactly 0 (bit-identity). (ii) size-matched
random-position patch. (iii) unrelated-fact donor (system prompts differ in an irrelevant
clause, same token count). (iv) **delivery control for the precedent null**: re-run Probe 1 on
the patched run's states — a behavioral null under precedent is reportable only with the probe
reading A-presence in patched-B (delivered but suppressed), else it is a failed transplant.

**Pre-registered predictions (from our results).** Neutral-filler A→B patch flips the
prefix contrast toward A (decode-time steering already installs the mode). The precedent cell
is open: erase-nulls suggest suppression; the upclamp beating 42 exemplars says the
instruction can win when re-weighted. Either outcome is a result.

## 3. Files likely to be modified

- `src/probes/context_fatigue/activation_patch.py` (new): `SpanActivationPatch`, donor
  capture, position/layer selection helpers.
- `tests/probes/context_fatigue/test_activation_patch.py` (new).
- `scripts/context_fatigue/run_format_patch.py` (new driver; reuses `_cf_common`,
  `run_format_erosion`'s filler machinery, `SpanAttentionClamp`, letter-variant scoring).
- `results/context_fatigue/e7_format_patch*/` artifacts; report `E7_FORMAT_PATCH.md`;
  `numbers.md` rows if any number enters the tex.

## 4. Non-goals

- No goal-value (Zurich/Tokyo) variant in this pass — follow-on with the same instrument.
- No erasure re-runs, no per-head patching, no Qwen (OLMo-2-7B-Instruct first).
- No paper edits until the report is written and numbers are rowed.

## 5. Operational constraints

- Box: A100 80 GB preferred (interleaves with the Qwen queue; ~6–10 GPU-hours total) or the
  4500 after the all-layer queue drains. `HF_HUB_OFFLINE=1`; preflight before every stage.
- Donor/recipient position alignment asserted (identical seq length; identical token ids
  outside the system span) — abort loudly on mismatch, never truncate/pad silently.
- Patching uses full-sequence forwards (no KV-cache surgery); batch 1.

## 6. Acceptance criteria

- A→A patch: max |Δ logits| = 0.0 (test-pinned and asserted in the driver preflight).
- Token-count equality of A/B (and unrelated-fact) system prompts asserted.
- Stage-1 report states, per filler condition and direction: ΔΔ with case-resampled 95% CI,
  random-position and unrelated-fact controls, and the delivery-probe AUC on patched states.
- Generation subset agreement: prefix-score sign predicts graded reply mode in ≥90% of the
  n≈40; otherwise the metric is revised before any claim is made.
- A falsifying outcome (no flip under neutral filler; flip under precedent) is a result and
  gets reported as one.

## 7. TDD test-forward process

Instrument before driver: (1) capture/substitute round-trip on the tiny OLMo-2 fixture —
patched forward equals donor forward when ALL positions+layers are patched; (2) A→A
bit-identity; (3) position-subset patch changes only downstream logits (causality sanity);
(4) misaligned lengths raise. Then driver preflight: 1 pair end-to-end, empirical prefixes
printed, transcripts rendered and read before the full run.

## 8. Test expectations

New tests live in `tests/probes/context_fatigue/test_activation_patch.py`, run offline on the
config-built tiny OLMo-2 fixture (see current_task.md "Verified facts"), and the full suite
stays green (678 + new).
