# PARKED (Paper A) — cross-model LoRA capability transmission via linear maps

**Status:** designed in chat 2026-08-21, user: "let's try to do it later". Not started.

**Hypothesis (user's):** fine-tuned chat models have similar-enough activation-space structure
that a LoRA's capability, trained on chat model A, can be transmitted to an untrained chat
model B through a learned linear map between their activation spaces.

**Assets that make this cheap here:** the same DDXPlus task LoRA trained independently on
Qwen2.5-7B-Instruct and gemma-2-9b-it (+ Qwen2.5-1.5B-Instruct for a within-family rung);
`extract_shifts.py` (LoRA shift sets vs shared unadapted baseline); `run_subspace_comparison.py`
(shift subspaces); `AdditionSteeringHook` with `last_token`/`decode_time` modes.

**Design (activation route; weight-space mapping second if ever):**
1. Fit per-layer linear maps M: A→B (ridge/Procrustes) on final-position residuals over a
   shared prompt corpus — prompt-level alignment, so tokenizer mismatch is moot.
2. Transmit δ_A = shift((A+LoRA) − A); steer B with M·δ_A at the mapped layer
   (function-vector application). Score DDXPlus.
3. Calibrated readout: B's gain as a fraction of (B's own-LoRA ceiling − B's unadapted floor).
   The gemma adapter is the ceiling — this turns partial transfer into a number.
4. Controls: norm-matched random vector through the same map (dose-matched, per the E6 steering
   lesson); the shuffled-label adapter's shift through the map (format-not-task probe); A→A
   self-transfer sanity; report the map's held-out R² so a null is attributable.
5. Ladder: Qwen-7B → Qwen-1.5B first (same family/tokenizer; if this fails, stop), then
   Qwen ↔ Gemma both directions.

**Refinement if mean-shift is too blunt:** transmit the top-k principal directions of the shift
subspace separately rather than the mean.

**Tie-in:** Paper B's E6 showed a mean-diff mode vector installs behavior direction-specifically
within a model; this asks whether such vectors survive a linear change of basis between models.
