# Is the L20 trajectory state sparse in time, or dense? (temporal decomposition of the oracle)

**Question.** E3 (`2026-06-13-compute-vs-communicate-L20.md`) showed the L20 full-δ lockstep oracle
installs a *distributed intermediate*, not a transported answer, and E1b showed base's deficit is
**trajectory control**, not per-step arithmetic (teacher-forced on the correct chain, base predicts
computed-result tokens at 96.8%). That leaves one quantity undetermined and decision-critical for a
donor-free steerer: **how much** of the LoRA's per-step state must be transported. Is the recovery
carried by a *sparse* set of decode steps (a few structural decision points — then a small stateful
controller is well-posed), or is it *dense* (near-every-step — then "transplant the full trajectory"
hardens into a wall)?

No prior sweep touched the time axis: `lockstep_fidelity_sweep` varied shift *magnitude* (t),
`lockstep_pca_band` varied *dimensions* (V). This varies **time**: patch only the decode steps a
**gate** selects and measure recovery vs the fraction of steps patched.

**Apparatus.** Same L20 oracle and base-fails/LoRA-solves contrast set (first 20, byte-identical to
the E3/E1b set). New code: `src/probes/attribution/temporal_gate.py` (`gated_lockstep_generate` +
gates), driver `scripts/attribution/temporal_oracle_gsm8k.py`, plot `plot_temporal_oracle.py`, tests
`tests/test_temporal_gate.py` (9 passing, CPU). A skipped step is just `inject.set_values({})` (the
overwrite hook no-ops a layer absent from `values`) and skips the LoRA capture forward, so the gated
decode is additive over the oracle and sparser gates are cheaper. Gates decide from the
**already-emitted** context only (no look-ahead). Structural gates reuse the post-`=` result-span
state machine from `gold_token_lens_gsm8k.computed_flags`, with one refinement for gating: a newline
**closes** the span (a line boundary is a structural/planning token, not an intra-result space).

## Result

| gate | fraction of steps patched | recovery (n=20) |
|---|--:|--:|
| **periodic_1** (full oracle, positive control) | 1.000 | **0.750** |
| periodic_2 | 0.500 | 0.000 |
| periodic_3 | 0.336 | 0.000 |
| periodic_4 | 0.250 | 0.000 |
| periodic_6 | 0.168 | 0.000 |
| periodic_8 | 0.125 | 0.000 |
| result_only (computed digits after `=`) | 0.029 | 0.000 |
| **planning_only** (complement of result_only) | 0.944 | **0.700** |
| step_boundary (first token of each line) | 0.094 | 0.000 |

`periodic_1` reproduces the full-δ oracle exactly (0.750 == `downstream_lesion_L20.json` k=0), so the
wiring is sound. Figure: `results/figures/temporal_oracle_L20.png` — a step function, flat at 0 until
~0.5 and rising to 0.75 only at full patching.

## Reading — the trajectory state is DENSE in time; the predicted sparse target does not exist

Recovery is **all-or-nothing in the fraction of steps patched**. Every gate that patches ≤ 50% of
steps recovers exactly **0.000** — including the blind `periodic_2` (every other step) *and* the
structural `step_boundary` (line-starts, 9.4%). The only gates that recover anything are the two that
patch ≥ 94% of steps. So there is **no sparse subset** — periodic or structural — that installs the
capability. The pre-registered "planning_only at a small fraction ≈ oracle" hypothesis is **refuted
in its sparse form**: planning_only recovers (0.700 ≈ 0.750) but only by patching **94.4%** of steps.

Two facts survive and refine, rather than overturn, E1b:
- **Skipping the result digits is nearly free** (0.750 → 0.700): `result_only` is just 2.9% of decode
  steps, and removing exactly those from the oracle barely dents recovery. This is the causal echo of
  E1b — base computes the post-`=` results itself given the surrounding chain, so the oracle need not
  supply them. The deficit is the *other* 94%, the structure-laying tokens, exactly as E1b's
  planning-not-arithmetic split predicted.
- **But "planning" here is ~94% of the trajectory, not a few decision points.** The structure base
  cannot lay down autonomously is *almost the entire per-step trajectory*, not a sparse set of
  branch points. Dense planning, not sparse control.

**Caveat (lower vs upper bound).** `periodic_k` is a *blind* gate: an unpatched step lets base's
trajectory drift, so each subsequent patch re-injects onto a desynced context — periodic→0 is a
**lower bound** on achievable sparsity. But `step_boundary` is a *coherent structural* sparse gate
(it patches the same line-start tokens the LoRA emits) and also →0, and the only recoveries sit at
≥94%. The conclusion (dense, not sparse) does not rest on the blind gate alone. Unprobed: the
0.50–0.94 fraction gap — we know the threshold is above 50% and at/below 94%, not its exact value;
this refines *where* the wall is, not *whether* it is dense.

## Verdict

**The L20 trajectory state that the oracle transports is dense in time: recovery requires patching
the overwhelming majority (≥94%) of decode steps; every sparse gate (≤50% periodic, 9% structural
line-starts, 3% result-digits-only) recovers nothing.** The one cheap omission is the computed-result
digits (E1b's arithmetic, which base does itself). This **hardens** the standing negative — a
procedure edit is genuine distributed computation, recoverable only by transplanting essentially the
full per-step trajectory state, not a low-rank/pointwise *or time-sparse* function of the recipient
activation.

**Phase 2 decision — do NOT build the sparse stateful controller.** The motivation for a donor-free
recurrent/state-space steerer firing at a few decision points was a *sparse* target. The target is
dense: such a controller would have to reproduce the LoRA's near-exact 4096-d L20 state at ~94% of
steps — i.e. the full trajectory, against the near-exact bar `lockstep_fidelity_sweep` already
established (recovery rises only near t=1). That is the donor itself, not a learnable shortcut. The
honest next step is the strengthened negative recorded here, not the controller.

## Reproduce
```
uv run python -m scripts.attribution.temporal_oracle_gsm8k \
    --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20 \
    --gates periodic:1,2,3,4,6,8 result_only planning_only step_boundary
uv run python -m scripts.attribution.plot_temporal_oracle --json results/attribution/temporal_oracle_L20.json
```
JSON: `results/attribution/temporal_oracle_L20.json`. Brief: `tasks/temporal_oracle_decomposition.md`.
