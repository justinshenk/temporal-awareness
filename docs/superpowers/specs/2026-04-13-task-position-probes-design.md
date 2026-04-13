---
name: Task-Position Probes for Context Fatigue (v1)
description: Design for linear and MLP probes that decode task-position signals from Gemma-9B-IT residual streams, and the analyses they unlock for the context-fatigue investigation.
status: draft
date: 2026-04-13
owner: research
---

# Task-Position Probes for Context Fatigue (v1)

## Motivation

The context-fatigue writeup (`scripts/context_fatigue/WRITEUP.md`) established that fatigue is linearly decodable from the residual stream, but flagged a critical caveat: the AUC-1.0 "context fill" probe may partly reflect raw positional encoding rather than a semantic fatigue signal. Meanwhile, upcoming-failure prediction stalls at AUC 0.67–0.68, and F90871 (a document-boundary SAE feature) is universally suppressed as context fills.

The research team's existing lane is *temporal activation monitoring* — probing the model's belief about how far it is through a task. v1 of this study builds the probe family that lets us ask whether fatigue is a function of the model's **subjective sense of task-lateness**, not just raw token count. This is the prerequisite for later causal work (steering vectors, refresh-controller experiments).

## Scope

**In scope (v1):**
- Train three regression probes on Gemma-9B-IT residual streams against task-position targets
- Validate probes against strong baselines that isolate the signal from raw positional encoding
- Run five analyses that directly test the hypotheses raised by the context-fatigue writeup

**Out of scope (deferred to v2+):**
- Causal interventions (steering vectors, activation patching)
- Prompt-refresh controller experiments
- Heterogeneous multi-task and single-task (NarrativeQA) probes
- Cross-model replication on Qwen-7B and Llama-8B

## Model and data

### Primary model
- **Gemma-2-9B-IT** (42 layers, 8k context). Chosen because (a) the existing SAE work anchors at L20, enabling direct comparison with F90871, and (b) IT-base pairs exist for the eventual base-vs-IT analysis.

### Base-model comparison
- **Gemma-2-9B (base)** for Analysis 5 only.

### Source traces
- Existing DDXPlus multi-task runs (Gemma-9B-IT) from the context-fatigue sweep. Homogeneous, sharp case boundaries, already labeled for correctness.

**Open question for user (data provenance):**
Are the residual-stream activations from the existing Gemma-9B-IT DDXPlus runs still persisted on disk? If yes, v1 is ~a day of probe training. If no, we need a re-extraction pass (~half a day of GPU). The spec assumes we may need to regenerate; the implementation plan will branch on this.

## Label schema

For every token position `t` in a trace with `N` cases and case boundaries `b_1 < b_2 < ... < b_{N+1}` (where `b_i` is the start token of case `i`):

| Target | Definition | Type |
|---|---|---|
| `task_index(t)` | `i` such that `b_i ≤ t < b_{i+1}` | ordinal, 1..N |
| `within_task_fraction(t)` | `(t - b_i) / (b_{i+1} - b_i)` | continuous, [0, 1] |
| `tokens_until_boundary(t)` | `b_{i+1} - t`, trained in `log1p` space | positive integer |

**Covariates retained for controls:**
- `raw_token_position`: `t`
- `total_context_length`: `b_{N+1}` (trace length)

Labelers must be pure functions on `(trace_tokens, case_boundaries)`, implemented in a dedicated module and unit-tested without a model.

## Probe architecture and training

### Probe types
- **Linear**: ridge regression (sklearn), default α tuned on held-out trace split
- **MLP**: single hidden layer, width 128, ReLU, MSE loss, Adam. Only meaningful if it beats linear — per the writeup's Section 4.1 finding, we expect minimal lift.

### Layer sweep (Gemma-9B-IT, 42 layers)
`{0, 10, 20, 30, 41}`. L20 is mandatory (SAE anchor); others give early/mid/late/final coverage.

### Activation granularity
Residual stream at **every token**, not just case-final tokens. This enables per-trace probe-readout trajectories, which are the headline figure.

### Train/test split
**By-trace**, 80/20. Per-token splits leak because within-trace tokens are highly correlated. Fix the random seed (`42`) in the spec so splits are reproducible.

### Metrics

| Target | Primary metric | Secondary metric |
|---|---|---|
| `task_index` | Spearman ρ | MAE (in cases) |
| `within_task_fraction` | R² | MAE (in fraction units) |
| `tokens_until_boundary` | R² on `log1p` space | median APE (raw tokens) |

### Critical baselines (must be beaten)

1. **Raw-token-position baseline.** A probe whose only feature is the scalar `t`. If the residual-stream probe doesn't beat this, it is reading positional encoding, not a semantic signal. **This is the single most important control in v1.**
2. **Context-fill projection.** Project residual-stream activations onto the existing context-fill probe direction, then train a linear head from that scalar to each target. If the residual probes don't beat this, they learned nothing new.

## Analyses (the deliverable)

### A1. Task-position vs. context-fill orthogonality
**Question:** Is task-position a new direction or a renaming of context-fill?
**Method:** Cosine similarity between each learned probe direction and the context-fill probe direction, per layer.
**Prediction:** `task_index` aligns strongly (both monotone in total tokens). `within_task_fraction` and `tokens_until_boundary` are substantially orthogonal — they are the *new* signals.

### A2. Upcoming-failure prediction (headline result)
**Question:** Does adding task-position features improve upcoming-failure prediction over the existing AUC 0.67–0.68 baseline from the writeup's Section 4.1?
**Method:** Rerun the existing failure probe with task-position features concatenated. Target AUC to beat: 0.68 (Gemma comparable).
**Success criterion:** A clean win is +0.05 AUC or more. Anything smaller is ambiguous and reported as such.

### A3. F90871 ↔ `tokens_until_boundary` correlation
**Question:** Is the universally-suppressed boundary detector F90871 the mechanistic correlate of `tokens_until_boundary`?
**Method:** Sliding-window correlation between F90871 activation (L20, already instrumented in existing SAE scripts) and the `tokens_until_boundary` probe readout across traces. Also: does the correlation decay as context fills?
**Prediction:** F90871 spikes when `tokens_until_boundary → 0` in early-context traces, but the spike dampens in late-context traces — quantifying the "boundary detection is suppressed" story in position-belief terms.

### A4. Calibration gap vs. probe readout (highest-value finding)
**Question:** Does the model's overconfidence grow with its *subjective* sense of task-lateness, independent of objective difficulty?
**Method:** Plot ECE (or Brier score) binned by `within_task_fraction` and `tokens_until_boundary`. Overlay raw-token-position ECE as a control.
**Prediction:** Calibration degrades with probe-subjective lateness more sharply than with raw position. If confirmed, this is the finding that sells the paper — it operationalizes "confidence outpaces accuracy" as a function of the model's internal state, not just context length.

### A5. Base vs. IT probe-strength comparison
**Question:** Is the "sense of lateness" installed by RLHF, or present in the base model?
**Method:** Train the same probes on Gemma-9B-base using the DDXPlus base-model traces. Compare R² and probe direction sharpness (norm of the ridge solution / effective rank).
**Prediction (from writeup Section 7.5):** IT has sharper, higher-R² probes than base. If confirmed, RLHF is specifically what installs the temporal self-model that fatigue corrupts.

### Analysis priority
- **MVP (must land in v1):** A1, A2, A4
- **Stretch (v1 if time permits):** A3, A5

## Code organization

Follows the project's CLAUDE.md conventions: BaseSchema for dataclasses, all imports at top of file, auto-export in `__init__.py`, no dead code, no backwards-compatibility shims.

### New module: `src/probes/task_position/`
- `labels.py` — pure-function labelers: `label_trace(trace, case_boundaries) -> TaskPositionLabels` (BaseSchema dataclass)
- `probes.py` — probe training and evaluation: linear ridge and MLP. One class per probe type, shared interface.
- `analysis.py` — the five analyses (A1–A5), each as a pure function consuming trained probes + raw activations.
- `__init__.py` — auto-exports all public symbols.

### Shared refactor (targeted, stays in scope)
If residual-stream extraction currently lives inline in `scripts/context_fatigue/run_ddxplus_probe.py` or similar, extract it into `src/probes/extraction.py` as a single reusable function. This is the only pre-existing code touched by v1, and the refactor is motivated by the new work, not speculative cleanup.

### Driver scripts
Thin entry points in `scripts/probes/task_position/`:
- `extract_activations.py` — one-shot extraction (skipped if cached artifacts exist)
- `train_probes.py` — fits all (target × layer × probe type) combinations
- `run_analyses.py` — runs A1–A5 and writes a results markdown

### Tests
- `tests/probes/task_position/test_labels.py` — unit tests for the labeler (trivial because labelers are pure functions on small arrays; no model required)
- Probe training and analyses are validated by the analyses themselves (baselines must be beaten); no additional unit tests unless a bug surfaces.

### Deliverables
1. Trained probe artifacts per `(target, layer, probe_type)` saved to `results/probes/task_position/gemma-9b-it/`
2. Results markdown at `results/probes/task_position/2026-04-13-v1-results.md` documenting A1–A5 with figures
3. Code changes and spec committed to the `context-fatigue-datasets` branch

## Explicitly out of scope for v1

- Steering vectors or any causal intervention
- Prompt-refresh controller experiments
- Qwen-7B, Llama-8B replication
- Heterogeneous mixed-task traces
- NarrativeQA single-task probes
- Subjective self-report probes (prompt the model "how far along are you?")
- Hyperparameter sweeps beyond ridge α on held-out split

## Success criteria for v1

v1 is a success if **all three** of these hold:

1. At least one of the three task-position probes beats both baselines (raw-token-position and context-fill-projection) on its primary metric, evaluated on the held-out test split, at the best layer, for at least one target. (Tests: *is there a signal at all?*)
2. A1 produces an interpretable orthogonality picture — whether positive or negative for the new-signal hypothesis. (Tests: *did we learn something about the structure?*)
3. A4 produces a plot that either supports or refutes the calibration-gap hypothesis with a clear visual story. (Tests: *do we have the headline figure?*)

v1 is a *strong* success if A2 additionally beats the 0.68 upcoming-failure AUC baseline by ≥0.05.

v1 is **not** required to land A3 and A5. If they land, the scope of the follow-up (v2 steering study) expands accordingly.
