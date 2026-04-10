# AGENTS.md

This file is a working guide for coding agents operating in this repository.

## 1. What This Repo Is

- Name: `temporal-awareness`
- Purpose: research code for detecting and steering temporal preference/temporal awareness in language models.
- Primary active stack today is the `src/intertemporal` + `src/inference` + `src/binary_choice` + `src/activation_patching` + `src/attribution_patching` path.
- There is also an older stack (`src/data`, `src/experiments`, `src/models`, many `scripts/experiments/*`) that appears partially stale.

## 2. High-Level Architecture

- `src/intertemporal/*`: current intertemporal pipeline.
- `src/inference/*`: model abstraction (`ModelRunner`) and backends (HuggingFace, Pyvene, MLX, TransformerLens, NNsight).
- `src/binary_choice/*`: preference-choice logic built on `ModelRunner`.
- `src/activation_patching/*`: causal activation patching types and routines.
- `src/attribution_patching/*`: attribution patching (standard EAP + IG variants).
- `src/common/*`: shared schemas, token position logic, file I/O, math/profiler utilities.
- `src/viz/*`: plotting and heatmap helpers.

Important orchestrator:
- `src/intertemporal/experiments/intertemporal_experiment.py`
  - Step order: preference data -> attribution patching -> activation patching -> visualization.

Primary CLIs (newer):
- `scripts/intertemporal/generate_prompt_dataset.py`
- `scripts/intertemporal/query_llm_preference.py`
- `scripts/intertemporal/run_intertemporal_experiment.py`
- `scripts/intertemporal/run_sae_pipeline.py`

## 3. Repository Layout (Operational)

- `data/raw/*`: source preference datasets and variants.
- `out/*`: runtime outputs for newer intertemporal pipeline (prompt datasets, preference datasets, experiment outputs).
- `results/*`: tracked historical outputs/checkpoints/figures.
- `scripts/*`: mix of old and new scripts.
- `tests/*`: mixed old and new test suites, with duplication.

## 4. Setup and Environment

- Python requirement: `>=3.10`.
- Install: `pip install -e .`.
- Key dependency: `latents` from GitHub (`pyproject.toml`), plus `torch`, `transformers`, `transformer_lens`, `scikit-learn`, etc.
- `.env.example` contains placeholders for:
  - `OPENAI_API_KEY`
  - `OPENROUTER_API_KEY`
  - `HF_TOKEN`
  - GCP vars (`GCP_PROJECT`, `GCP_ZONE`, `GCP_INSTANCE`, `GCS_BUCKET`)

Notes:
- Current shell environment may not have `python`; use `python3` or `uv run python ...`.
- Model-heavy scripts/tests require GPU or substantial local resources.

## 5. Canonical Commands

- Install:
  - `make install`
- Quick claim verification (mostly reported/cached values):
  - `make verify-quick`
  - `python scripts/verify_all_claims.py --quick`
- Generate prompt dataset (new stack):
  - `python scripts/intertemporal/generate_prompt_dataset.py --config cityhousing`
- Query model preferences (new stack):
  - `python scripts/intertemporal/query_llm_preference.py --config <config-or-path>`
- Run new intertemporal experiment:
  - `python scripts/intertemporal/run_intertemporal_experiment.py --full`
- Run SAE pipeline:
  - `python scripts/intertemporal/run_sae_pipeline.py --new`

## 6. Backends and Runtime Behavior

`ModelRunner` backend selection:
- Inference default: MLX on Apple Silicon if available, otherwise HuggingFace.
- Internals/interventions recommended: Pyvene (`get_recommended_backend_internals()` / `...interventions()`).

`ModelRunner` behavior to know:
- Applies chat template automatically for chat/instruct models.
- Adds `skip_thinking_prefix` (`<think>\n</think>\n\n`) for reasoning models in binary-choice paths.
- Supports cache capture, interventions, and intervention+cache+grad paths.

## 7. Data and File Naming Conventions

New prompt dataset pathing:
- Default output dir via `src/intertemporal/common/project_paths.py`: `out/prompt_datasets/`.
- Prompt filename pattern: `{name}_{dataset_id}.json`.

Preference dataset pathing:
- Default output dir: `out/preference_datasets/`.
- Preference filename pattern: `{prompt_dataset_id}_{model_name}[_{prompt_dataset_name}].json`.
- Internals tensors stored separately as `.pt` under `out/preference_datasets/internals/`.

## 8. Tests: What Is Reliable vs Legacy

Potentially active/new tests are mainly under:
- `tests/inference/*`
- `tests/attribution_patching/*`
- `tests/common/*`

Legacy/duplicated tests exist under:
- `tests/src/*` (mostly older API pathing)
- `tests/intertemporal/*` currently imports modules like `src.datasets`, `src.models`, `src.common.io` that do not appear to be part of the current stack.

Practical test guidance:
- Start with unit-ish tests that avoid model downloads.
- Use `--skip-slow` to avoid expensive tests.
- Expect many model/backend tests to be slow and environment-dependent.

## 9. Known Inconsistencies and Traps

Mixed old/new code paths are the biggest risk.

### 9.1 New path (prefer)
- Prefer `src/intertemporal/*`, `src/inference/*`, `src/binary_choice/*`.
- Prefer `scripts/intertemporal/*`.

### 9.2 Old path (treat as legacy unless explicitly needed)
- `src/data/*`, `src/experiments/*`, `src/models/*` and many scripts in `scripts/experiments`, `scripts/circuits`, `scripts/probes` reference old symbols such as:
  - `src.models.*`
  - `src.datasets.*`
  - `src.common.io`
- These references are inconsistent with current module layout (`src/inference`, `src/common/file_io.py`, etc.).

### 9.3 Specific code issues observed
- `src/intertemporal/preference/preference_dataset.py` `merge()` constructs `PreferenceSample` with `alt_prob=` and `choice_prob=` fields that are not dataclass fields in current `PreferenceSample`; this path is likely broken if executed.
- `src/inference/captured_internals.py` `InternalsConfig.get_names()` iterates `internals.activations` instead of `self.activations`; likely a bug.
- Some config JSONs include trailing commas or unusual ranges; `load_json()` intentionally strips trailing commas, so parsing may still succeed.

## 10. Coding Conventions in This Repo

- Many packages use `auto_export` in `__init__.py`:
  - Avoid hand-maintained `__all__` where auto-export is already used.
- Data classes inherit from `BaseSchema` for deterministic IDs and robust `from_dict` conversion.
- Token position logic is central for patching/probing; reuse helpers in `src/common/token_positions.py` and prompt-format-specific anchor logic.
- Notebook preference:
  - if a training / fitting / activation-extraction step takes noticeable time, keep plotting in a separate cell so results can be re-plotted without re-running the expensive computation.
- For notebook comparison line plots, prefer `marker='o'` for all series unless marker shape carries a necessary semantic distinction.

## 11. Working Rules for Agents

- Before editing, determine whether target file is in active or legacy stack.
- If asked to implement a new feature, implement in the active intertemporal stack unless user requests legacy compatibility.
- If fixing tests, prioritize `tests/inference`, `tests/common`, `tests/attribution_patching` first.
- Avoid sweeping refactors across old/new stacks in one change; changes are safer when scoped to one stack.
- For expensive workflows, prefer minimal configs (`MINIMAL_EXPERIMENT_CONFIG`) and low sample counts first.

## 12. Useful File Pointers

Core orchestrator and configs:
- `src/intertemporal/experiments/intertemporal_experiment.py`
- `src/intertemporal/data/default_configs.py`
- `src/intertemporal/common/project_paths.py`

Prompt and preference pipeline:
- `src/intertemporal/prompt/prompt_dataset_generator.py`
- `src/intertemporal/prompt/prompt_dataset.py`
- `src/intertemporal/preference/preference_querier.py`
- `src/intertemporal/preference/preference_dataset.py`

Inference and interventions:
- `src/inference/model_runner.py`
- `src/inference/backends/backend_selection.py`
- `src/inference/interventions/intervention.py`
- `src/inference/interventions/intervention_factory.py`

Patching:
- `src/intertemporal/experiments/activation_patching.py`
- `src/intertemporal/experiments/attribution_patching.py`
- `src/activation_patching/*`
- `src/attribution_patching/*`

## 13. Submodule

- Git submodule present:
  - `scripts/data/validation/llm_council` -> `https://github.com/AsyaPronina/llm-council.git` (branch `format_chairman_response`)
- If data validation workflows fail, confirm submodule init/update status.

## 14. MM Causal Steering Experiment Notes (Critical)

This section summarizes both the historical causal MM work in `notebooks/mmraz-exploration.ipynb` and the current preferred steering setup in `notebooks/mmraz-probe-steering-options-answer.ipynb`, with emphasis on pitfalls that can make results look broken.

### 14.1 Where the experiment currently lives

- Preferred current steering notebook:
  - `notebooks/mmraz-probe-steering-options-answer.ipynb`
  - Qwen steering benchmark using the current unified `Options: ... Answer:` probe format.
  - Keeps `STEERING_DATASET_SOURCE = "expanded"`; it still trains the steering directions only from the original expanded explicit dataset.
- Historical / legacy steering notebooks:
  - `notebooks/mmraz-probe-steering.ipynb`
  - `notebooks/mmraz-exploration.ipynb`
- The older notebooks are still useful for reproducing earlier numbers, but they use older prompt families and token targets.

### 14.2 Experiment design used

- Data split for causal benchmark is **pair-level**:
  - explicit pairs: 80/20 split (`train_test_split(..., random_state=42)`).
  - MM direction training uses only explicit-train pairs.
  - Evaluation datasets:
    - explicit test pairs (held-out from explicit),
    - full implicit pairs (or capped subset in quick mode).
- Current preferred prompt family for steering and probe-direction training:
  - prompt:
    - `Options:\n{option_a_text_stripped}\n{option_b_text_stripped}\nAnswer:\n`
  - continuation for teacher forcing:
    - stripped semantic answer text only,
    - no `(A)` / `(B)` tokens,
    - no free-form explanation text.
- Probe-direction training inside the steering copy creates two teacher-forced continuations per pair:
  - immediate / short-term answer,
  - long-term answer.
- Current preferred probe feature for steering directions:
  - mean hidden state across **all answer tokens** in the teacher-forced continuation,
  - not the last token of the continuation.
- Probe direction:
  - standard MM (difference-of-means) per layer: `mu_long_term - mu_immediate`.
  - direction is injected at the target layer on the **last prompt token** (`prompt_len - 1`), i.e. the final token of the `Answer:` prefix before answer generation starts.
- This is an intentional approximation, not exact token identity:
  - training labels live on the answer-token span,
  - steering must act on a controllable pre-answer position,
  - so the best practical alignment is:
    - train on the answer span,
    - steer on the final prompt token immediately before the answer.
- Conditions:
  - baseline (no intervention),
  - `add_mm` (positive direction),
  - `sub_mm` (negative direction).
- Strength sweep:
  - includes high values to test saturation/collapse behavior (`0, 2, 5, 10, 20, 40, 80, 160`).
- Historical notebooks used older prompt/feature choices:
  - `question + "\n\nChoices:\n" + one option`,
  - last-token activation of the whole sequence.
  - Keep these only for reproducing old results, not as the default design for new work.

### 14.3 Most important failure mode discovered

The initial collapse (`fallback_rate` near 1.0 on explicit, high on implicit) was caused by a **measurement pipeline issue**, not just steering weakness:

- Old method effectively relied on first next-token logits at `Answer:`.
- GPT-2 frequently emits newline/prose first (not `A`/`B`) for this prompt format.
- That caused frequent parser misses and fallback.
- Fallback used A/B token logits at the wrong decode step, creating strong apparent A bias.

This made results look like "always A" even when continuation text was variable.

### 14.4 Parsing + prompting findings that are easy to miss

- The parseability notes below are mainly about the **legacy free-generation** prompt families in `mmraz-exploration.ipynb` and the original `mmraz-probe-steering.ipynb`.
- The `options-answer` steering copy keeps the intervention point cleaner, but free-generation parsing still matters for steering evaluation outputs.
- Parsing from full generated continuation is required; first-token-only parsing is brittle.
- The exact answer suffix strongly changes parseability:
  - `Answer:` vs `Answer: ` vs `Answer: (` gives very different behavior.
  - In diagnostics, `Answer: (` dramatically increased parse coverage on sampled prompts.
- GPT-2 often outputs `1)` as first marker. Parser must map:
  - `1 -> A`, `2 -> B`.
- Even with better parsing and near-zero fallback in some prompt variants, GPT-2 still shows strong first-option/`A` bias on this task format.

### 14.5 Intervention behavior observed

- Steering hook is active (continuation text changes under large strengths).
- However, label flips (`A` <-> `B`) may not appear even at very high strengths in this setup.
- Large unnormalized vectors can cause off-distribution gibberish (`unknown`/repetition loops), so direction scaling/normalization matters.
- Practical implication: changed generations do not automatically imply changed binary-choice label.

### 14.6 Logging and artifacts to inspect

Do not trust aggregates without checking raw outputs.

- Main debug outputs saved under `results/`:
  - `mm_intervention_summary_debug.csv`
  - `mm_intervention_response_logs_debug.csv`
- Additional focused debug runs generated:
  - `results/mm_intervention_debug_log_tiny.csv`
  - `results/mm_intervention_debug_log_layer8.csv`
  - `results/mm_intervention_debug_summary_layer8.csv`

Recommended manual checks in logs:
- continuation text by `(dataset, layer, condition, strength, prompt_idx)`,
- parser method (`generation_parse` vs fallback),
- fallback rates by condition/strength,
- whether continuation changes without label changes.

### 14.7 Code design guidance for future work

- Keep causal eval code path explicit and testable:
  - separate prompt formatting, generation, parsing, and aggregation functions.
- Always log raw continuations; never only store parsed labels.
- Track parse method and fallback flags in final tables.
- Keep a `quick_mode` path for cheap iteration before full layer sweeps.
- Preserve deterministic split seeds for comparability (`random_state=42` used here).

### 14.8 Recommended next metric design

For robust causal effect estimation on this binary choice, prefer forced-choice scoring as a companion metric:

- Compare log-likelihood/logprob of continuations constrained to `A` vs `B` (or `(A)` vs `(B)`), instead of relying only on free-generation parsing.
- Use free-generation parsing for qualitative behavior, but use forced-choice likelihood for the primary quantitative causal metric.

## 15. Probe Accuracy Experiment Notes (LR / MM / Whitened LR / Whitened MM)

This section captures both the historical probe-accuracy setup from `notebooks/mmraz-exploration.ipynb` and the current preferred `options-answer` setup used in the newer probe notebooks.

### 15.1 Scope and purpose

- Goal: validate temporal separability in GPT-2 activations and compare four linear probe families:
  - LogisticRegression (LR),
  - mean-mass (MM, difference-of-means),
  - whitened LogisticRegression (WLR),
  - whitened MM (WMM).
- Also compares against saved per-layer LR checkpoints in:
  - `results/checkpoints/temporal_caa_layer_{layer}_probe.pkl`.

### 15.2 Data files and prompt construction

- Datasets:
  - explicit: `temporal_scope_caa.json` (CAA explicit temporal pairs),
  - implicit: `temporal_scope_implicit.json` (semantic/generalization set).
- Notebook path resolution checks multiple candidate locations and takes first existing.
- Historical / legacy format:
  - from each pair, create two prompts:
    - `question + "\\n\\nChoices:\\n" + immediate` -> label `0`,
    - `question + "\\n\\nChoices:\\n" + long_term` -> label `1`.
  - This format came from the original CAA scripts and is useful mainly for reproducing earlier numbers.
- Current preferred format in:
  - `notebooks/mmraz-probe-variations-red-team-20260322-090107.ipynb`
  - `notebooks/mmraz-probe-steering-options-answer.ipynb`
- Current preferred sample construction:
  - prompt:
    - `Options:\n{option_a_text_stripped}\n{option_b_text_stripped}\nAnswer:\n`
  - two teacher-forced continuations per pair:
    - stripped immediate / short-term answer -> label `0`
    - stripped long-term answer -> label `1`
  - `(A)` / `(B)` markers are stripped from both the prompt options and the continuations.
- When red-team examples are reused for probe training:
  - do **not** train on raw `prompt_text + completion_text`,
  - first run `scripts/mmraz_intertemporal/build_stripped_red_team_probe_dataset.py`,
  - then train from `out/mmraz_intertemporal/adversarial_red_teaming/runs/{run_id}/probe_dataset_stripped.jsonl`.
- Why the newer format is preferred:
  - the prompt now contains the full binary-choice context, not one option in isolation,
  - stripping `(A)` / `(B)` reduces syntax leakage,
  - stripping free-form justifications keeps the probe focused on the choice itself rather than explanation style.

### 15.3 Activation extraction details

- Model: GPT-2 (`AutoModelForCausalLM.from_pretrained("gpt2")`).
- Historical / legacy extraction:
  - last-token hidden state from every transformer block output:
    - `out.hidden_states[layer + 1][row_idx, last_idx, :]`
  - This means:
    - on legacy explicit prompts, the probe classified from the final token of the single option string,
    - on raw red-team prompt+completion strings, the probe could end up classifying the final token of a free-form justification.
- Current preferred extraction:
  - mean hidden state over **all answer tokens** in the teacher-forced continuation after `Answer:\n`
  - prompt tokens are excluded from this pooled feature
  - per-example pooled vector is then used for MM / WMM fitting and scoring.
- Classification rule for the current MM probes:
  - direction `d = mu_long_term - mu_short_term`
  - score `s(x) = x^T d`
  - predict class `1` iff `s(x) > 0`
- Why the current token target is preferred:
  - avoids dependence on the very last punctuation token,
  - reduces answer-length bias,
  - prevents the probe from keying on the tail of a justification,
  - produces a smoother direction that is more sensible to add continuously during steering.
- Extracted separately for explicit and implicit prompt sets:
  - `X_exp[layer]`, `X_imp[layer]`.

### 15.4 Train/test split and metric definitions

- Accuracy split is **prompt-level**, not pair-level:
  - `train_test_split(indices, test_size=0.2, random_state=42, stratify=y_exp)`.
- Metrics:
  - `explicit_test_acc`: held-out 20% prompt split from explicit prompts.
  - `implicit_acc`: full implicit prompt set, never used for fitting.
  - `saved_probe_explicit_full_acc`: saved probe on all explicit prompts.

Important caveat:
- Prompt-level split can place prompts from the same original pair on opposite sides of train/test.
- This is consistent with original probe scripts but is less strict than pair-level splitting.

### 15.5 Exact probe variants implemented

- Saved LR checkpoints:
  - loaded from `results/checkpoints/temporal_caa_layer_*.pkl`,
  - scored on explicit test/full and implicit.
- Retrained LR:
  - `LogisticRegression(max_iter=1000, random_state=42)` on raw `X_train`.
- MM probe:
  - direction `d = mu1 - mu0` from train activations,
  - score `s(x) = x^T d`,
  - predict class `1` iff `s(x) > 0`.
- WLR:
  - fit train-only whitener:
    - center by train mean,
    - covariance regularization: `Sigma_reg = Sigma + reg * avg_var * I`,
    - whitening transform uses `Sigma_reg^{-1/2}`.
  - train/infer LR on whitened activations.
- WMM:
  - same train-only centered covariance/regularization,
  - effective direction `d_eff = Sigma_reg^{-1} (mu1 - mu0)`,
  - score rule `(x - mean_train)^T d_eff`.

### 15.6 Whitening/numerics choices

- Default regularization used in notebook: `reg = 1e-2`.
- Uses pseudo-inverse and eigenvalue clipping (`>= 1e-12`) for stability.
- Tracks covariance condition number in outputs:
  - `wlr_cov_reg_condition_number`,
  - `wmm_cov_reg_condition_number`.

Why this matters:
- `d_model` is high relative to train samples, so raw covariance is singular/ill-conditioned.
- Without regularization, whitening-based probes are numerically unstable.

### 15.7 Additional diagnostics computed

- Geometry alignment per layer:
  - cosine between retrained LR weight vector and MM direction,
  - cosine between retrained LR weight vector and WMM effective direction.
- README claim check:
  - compares measured accuracies to claimed peaks (e.g., layer-8 explicit, layer-6 implicit).
- Dataset integrity check:
  - compares current dataset SHA-256 hashes against hashes stored in `results/probe_validation_results.json`.

### 15.8 What was surprising / easy to misread

- Hash mismatch can legitimately change numeric results even with identical code.
- "explicit test" in probe-accuracy section is prompt-level split, while causal section uses pair-level split; these are not interchangeable.
- Saved probes and retrained LR are close but not guaranteed identical due file/version/data drift.
- WLR/WMM improvements are not monotonic across layers; whitening helps some layers and hurts others.
- The old one-option `Choices:` format was mostly legacy compatibility, not a principled representation of the binary choice task.
- Raw red-team `prompt_text + completion_text` is a poor direct probe-training target because the final token often belongs to the explanation, not the decision.

### 15.9 Relevant source files for this experiment

- Historical notebook implementation:
  - `notebooks/mmraz-exploration.ipynb`
- Current preferred probe-format notebooks:
  - `notebooks/mmraz-probe-variations-question-options-answer.ipynb`
  - `notebooks/mmraz-qwen-probe-variations-question-options-answer-vast.ipynb`
  - `notebooks/mmraz-probe-variations-red-team-20260322-090107.ipynb`
  - `notebooks/mmraz-probe-steering-options-answer.ipynb`
- Original LR probe training script:
  - `scripts/probes/train_temporal_probes_caa.py`
- Red-team postprocessing script for probe training:
  - `scripts/mmraz_intertemporal/build_stripped_red_team_probe_dataset.py`
- Validation script with explicit->implicit protocol and reproducibility metadata:
  - `scripts/validate_probes_gcp.py`

### 15.10 Guidance for future contributors

- If the goal is strict generalization estimate, add a pair-level split variant for LR/MM/WLR/WMM and report both.
- Keep train-only covariance estimation for all whitened variants (avoid test leakage).
- Always record dataset hashes, model version, split seed, and probe hyperparameters with results tables.
- Default to the `Question + Options + Answer:` format, stripped option labels, and mean answer-token pooling for new MM probe work.
- When red-team examples are appended in the question-options-answer notebook, rebuild the prompt from `question_text`, `option_a_text`, and `option_b_text` when those fields are available instead of trusting an older `probe_prompt` string.
- Use the legacy one-option `Choices:` / last-token setup only when you explicitly need historical comparability.
- Preserve existing metric names to avoid confusion with past outputs:
  - `explicit_test_acc`, `implicit_acc`, `saved_probe_*`.

## 16. MM Probe Red-Teaming Notes (Critical)

This section captures the current adversarial red-teaming implementation for the temporal MM probe.

### 16.1 Where the red-teaming code lives

- Main implementation:
  - `src/mmraz_intertemporal/red_teaming.py`
- CLI entrypoint:
  - `scripts/mmraz_intertemporal/run_adversarial_probe_red_team.py`
- Analysis notebooks:
  - current loop: `notebooks/mmraz-red-teaming-run-analysis-full-history.ipynb`
  - legacy loop: `notebooks/mmraz-red-teaming-run-analysis.ipynb`

### 16.2 Current experimental setup

- The current implementation is a paper-faithful **multi-turn black-box** red-teaming loop.
- It keeps the attacker in a single conversation history and uses a separate LLM judge for ground-truth temporal labels.
- It does **not** mutate seed examples from `mmraz_probe_steering`.
- Target probe:
  - GPT-2 mean-mass (MM) probe,
  - layer `6`,
  - loaded from `RedTeamConfig.mm_probe_checkpoint_path`,
  - current default path is:
    - `results/checkpoints/mmraz_probe_variations_red_team_augmented_20260322-090107_options_answer_stripped_mean_answer_tokens/mmraz_gpt2_explicit_expanded_plus_redteam_options_answer_mm_probe_layer_6.json`
  - This is the notebook-trained **explicit expanded train split + prior red-team examples** checkpoint from the question-options-answer workflow.
  - The runtime attacks this fixed saved probe; it does **not** retrain the probe online during a run.
- Attacker model default:
  - `claude-sonnet-4-20250514` via Anthropic Messages API.
- Judge model default:
  - `claude-sonnet-4-20250514` via Anthropic Messages API.
- Attack surface:
  - prompt + completion jointly.
- Required attacker format:
  - `prompt_text` must use the question-options-answer template:
    - question or task on the first line,
    - `Options:`,
    - `(A) ...`,
    - `(B) ...`,
    - `Answer:`
  - one option must be semantically short-term,
  - the other must be semantically long-term,
  - `completion_text` must begin with `(A)` or `(B)`, restate the chosen option, and briefly justify it,
  - `intended_label` must be `short_term` or `long_term`.
- Important distinction:
  - the attacker and judge still operate on raw `prompt_text` / `completion_text` with visible `(A)` / `(B)` labels,
  - but the probe now scores a stripped teacher-forced representation derived from those fields:
    - `question_text`,
    - `option_a_text`,
    - `option_b_text`,
    - `chosen_option_letter`,
    - `probe_prompt`,
    - `probe_completion`.
  - In other words: raw labels stay in the human-readable artifacts, but the probe sees stripped option texts and a stripped chosen continuation.

### 16.3 How the iterative loop works

- Each round generates `k` fresh prompt+completion pairs as the next turn in the same attacker conversation.
- The attacker always sees the full previous batch in feedback; there is no separate feedback-example cap anymore.
- During the attack itself, the probe scores `probe_prompt` + `probe_completion`, not raw `prompt_text + completion_text`.
- For the current default checkpoint, this uses the notebook-style `mean_answer_token_activations` pooling over the chosen answer span.
- The judge then assigns the ground-truth temporal label for each candidate.
- Adversarial success is computed from `probe_label` vs `judge_label`, not just the attacker-intended label.
- The next attacker turn receives structured feedback from the previous round inside the same conversation history.
- The attacker does **not** see the actual MM probe vector or weights.
- The feedback shown back to the attacker includes:
  - `candidate_id`,
  - `attack_strategy`,
  - `intended_label`,
  - `probe_label`,
  - `judge_label`,
  - `probe_margin`,
  - `is_adversarial_success`,
  - judge reasons.

### 16.4 Where to edit `k` and `n`

- In code defaults:
  - `src/mmraz_intertemporal/red_teaming.py`
  - `RedTeamConfig.num_rounds` = `n`
  - `RedTeamConfig.candidates_per_round` = `k`
- Current defaults:
  - `num_rounds = 20`
  - `candidates_per_round = 10`
- In the CLI:
  - `--num-rounds` controls `n`
  - `--candidates-per-round` controls `k`
  - `--progress` / `--no-progress` controls live tqdm progress output
- There is no separate prompt-feedback-count knob anymore:
  - the attacker always sees the full previous batch.
- Example:
  - `python scripts/mmraz_intertemporal/run_adversarial_probe_red_team.py --num-rounds 30 --candidates-per-round 8`

### 16.5 Config and prompt entrypoints

- Main runtime config lives in:
  - `src/mmraz_intertemporal/red_teaming.py`
  - `RedTeamConfig`
- CLI surface lives in:
  - `scripts/mmraz_intertemporal/run_adversarial_probe_red_team.py`
- If you want to attack a different prior augmented probe checkpoint, change:
  - `--mm-probe-checkpoint-path`, or
  - `DEFAULT_MM_PROBE_CHECKPOINT_PATH` in `src/mmraz_intertemporal/red_teaming.py`
- Attacker prompt templates live in:
  - `_build_attacker_system_prompt()`
  - `_build_initial_attacker_user_prompt()`
  - `_build_feedback_user_prompt()`
- The actual prompt text used in a run is written to:
  - `attacker_system_prompt.txt`
  - `attacker_initial_user_prompt.txt`
  - `round_XXX_attacker_request_messages.json`
  - `round_XXX_feedback_to_attacker.txt`

### 16.6 Output artifacts and where to inspect runs

- Output root:
  - `out/mmraz_intertemporal/adversarial_red_teaming/runs/{run_id}/`
- Run-level files written at startup:
  - `run_config.json`
  - `target_probe.json`
  - `attacker_system_prompt.txt`
  - `attacker_initial_user_prompt.txt`
  - `judge_system_prompt.txt`
- Files written every completed round:
  - `round_XXX_attacker_request_messages.json`
  - `round_XXX_attacker_api_usage.json`
  - `round_XXX_attacker_response.txt`
  - `round_XXX_judge_api_usage.json`
  - `round_XXX_judge_system_prompt.txt`
  - `round_XXX_judge_user_prompt.txt`
  - `round_XXX_judge_raw_response.txt`
  - `round_XXX_judge_evaluations.json`
  - `round_XXX_feedback_to_attacker.txt`
  - `round_XXX_candidates.jsonl`
  - `round_XXX_summary.json`
- If a generation round fails after all retries:
  - `{stem}_generation_failure.json`
  - `{stem}_attempt_YY_raw_response.txt`
  - `{stem}_attempt_YY_api_usage.json`
  - `{stem}_system_prompt.txt`
  - `{stem}_request_messages.json`
- Completed-run aggregates:
  - `all_candidates.jsonl`
  - `successful_adversarial_examples.jsonl`
  - `attacker_api_usage.jsonl`
  - `judge_api_usage.jsonl`
  - `attacker_conversation.json`
  - `final_summary.json`
- Candidate rows now store both the raw attacker text and the probe-scored stripped view:
  - raw: `prompt_text`, `completion_text`
  - parsed: `question_text`, `option_a_text`, `option_b_text`, `chosen_option_letter`
  - probe input: `probe_prompt`, `probe_completion`
  - outputs: `probe_label`, `probe_margin`, `probe_confidence`, `probe_p_long_term`, `judge_label`, `is_adversarial_success`

Important interruption behavior:
- If a run dies mid-way (for example due to API credit exhaustion), the completed per-round files remain on disk.
- In interrupted runs, the aggregate end-of-run files may be missing even though the round files are valid.
- Partial-run analysis should therefore load `round_*_candidates.jsonl`, not rely only on `all_candidates.jsonl`.

### 16.7 Strategy-summary postmortem

- After a completed run, the attacker is now asked for a qualitative postmortem similar in spirit to the red-teaming paper.
- The final summary call asks for:
  - overall takeaways,
  - effective strategies,
  - ineffective strategies,
  - probe-heuristic hypotheses,
  - recommended next directions,
  - diversity gaps.
- Artifacts written for this step:
  - `attacker_strategy_summary.json`
  - `attacker_strategy_summary.md`
  - `attacker_strategy_summary_request_messages.json`
  - `attacker_strategy_summary_user_prompt.txt`
  - `attacker_strategy_summary_raw_response.txt`
  - `attacker_strategy_summary_api_usage.json`
- If that final summary call fails, the run still completes and writes:
  - `attacker_strategy_summary_error.json`

### 16.8 Probe metadata vs what the attacker sees

- `target_probe.json` stores probe metadata such as:
  - probe type,
  - model name,
  - layer,
  - dataset path,
  - train/test size,
  - train/test accuracy,
  - direction norm,
  - score scale,
  - probe format,
  - answer pooling.
- `target_probe.json` does **not** currently store the actual MM direction vector.
- The attacker is **not** directly given `target_probe.json`.
- The attacker only sees the prompt files written for each round plus prior scored feedback.

### 16.9 Analysis and review workflow

- The notebook `notebooks/mmraz-red-teaming-run-analysis.ipynb` is the main analysis entrypoint.
- It is designed to work on interrupted runs by reconstructing the dataset from `round_*_candidates.jsonl`.
- Main metrics in the notebook:
  - per-round probe failure rate,
  - cumulative probe failure rate,
  - failure rate split by intended label,
  - strongest successful adversarial examples.

### 16.10 Operational caveats

- Full runs require a valid `ANTHROPIC_API_KEY` and enough API credits to finish all rounds.
- The run now emits live tqdm progress by default, showing completed rounds and a postfix with saved candidates, successes, success rate, and cumulative attacker-request tokens.
- Because the run is cold-start, there are no `source_prompt_text` / `source_completion_text` fields in the current candidate schema.
- The current implementation is black-box with respect to probe parameters, but the attacker gets in-context learning through scored batch feedback.
- If you want to reuse a run for probe training, do not feed raw `prompt_text + completion_text` back into probe fitting. First build `probe_dataset_stripped.jsonl` with `scripts/mmraz_intertemporal/build_stripped_red_team_probe_dataset.py`.
- The stripped-dataset builder now writes `Question + Options + Answer` prompts with stripped option labels, matching the question-options-answer training notebook.
- Anthropic Messages API does not expose a direct "context left" meter in the response.
- The run therefore stores:
  - exact API usage per attacker call (`requested_max_output_tokens`, `input_tokens`, `output_tokens`, cache-token fields),
  - exact cumulative token totals used so far in the run,
  - an estimated remaining context token count based on the model context window.
- Retry policy for attacker truncation:
  - if an attempt fails and Anthropic reports `stop_reason = "max_tokens"`,
  - the next retry automatically doubles `max_output_tokens`,
  - capped by the estimated remaining context budget when available.
- Candidate parsing now has fallback recovery logic:
  - it first tries full JSON parsing,
  - then balanced-object extraction / cleanup,
  - then candidate-by-candidate salvage if the overall JSON wrapper is malformed.

## 17. Qwen3 4B Probe / Steering / Time-Utility Notes (Critical)

This section captures the current Qwen3 4B workflow across Vast, Google Colab, the standalone steering experiment, and the deterministic time-utility experiment.

### 17.1 Active notebooks and scripts

- Probe training / evaluation notebooks:
  - `notebooks/mmraz-qwen-probe-variations-question-options-answer-vast.ipynb`
  - `notebooks/mmraz-qwen-probe-variations-question-options-answer-colab.ipynb`
- Standalone probe-steering notebooks:
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-vast.ipynb`
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-colab.ipynb`
- Standalone steering helper script:
  - `scripts/vast/run_mmraz_qwen3_probe_artifact_steering.py`
- Deterministic time-utility notebooks:
  - `notebooks/mmraz-time-utility-experiment-vast.ipynb`
  - `notebooks/mmraz-time-utility-experiment-vast-2.ipynb`
  - `notebooks/mmraz-time-utility-experiment-colab.ipynb`
- Replot notebooks:
  - `notebooks/mmraz-qwen3-probe-variations-question-options-answer-vast-plots.ipynb`
  - `notebooks/mmraz-qwen3-probe-variations-question-only-vast-plots.ipynb`
  - `notebooks/mmraz-time-utility-experiment-vast-plots.ipynb`
  - `notebooks/mmraz-time-utility-experiment-colab-plots.ipynb`
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-plots.ipynb`
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-colab-plots.ipynb`
- Probe-vector analysis notebook:
  - `notebooks/mmraz-qwen3-probe-vector-logit-lens.ipynb`

### 17.2 Shared model and prompt settings

- The active model for this pipeline is:
  - `Qwen/Qwen3-4B`
- The older Qwen2.5 14B variant is no longer the active pipeline:
  - it was dropped because the single-GPU `24 GB` class setup hit OOM in practice
  - the current notebooks are aligned around Qwen3 4B instead
- All three stages must use the same prompt-formatting convention:
  - probe training
  - standalone steering
  - time-utility steering
- Current target prompt family for new work is the probe-style:
  - `Question + Options + Answer`
- Current alignment status:
  - `notebooks/mmraz-qwen-probe-variations-question-options-answer-vast.ipynb` uses the stripped `Question + Options + Answer:` prompt family
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-vast.ipynb` is aligned to the same family
  - `notebooks/mmraz-time-utility-experiment-vast.ipynb` was updated to match that family
  - `notebooks/mmraz-time-utility-experiment-vast-2.ipynb` and `notebooks/mmraz-time-utility-experiment-colab.ipynb` still use the older direct question prompt
- Do not silently compare or merge time-utility results across those prompt families as if they were directly interchangeable.
- Shared prompt-formatting rules:
  - chat template enabled
  - thinking trace explicitly disabled
- Thinking suppression implementation:
  - first try `tokenizer.apply_chat_template(..., enable_thinking=False)`
  - if that tokenizer call does not support the flag, fall back to appending:
    - `<think>\n</think>\n\n`
- This consistency matters:
  - changing chat templating or thinking-prefix handling changes the token sequence
  - probe vectors trained under one prompt encoding should not be reused under another

### 17.3 Pair-level explicit split semantics

- The explicit split is now strict pair/question-level, not prompt/example-level.
- Desired and current behavior:
  - start from `500` explicit questions / pairs
  - split `80/20` at the question level with `random_state=42`
  - `400` train pairs
  - `100` held-out test pairs
- Probe training uses both teacher-forced continuations from the `400` train pairs:
  - `800` explicit-train teacher-forced examples total
- Probe validation uses both continuations from the `100` held-out pairs:
  - `200` explicit-test teacher-forced examples total
- Standalone steering evaluation on `explicit_test` uses the same held-out `100` pair indices from the artifact:
  - it does not resplit the explicit dataset independently anymore
- This is important because the old prompt-level split could place:
  - the short-term continuation of one question in train
  - and the long-term continuation of the same question in test
- That older behavior is now considered incorrect for this Qwen3 pipeline.

### 17.4 Probe training notebook design

- Prompt family:
  - `Question + Options + Answer`
  - exact stripped prompt:
    - `question_text`
    - `Options:`
    - stripped option A text
    - stripped option B text
    - `Answer:`
- Teacher-forced continuations:
  - stripped immediate / short-term answer -> label `0`
  - stripped long-term answer -> label `1`
- Layers trained and evaluated:
  - `10, 14, 18, 22, 26`
- Feature families trained and saved:
  - `last_answer_token`
  - `mean_answer_tokens`
- Important terminology:
  - "average-token probe" in this workflow means `mean_answer_tokens`
  - this is the mean over the answer-token span only
  - it is not the mean over the whole prompt
- Probe families trained:
  - LR
  - WLR
  - MM
  - WMM
- Training regimes stored in the same artifact bundle:
  - `explicit_train_only`
  - `explicit_train_plus_redteam_20260322-090107`
  - `explicit_train_plus_redteam_20260322-090107_plus_20260403-183445`
- Red-team augmentation input preference:
  - prefer `probe_dataset_stripped.jsonl`
  - fallback to `all_candidates.jsonl` only if the stripped file is missing
- Do not retrain probes from raw:
  - `prompt_text + completion_text`

### 17.5 Probe artifact schema and compatibility rules

- Current artifact format version:
  - `artifact_format_version = 4`
- The `.npz` bundle stores all trained probes across:
  - train regime
  - feature family
  - layer
- Important array keys include:
  - `train_regimes`
  - `train_regime_labels`
  - `feature_names`
  - `layers`
  - `lr_coef`
  - `lr_intercept`
  - `wlr_effective_coef`
  - `wlr_effective_intercept`
  - `mm_raw_directions`
  - `mm_steering_vectors`
  - `wmm_effective_directions`
  - `wmm_steering_vectors`
  - `wmm_mean_train`
- Important metadata fields now required by downstream consumers:
  - `artifact_format_version >= 4`
  - `explicit_split_granularity = "pair"`
  - `probe_prompt_use_chat_template = true`
  - `probe_prompt_disable_thinking_trace = true`
  - `explicit_train_pair_indices`
  - `explicit_test_pair_indices`
- Downstream loaders in the steering and time-utility notebooks reject older artifacts that are:
  - example-level split
  - pre-chat-template
  - thinking-enabled
  - older than format version `4`
- Recommended downstream steering metadata stored intentionally in the artifact:
  - `recommended_time_utility_train_regime = "explicit_train_only"`
  - `recommended_time_utility_feature_name = "mean_answer_tokens"`
  - `recommended_time_utility_vector_key = "mm_steering_vectors"`
  - `recommended_time_utility_layer = 22`

### 17.6 Probe artifact output roots and filenames

- Vast training output root:
  - `results/qwen_question_options_answer_probe_variations_vast/{run_id}/`
- Colab training output root:
  - `/content/drive/MyDrive/spar-ai-temporal-activations-experiments-results/qwen_question_options_answer_probe_variations_colab/{run_id}/`
- Current filename prefix is Qwen3-specific:
  - `qwen3_4b_question_options_answer_probe_*`
- Main files written per run:
  - metrics CSV
  - summary CSV
  - probe artifact `.npz`
  - metadata JSON
  - figure index CSV
  - plot PNGs
- The Vast replot notebook only accepts the corrected artifact format and should ignore the older bad runs.

### 17.7 Standalone probe-steering experiment design

- The standalone steering experiment now loads probe vectors from the Qwen3 probe-training artifacts.
- Default steering slice:
  - train regime: `explicit_train_only`
  - feature family: `mean_answer_tokens`
  - vector key: `mm_steering_vectors`
- Datasets evaluated:
  - `explicit_test`
  - `implicit_full`
- Explicit-test evaluation must reuse:
  - `explicit_test_pair_indices` from the artifact metadata
- Steering layers:
  - `10, 14, 18, 22, 26`
- Steering strengths:
  - `±2, ±4, ±8, ±16, ±32`
- Prompt family matches training exactly:
  - `question`
  - `Options:`
  - stripped option A text
  - stripped option B text
  - `Answer:`
- Decoding:
  - `do_sample = False`
- Monitoring:
  - includes `tqdm`
- The run stores full model responses in the logs table.
- Important summary/log metrics include:
  - `prop_choose_long_term`
  - `prop_choose_immediate`
  - `fallback_rate`
  - `direct_parse_rate`
  - `mean_long_minus_immediate_avg_logprob`
  - `delta_long_minus_immediate_avg_logprob_vs_baseline`
  - `prop_logprob_prefers_long_term`
- Output roots:
  - Vast:
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_vast/{run_id}/`
  - Colab:
    - `/content/drive/MyDrive/spar-ai-temporal-activations-experiments-results/qwen3_4b_probe_artifact_steering_question_options_answer_colab/{run_id}/`
- Main files written:
  - summary CSV
  - logs CSV
  - probe-slice CSV
  - config JSON
  - artifact-index CSV
  - per-dataset heatmap PNGs
- Interrupted / partial runs still remain usable:
  - the Colab/Vast steering code writes partial per-condition files under `partial/`
  - the plotting notebook is designed to merge the top-level summary with `partial/*_summary.csv`
- Vast resume / reuse behavior:
  - `scripts/vast/run_mmraz_qwen3_probe_artifact_steering.py` now defaults to:
    - `reuse_existing_results = True`
    - `reuse_result_search_roots = None`
  - with reuse enabled, the runner scans both:
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_vast/`
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_colab/`
    - plus broader `results/` and `/workspace*` fallbacks
  - reusable points are matched by:
    - dataset
    - condition
    - layer
    - strength
    - and exact `prompt_idx/question/immediate/long_term/prompt` content checks
  - this allows Colab-produced steering artifacts to be reused by the Vast notebook when the held-out split and prompt strings match
  - reused points are not recomputed, but the runner still writes a fresh run directory containing:
    - summary CSV
    - logs CSV
    - partial per-point CSVs
    - `reuse_coverage_csv`
  - reused rows are marked with:
    - `result_source = "reused_existing"`
    - `reused_from_path`
  - newly computed rows are marked with:
    - `result_source = "computed_now"`
  - if all planned points are reusable, the runner skips model loading entirely
- Important metric caveat for standalone steering:
  - `prop_choose_long_term` / `prop_choose_immediate` come from free generation plus parsing
  - `mean_long_minus_immediate_avg_logprob` and `delta_long_minus_immediate_avg_logprob_vs_baseline` come from teacher-forced scoring of the canonical immediate vs long-term continuations
  - these are not currently measured under identical steering semantics:
    - free generation uses `patch_decode_tokens = cfg['patch_generation_tokens']`
    - candidate-logprob scoring currently calls the hook with `patch_decode_tokens = False`
  - so the free-generation choice rate and the logprob-margin metric can diverge substantially
  - especially on `implicit_full`, negative signed steering can lower `prop_choose_long_term` while the raw `delta_long_minus_immediate_avg_logprob_vs_baseline` still increases
  - interpret the logprob heatmap as a separate diagnostic, not a guaranteed monotone proxy for parsed generation choice

### 17.8 Deterministic time-utility notebook design

- Model:
  - `Qwen/Qwen3-4B`
- Current active Vast notebook:
  - `notebooks/mmraz-time-utility-experiment-vast.ipynb`
- Current active Vast prompt template is exact and probe-aligned:
  - `What would you prefer?`
  - `Options:`
  - `  {x}$ now`
  - `  {y}$ in {t}`
  - `  Answer:`
- This current `mmraz-time-utility-experiment-vast.ipynb` prompt does **not** contain:
  - `Answer in just a few words`
  - `few words only`
  - or any equivalent brevity instruction
- The source-of-truth prompt lives in the notebook source cell that defines:
  - `build_option_texts(...)`
  - `build_prompt(...)`
- The saved config metadata in that notebook should stay synchronized with the code:
  - `prompt_format = "question_options_answer"`
  - `prompt_template = "What would you prefer?\\nOptions:\\n  {x}$ now\\n  {y}$ in {t}\\n  Answer:\\n"`
- Current source-level steering sweep in `notebooks/mmraz-time-utility-experiment-vast.ipynb` is:
  - `STEERING_STRENGTHS = [16.0, 32.0, 64.0]`
- Older time-utility variants still use the legacy direct-question prompt:
  - `notebooks/mmraz-time-utility-experiment-vast-2.ipynb`
  - `notebooks/mmraz-time-utility-experiment-colab.ipynb`
  - legacy prompt:
    - `What would you prefer {x}$ now or {y}$ in {t}? Answer in just a few words.`
- Important notebook trap:
  - stored output cells inside `mmraz-time-utility-experiment-vast.ipynb` may still show old prompt strings until the notebook is rerun
  - when checking the live prompt format, inspect the source cell definitions rather than trusting rendered output tables
- Decoding:
  - `do_sample = False`
  - `max_new_tokens = 128` when thinking is disabled
  - `max_new_tokens = 2048` when thinking is enabled
  - one repeat per `(x, y, t)` point
- Thinking-mode support in the current Vast notebook:
  - `ENABLE_THINKING = False` => `without_thinking`
  - `ENABLE_THINKING = True` => `with_thinking`
  - when thinking is enabled, parsing strips the `<think>...</think>` block before scoring the final answer
- The notebook intentionally plots only:
  - `choose x = 100 now`
  - `unparsed rate`
- The time-utility heatmaps should render with:
  - lowest `y` at the top
  - highest `y` at the bottom
  - this is enforced via `origin='upper'`

### 17.9 Time-utility steering handoff and output roots

- The time-utility steering section consumes the Qwen3 probe-training artifact bundle, not the old Qwen2.5 steering-artifact format.
- Default steering selection:
  - train regime: `explicit_train_only`
  - feature family: `mean_answer_tokens`
  - vector key: `mm_steering_vectors`
  - layer: `22`
- Vast notebook:
  - baseline first
  - source notebook currently declares a steered rerun with:
    - `+16` as `steer_long_term_plus16`
    - `-16` as `steer_immediate_minus16`
    - `+32` as `steer_long_term_plus32`
    - `-32` as `steer_immediate_minus32`
    - `+64` as `steer_long_term_plus64`
    - `-64` as `steer_immediate_minus64`
- Important workspace-state caveat as of `2026-04-09`:
  - the latest saved `without_thinking` steered config under:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_steered_probe_variations_without_thinking/`
  - currently records only:
    - `steer_long_term_plus16`
    - `steer_immediate_minus16`
    - `steer_long_term_plus32`
    - `steer_immediate_minus32`
  - do not assume the saved outputs fully reflect the current source notebook sweep without checking the output `config.json`
- Colab notebook:
  - baseline first
  - then a steered rerun with:
    - `+16`
    - `-16`
  - plus appended stronger-sweep cells with:
    - `+32`
    - `-32`
    - `+64`
    - `-64`
- Vast output roots:
  - baseline:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_without_thinking/`
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_with_thinking/`
  - steered:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_steered_probe_variations_without_thinking/`
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_steered_probe_variations_with_thinking/`
- Colab output roots:
  - baseline:
    - `results/time_utility_experiment_qwen3_4b_colab_deterministic/`
  - `±16` steering:
    - `results/time_utility_experiment_qwen3_4b_colab_deterministic_steered_probe_variations/`
  - `±32 / ±64` stronger sweep:
    - `results/time_utility_experiment_qwen3_4b_colab_deterministic_steered_probe_variations_stronger/`
- Overwrite semantics are important:
  - these Vast time-utility output roots are mode-specific but not run-id-scoped
  - rerunning `without_thinking` baseline or steering writes back into the same CSV / JSON / PNG filenames
  - if you need to preserve an old result bundle before rerunning, copy the whole directory to a new name first
- Current workspace state as of `2026-04-09`:
  - `without_thinking` baseline files in:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_without_thinking/`
  - were rewritten on `2026-04-08`
  - `without_thinking` steered CSV / raw / config in:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_steered_probe_variations_without_thinking/`
  - were rewritten on `2026-04-08`
  - `with_thinking` baseline and steered files remained from `2026-04-06`
  - therefore the current local Vast result set mixes:
    - newer prompt-aligned `without_thinking` outputs
    - older `with_thinking` outputs
- Recovery caveat:
  - the time-utility result files under these `results/` directories are currently untracked in git
  - if a CSV / raw / config file is overwritten in place, git cannot restore the previous version
  - recovery is only possible from an external copy such as:
    - Time Machine / local snapshots
    - the original Vast workspace
    - rsynced backups
    - manually copied archives

### 17.10 Runtime and environment caveats

- Do not run the Qwen3 probe-training notebook on CPU unless there is no alternative:
  - the runtime is many hours to days
  - Colab runs should be interrupted if the notebook reports `Qwen device: cpu`
- For Colab, the main cause of `Qwen device: cpu` is:
  - no usable GPU runtime attached
  - not the missing `HF_TOKEN`
- The current Colab training notebook expects uploaded data roughly in this layout:
  - `MyDrive/spar-ai/data.zip`
  - March red-team run:
    - `.../20260322-090107/probe_dataset_stripped.jsonl`
  - April red-team run:
    - `.../20260403-183445/all_candidates.jsonl` is sufficient
- Vast / CUDA caveats:
  - do not assume `cuda:0` is the correct target device
  - move inputs to the actual model parameter device
  - if CUDA errors poison the session, restart the kernel
- The earlier 14B attempt on a `24 GB` class GPU failed due to OOM:
  - that is why the current aligned pipeline is Qwen3 4B

### 17.11 Local plotting notebooks for saved steering / time-utility outputs

- The local Vast time-utility replot notebook:
  - `notebooks/mmraz-time-utility-experiment-vast-plots.ipynb`
  - reads fixed mode-specific summary and raw CSV paths under:
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_{with,without}_thinking/`
    - `results/time_utility_experiment_qwen3_4b_vast_deterministic_steered_probe_variations_{with,without}_thinking/`
  - for the current mode-specific roots, you usually do **not** need to edit the notebook paths to see the latest results
  - rerun the plots notebook after regenerating those CSVs
  - because the plot notebook reads fixed paths rather than run ids, it will always show whatever files currently sit at those paths
  - this means it can silently combine outputs from different rerun dates or prompt variants if the directories were overwritten in place
  - if you need clean like-for-like comparisons across prompt families or rerun dates, write results to separate directories first
  - current `CONDITION_ORDER` in that notebook intentionally emphasizes:
    - baseline
    - `±16`
    - `±32`
  - the `±64` entries are currently commented out there to avoid over-emphasizing stale or mismatched saved artifacts
  - the notebook writes combined replot PNGs under:
    - `results/time_utility_experiment_qwen3_4b_vast_replots_both_thinking_modes/`

- The local time-utility replot notebook:
  - `notebooks/mmraz-time-utility-experiment-colab-plots.ipynb`
  - reads extracted local results under:
    - `results/time_utility_experiment_qwen3_4b_colab_deterministic*`
  - replots combined baseline / `±16` / `±32` / `±64`
  - plots only:
    - `choose_x_now`
    - `unparsed_rate`
  - uses the same low-to-high top-to-bottom `y` orientation as the experiment notebook
- The canonical local standalone-steering replot notebook is now:
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-plots.ipynb`
  - reads both:
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_colab/{run_id}/`
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_vast/{run_id}/`
  - merges each run's top-level summary with `partial/*_summary.csv`
  - deduplicates overlapping points while preferring the newest run
  - current combined plot output root:
    - `results/qwen3_4b_probe_artifact_steering_question_options_answer_combined_plots/`
  - shows a coverage table over observed `(signed_strength, layer)` points
  - heatmap format intentionally matches the earlier Qwen2.5 steering style:
    - x-axis = layer
    - y-axis = signed strength
    - baseline injected as the `0` row
    - `coolwarm` colormap
  - current plotted metrics:
    - `prop_choose_long_term`
    - `delta_long_minus_immediate_avg_logprob_vs_baseline`
    - `fallback_rate`
  - `steering_success` is intentionally not plotted anymore
- The older notebook:
  - `notebooks/mmraz-qwen3-probe-artifact-steering-question-options-answer-colab-plots.ipynb`
  - should be treated as a Colab-only legacy plotting copy, not the canonical one for new work

### 17.12 Qwen3 Probe-Vector Logit Lens Notebook

- Notebook:
  - `notebooks/mmraz-qwen3-probe-vector-logit-lens.ipynb`
- Purpose:
  - compute a simple logit-lens style projection for saved Qwen3 probe vectors:
    - if the probe / steering vector is `e`, compute `e^T W_U`
    - where `W_U` is the Qwen3 unembedding matrix
- The notebook is designed to avoid loading the full Qwen3 model:
  - it only downloads / reads:
    - tokenizer files
    - the single HF weight shard containing `lm_head.weight` or the tied embedding matrix
- Default artifact selection:
  - latest compatible Qwen3 probe artifact bundle
  - usually the recommended slice:
    - `explicit_train_only`
    - `mean_answer_tokens`
- Default vector families shown:
  - `mm_steering_vectors`
  - `wmm_steering_vectors`
- Current display / CSV simplification:
  - only:
    - `rank`
    - `token_id`
    - `decoded_stripped`
    - `score`
- Token filtering / dedup behavior:
  - filter to ASCII, English-looking decoded tokens only
  - strip whitespace
  - lowercase before deduplication
  - top and bottom tables show top/bottom `10` unique `decoded_stripped` entries
  - this means tokens such as `long` and `LONG` collapse to one displayed item
- Vocab-size caveat:
  - tokenizer vocab size and model unembedding vocab size may differ
  - the notebook correctly treats `W_U.shape[1]` as the canonical dimension when constructing token masks

### 17.13 Question-Only Probe Training Variant on Vast

- New script entrypoint:
  - `scripts/vast/run_mmraz_qwen3_probe_variations_question_only_vast.py`
- This is currently script-first (no dedicated question-only training notebook is required to run it).
- Purpose:
  - train probes with a **question-only** prompt
  - model sees only the question in the prompt
  - immediate / long-term answers are teacher-forced as continuations
  - `(A)` / `(B)` labels are stripped in probe training mode
- Prompt family metadata written by this script:
  - `prompt_family = "question_only_teacher_forced_answers"`
  - `prompt_format_description = "question-only prompt; no options shown to model; immediate/long-term continuations teacher-forced with A/B stripped"`
- Default output root:
  - `results/qwen_question_only_probe_variations_vast/{run_id}/`
- Main output filename prefix:
  - `qwen3_4b_question_only_probe_*`

- How to run:
  - `python3 scripts/vast/run_mmraz_qwen3_probe_variations_question_only_vast.py`
  - optional config override file:
    - `python3 scripts/vast/run_mmraz_qwen3_probe_variations_question_only_vast.py --config-path <path/to/config.json>`

- Progress/monitoring behavior:
  - activation extraction uses `tqdm` batch progress bars
  - progress descriptions are dataset-scoped (for example: explicit, implicit, red-team runs)

- Explicit-only checkpoint behavior (important for long runs):
  - the script trains and saves `explicit_train_only` probes first
  - explicit-only stage artifacts are saved with suffix:
    - `_explicit_only_checkpoint`
  - after save, the script prints an explicit completion notification:
    - `Explicit-only checkpoint is ready: <artifact_path>`
  - then it continues with red-team-augmented regimes and writes final all-regime artifacts

- Run metadata convenience:
  - the script writes:
    - `qwen3_4b_question_only_probe_run_config_{run_id}.json`
  - this file records both:
    - `explicit_only_checkpoint_artifact`
    - `final_artifact`

- Notebook parity note:
  - `notebooks/mmraz-qwen-probe-variations-question-options-answer-vast.ipynb` has `tqdm`-style progress for extraction/training flow
  - but it does **not** have the same dedicated `Explicit-only checkpoint is ready: ...` completion print used in the new question-only script

- Question-only replot notebook:
  - `notebooks/mmraz-qwen3-probe-variations-question-only-vast-plots.ipynb`
  - mirrors the analysis/plot structure of:
    - `notebooks/mmraz-qwen3-probe-variations-question-options-answer-vast-plots.ipynb`
  - reads from:
    - `results/qwen_question_only_probe_variations_vast/{run_id}/`
  - artifact compatibility checks include:
    - `artifact_format_version >= 4`
    - `explicit_split_granularity = "pair"`
    - chat-template enabled
    - thinking-trace disabled
    - `prompt_family = "question_only_teacher_forced_answers"`
  - artifact selection behavior:
    - prefers final all-regime files (`qwen3_4b_question_only_probe_*_{run_id}`)
    - falls back to explicit-only checkpoint files (`..._explicit_only_checkpoint`) when final files are not present yet
  - output behavior:
    - writes replots under `replots_from_notebook/` in the selected run dir
    - writes index CSV:
      - `qwen3_4b_question_only_probe_replots.csv`
  - optional override:
    - set `RUN_DIR_OVERRIDE` in notebook globals to force a specific run directory

## 18. Qwen3 32B Question-Only / Steering / Time-Utility Notes (Critical)

This section captures the current Qwen3 32B workflow under the dedicated `qwen3_32b/` folders.

### 18.1 Active 32B files and folders

- Dedicated folders:
  - `scripts/qwen3_32b/`
  - `notebooks/qwen3_32b/`
  - `results/qwen3_32b/`
- Main 32B scripts:
  - question-only probe training:
    - `scripts/qwen3_32b/run_qwen3_32b_probe_variations_question_only.py`
  - standard steering on `Question + Options + Answer`:
    - `scripts/qwen3_32b/run_qwen3_32b_probe_artifact_steering_question_options_answer.py`
  - direct-prompt time-utility steering:
    - `scripts/qwen3_32b/run_qwen3_32b_time_utility_probe_steering.py`
  - red-teaming:
    - `scripts/qwen3_32b/run_qwen3_32b_probe_red_team.py`
- Main 32B plotting notebooks:
  - probe training plots:
    - `notebooks/qwen3_32b/mmraz-qwen3-32b-probe-variations-question-only-plots.ipynb`
  - standard steering plots:
    - `notebooks/qwen3_32b/mmraz-qwen3-32b-probe-artifact-steering-question-options-answer-plots.ipynb`
  - time-utility steering plots:
    - `notebooks/qwen3_32b/mmraz-qwen3-32b-time-utility-probe-steering-plots.ipynb`

### 18.2 32B question-only probe defaults

- Model:
  - `Qwen/Qwen3-32B`
- Probe training prompt family:
  - `question_only_teacher_forced_answers`
- Training split granularity:
  - explicit: question-level `80/20`
  - implicit: question-level `70/30`
- Layers trained:
  - `24, 28, 32, 36, 40, 44, 48`
- Main steering-relevant default slice:
  - `train_regime = "explicit_train_only"`
  - `feature_name = "mean_answer_tokens"`
  - `vector_key = "mm_probe_vectors"`
- Output root:
  - `results/qwen3_32b/question_only_probe_variations/{run_id}/`

### 18.3 Standard 32B steering on `Question + Options + Answer`

- Script:
  - `scripts/qwen3_32b/run_qwen3_32b_probe_artifact_steering_question_options_answer.py`
- Default probe slice:
  - `explicit_train_only`
  - `mean_answer_tokens`
  - `mm_probe_vectors`
- Steering layers:
  - `24, 28, 32, 36, 40, 44, 48`
- Current steering strengths:
  - `±2, ±4, ±8, ±16, ±32, ±64, ±128`
  - plus baseline
- Output root:
  - `results/qwen3_32b/probe_artifact_steering_question_options_answer_vast/{run_id}/`

- Reuse / rerun behavior:
  - this script has **cross-run reuse**, not only same-run resume
  - it scans prior:
    - top-level logs CSVs
    - `partial/*_logs.csv`
  - search roots include the current output dir plus:
    - `results/qwen3_32b/probe_artifact_steering_question_options_answer_vast/`
    - `results/qwen3_32b/probe_artifact_steering_question_options_answer_colab/`
    - broader `results/` and `/workspace*` fallbacks
  - reusable points are matched by:
    - `dataset`
    - `condition`
    - `layer`
    - `strength`
    - and exact prompt-content checks over:
      - `prompt_idx`
      - `question`
      - `immediate`
      - `long_term`
      - `prompt`
  - reused rows are marked with:
    - `result_source = "reused_existing"`
    - `reused_from_path`
  - newly computed rows are marked with:
    - `result_source = "computed_now"`
  - if all planned points are reusable, the script skips model load entirely
  - you can also pass:
    - `--run-id <existing_run_id>`
    - to write the fresh checkpointed outputs back into a chosen run directory while still using the same reuse logic

### 18.4 32B direct-prompt time-utility steering

- Script:
  - `scripts/qwen3_32b/run_qwen3_32b_time_utility_probe_steering.py`
- This is the direct-question version, not `Question + Options + Answer`.
- Current prompt family:
  - default:
    - `What would you prefer {x}$ now or {y}$ in {t}? Answer in just a few words.`
  - special case for `tomorrow`:
    - `What would you prefer {x}$ now or {y}$ tomorrow? Answer in just a few words.`
- Current `t_values` include:
  - `1 hour`
  - `8 hours`
  - `tomorrow`
  - `week`
  - `month`
  - `year`
  - `2 years`
  - `3 years`
  - `5 years`
  - `10 years`
- Default steering layer:
  - `40`
- Default probe slice:
  - `explicit_train_only`
  - `mean_answer_tokens`
  - `mm_probe_vectors`
- Built-in conditions now include:
  - `baseline` with signed strength `0`
  - steered conditions from the configured signed-strength list
- Output root:
  - `results/qwen3_32b/time_utility_experiment_probe_steered/{run_id}/`

- Reuse / rerun behavior:
  - this script currently has **same-run resume**, not broad cross-run reuse
  - it only looks inside the selected `run_id` directory for cached results
  - reusable sources are:
    - `partial/<condition>_raw.csv`
    - the run's combined raw CSV
  - reuse is condition-level:
    - each condition must match the exact expected prompt grid for the current:
      - `x_now`
      - `y_values`
      - `t_values`
      - `n_repeats`
      - prompt strings
  - if a condition matches, it is not regenerated
  - if all conditions match, the script skips model loading entirely
  - important implication:
    - rerunning **without** `--run-id` creates a new timestamped run directory and therefore recomputes everything
    - rerunning **with** `--run-id <old_run_id>` is required to continue an earlier run without recomputing valid cached conditions
  - this matters when the configured strength sweep is expanded later:
    - for example, after adding new signed strengths such as `±64` or `±128`, rerun with the old `run_id` so only the missing conditions are generated

### 18.5 32B plotting notebook run selection

- The standard steering plots notebook:
  - `notebooks/qwen3_32b/mmraz-qwen3-32b-probe-artifact-steering-question-options-answer-plots.ipynb`
- The time-utility steering plots notebook:
  - `notebooks/qwen3_32b/mmraz-qwen3-32b-time-utility-probe-steering-plots.ipynb`
- Both notebooks:
  - auto-select the latest compatible run unless `RUN_DIR_OVERRIDE` is set
  - now print the selected `Run ID` explicitly
- The time-utility plots notebook additionally:
  - prefers a `baseline` already present in the selected run summary
  - only falls back to the older external 32B baseline summary if the selected run does not contain `baseline`

### 18.6 32B probe-vector logit lens

- Notebook:
  - `notebooks/qwen3_32b/mmraz-qwen3-32b-probe-vector-logit-lens.ipynb`
- Purpose:
  - Qwen3-32B analogue of the 4B probe-vector logit-lens notebook
  - computes `e^T @ W_U` for saved 32B probe vectors and displays top positive / negative vocabulary tokens
- Artifact discovery:
  - searches primarily under:
    - `results/qwen3_32b/question_only_probe_variations/`
  - expects:
    - `model_name == "Qwen/Qwen3-32B"`
    - `prompt_family == "question_only_teacher_forced_answers"`
    - question-level explicit and implicit splits
    - chat template enabled
    - thinking trace disabled
- Current default notebook selections:
  - `train_regime = "explicit_train_only"` if no metadata recommendation exists
  - `feature_name = "mean_answer_tokens"` if no metadata recommendation exists
  - all available layers
  - vector keys:
    - `mm_probe_vectors`
    - `wmm_probe_vectors`
- Current artifact bundle observed locally:
  - `results/qwen3_32b/question_only_probe_variations/20260410-093720/qwen3_32b_question_only_probe_artifacts_20260410-093720.npz`
- Output root:
  - `results/qwen3_32b/probe_vector_logit_lens/`
- Notebook structure detail:
  - the final outputs are intentionally split into separate cells:
    - first MM logit lens only
    - then WMM logit lens only
  - saved CSVs are likewise per vector family rather than one mixed table

### 18.7 32B HF cache note for logit lens

- Local Hugging Face cache currently contains a **partial** `Qwen/Qwen3-32B` snapshot, not the full model.
- The local partial cache includes at least:
  - tokenizer files
  - `model.safetensors.index.json`
  - `model-00017-of-00017.safetensors`
- This is enough for the 32B logit-lens notebook because it only needs:
  - tokenizer
  - weight index
  - the shard containing `lm_head.weight` / `model.embed_tokens.weight`
- Full `Qwen/Qwen3-32B` weights are much larger:
  - `model.safetensors.index.json` reports total size about `61 GiB` (`~65.5 GB`, 17 shards)
- Practical guidance:
  - do **not** download/cache the entire 32B snapshot just for the logit-lens notebook
  - the partial local cache is already sufficient for that notebook
