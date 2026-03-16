# RQ3 — Context Window Degradation in Multi-Turn LLM Conversations

Datasets for studying how small LLMs degrade as their context window fills up across multi-turn conversations. Covers medical QA and code review domains.

## Models

| Model | Context Window |
|-------|---------------|
| microsoft/Phi-3-mini-4k-instruct | 4,096 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 8,192 |
| Qwen/Qwen2.5-7B-Instruct | 32,768 |

## Datasets

### `medqa/`

#### `medqa_incremental_reveal_turns` (79 rows)
Turn-level data from MedQA-USMLE cases where patient info is revealed incrementally (demographics → HPI → vitals → PE → labs → final question). Models: Phi-3, SmolLM2.

| Column | Description |
|--------|-------------|
| model | Model identifier |
| global_turn | Sequential turn number |
| case_idx | Case identifier |
| turn_type | demographics_reveal, hpi_reveal, vitals_reveal, pe_reveal, lab_reveal, diagnostic_question, recall_check, final_question |
| context_tokens / context_utilization | Context window usage |
| answer_correct | Whether final answer was correct (null for non-answer turns) |
| hedging_detected / hedge_words | Whether model hedged and which words |
| entropy_mean / entropy_max | Token-level entropy statistics |
| response_preview / user_input | Truncated text (200/300 chars) |

#### `medqa_interactive_turns` (311 rows) + `medqa_interactive_cases` (236 rows)
Models act as doctors, can request labs/imaging/history. Three prompt versions (v1, v2, v3). Models: Phi-3, SmolLM2, Qwen-7B.

**Turns** — per-turn metrics (entropy, hedging, context fill, logprobs).
**Cases** — per-case outcomes (correct_answer vs model_answer, num_followups, final_context_fill).

#### `medical_diagnosis_ddxplus_turns` (73 rows)
Multi-turn medical diagnosis on DDXPlus dataset. Symptoms revealed one at a time. Models: Phi-3, SmolLM2.

Includes `diagnosis_correct` (bool) for turns where diagnosis tracking was possible.

#### `activation_stats` (4,048 rows)
Per-layer, per-turn neural activation statistics (mean, std, norm, min, max) from the medical and MedQA experiments. Useful for studying representation drift across turns.

#### `conversations/` (10 JSONL files)
Full conversation transcripts in `{"role": ..., "content": ...}` format.

### `code_review/`

#### `code_review_turns` (41 rows)
Turn-level data from code review conversations (Phi-3 only, 3 versions). Tasks include explain, reason, modify, review with easy/hard difficulty.

| Column | Description |
|--------|-------------|
| turn_type | explain, reason, modify, review |
| difficulty | easy, hard |
| has_code | Whether response contained code |
| has_specific | Whether response was specific to the code |
| references_prior | Whether response referenced prior context |
| is_generic | Whether response was generic/boilerplate |

#### `code_review_activation_drift` (13 rows)
Compares activation vectors between code review v1 and v3 at each turn. Shows cosine similarity dropping from ~1.0 to ~0.5-0.6 as context fills.

| Column | Description |
|--------|-------------|
| v1_entropy / v3_entropy | Mean entropy for each version at this turn |
| cosine_similarity | Cosine sim between v1 and v3 activation vectors |
| delta_norm | Norm difference between activation vectors |

#### `conversations/` (3 JSONL files)
Full conversation transcripts for all three code review versions.

## File Formats

All tabular data is provided in both **CSV** (human-readable) and **Parquet** (efficient, typed) formats. Conversations are in **JSONL** (one JSON object per line).

## Key Metrics

- **entropy_mean**: Mean per-token entropy of model output (higher = more uncertain)
- **hedging_detected**: Whether response contained hedging language ("could be", "possibly", "might")
- **context_fill / context_utilization**: Fraction of max context window used (0.0 → 1.0)
- **cosine_similarity**: Similarity between activation vectors across experimental conditions
