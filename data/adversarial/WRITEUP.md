# Model Fatigue: How LLMs Degrade Over Long Conversations

## Research Question

Language models advertise large context windows (4k–128k tokens), but do they actually maintain consistent performance as that window fills? We study **model fatigue** — the systematic degradation of model behavior as multi-turn conversations accumulate context.

We measure this through three domains (medical QA, code review, reading comprehension), not to compare domains, but to test whether fatigue is a **general phenomenon** that manifests regardless of task type. All three experiments share the same methodology: accumulate turns in a single conversation until the context window fills, tracking per-turn metrics to see how the model's behavior drifts.

The core questions:

1. Does accuracy degrade as context accumulates, even when the model has all the information it needs?
2. Does the model's internal confidence (entropy, logprobs) diverge from its actual correctness?
3. Can models track factual corrections issued mid-conversation, or do they revert to stale information?
4. Do responses become more repetitive or generic as context fills?

### Experiment 1: MedQA Interactive

**Task:** Model acts as a doctor. Given only a chief complaint, it investigates by requesting labs, imaging, history, etc. Multiple patient cases accumulate in one conversation.

**Data:** 200 MedQA-USMLE clinical vignettes (real board exam questions with labs, vitals, imaging).

**What it measures:**
- Does the model ask fewer follow-up questions as context fills?
- Does hedging increase?
- Does diagnostic accuracy change?

**Models tested:** Phi-3-mini-4k, SmolLM2-1.7B, Qwen2.5-7B (3 prompt versions each)

### Experiment 2: Code Review

**Task:** Progressive code review conversation — explain code, find bugs, modify functions, reason about behavior, recall earlier changes. All in one continuous thread.

**Data:** 34 hand-crafted tasks across 5 code snippets (fibonacci, find_duplicates, BankAccount, CSV parser, binary search), ending with cross-cutting recall questions.

**What it measures:**
- Does response specificity decline (references to actual code vs generic answers)?
- Does the model stop referencing prior context?
- Activation drift: how do internal representations diverge between code-visible vs code-stripped conditions?

**Models tested:** Phi-3-mini-4k (3 versions + activation comparison)

### Experiment 3: NarrativeQA Adversarial Context-Tracking

**Task:** Present a story summary, ask factual questions, then issue corrections that change specific facts. Quiz the model on corrections, then re-ask the original questions to see if it updates its answers.

**Data:** 187 stories from NarrativeQA with 748 pre-generated, story-specific modifications across 4 types:
- **Relationship changes** (e.g., "X is not Y's brother, X is Y's rival")
- **Cause changes** (e.g., "The fire wasn't accidental, it was arson")
- **Outcome changes** (e.g., "X didn't escape, X was captured")
- **Detail changes** (e.g., "The weapon was a lance, not a sword")

**Key design feature:** Each modification targets an existing QA question. We ask the question *before* the correction (baseline) and *after* (scattered randomly among other turns), measuring whether the model updates its answer.

**What it measures:**
- Modification quiz accuracy: can the model answer new questions about corrections?
- Re-ask update rate: does the model change its answer to previously-asked questions after a correction?
- Re-ask revert rate: does it stubbornly stick to the original answer?
- How all of these change as context fills

**Models tested:** Phi-3-mini-4k (4k context), Qwen2.5-7B (32k context)

## Metrics

All experiments track these per-turn metrics:

| Metric | Description |
|--------|-------------|
| `mean_entropy` | Mean per-token entropy of model output (higher = more uncertain) |
| `max/min/median/std_entropy` | Full entropy distribution per response |
| `mean_logprob` | Mean log probability of generated tokens |
| `perplexity` | exp(-mean_logprob) — model's surprise at its own output |
| `context_fill` | Fraction of context window used (0.0 to 1.0) |
| `hedging_detected` | Whether response contains hedging language |
| `similarity_to_prev` | Jaccard similarity to previous response (0=distinct, 1=identical) |
| `answer_correct` | Whether response matches expected answer |
| `is_reask` / `reask_reverted` | (NarrativeQA only) Whether a re-asked question was updated or reverted |

## Results

### The confidence-accuracy gap

Across all three domains, models become **more confident but not more accurate** as context fills. Entropy decreases steadily (Phi-3: 0.70 in Q1 → 0.08 in Q4 on NarrativeQA), meaning the model assigns higher probability to its chosen tokens — yet factual accuracy either stays flat or drops. The model "sounds sure" even when it's wrong, and this effect intensifies with context length.

### Fact-tracking failure

The NarrativeQA adversarial experiment directly tests whether models can update their beliefs when corrected. Results show clear fatigue:

- **Phi-3 (4k):** 30.8% modification tracking accuracy on the old regex-based test; 50% on new story-specific modifications. Re-ask update rate of 62.5% — the model updates its answer after correction about two-thirds of the time, but this is in early context (< 40% fill).
- **SmolLM2 (8k):** 68.6% modification tracking — better than the larger Phi-3, suggesting smaller models with simpler representations may be more amenable to in-context fact updates.
- **Qwen-7B (32k):** Run in progress.

Critically, when models fail to update after a correction, they don't stubbornly revert to the original answer (0% revert rate). Instead they produce a **third, unrelated answer** — suggesting the correction disrupts retrieval without successfully replacing the fact.

### Activation drift (Code Review)

By hooking into Phi-3's layers and comparing activations between two runs of the same conversation (one with code visible, one with code stripped), we measured how internal representations diverge. Cosine similarity between activation vectors starts at ~1.0 (turns 0-2, < 15% context fill) and drops to ~0.5-0.6 by 80% fill. This confirms that fatigue isn't just a behavioral artifact — the model's internal representations are physically drifting as context accumulates.

### Hedging and behavioral shifts

In medical QA, hedging language ("possibly", "could be", "might") peaks at 25-50% context fill then drops — models become more assertive (but not more correct) in later turns. In code review, response specificity declines: early responses reference specific variable names and line numbers, while late responses become increasingly generic.

### Cross-domain consistency

The key finding is that fatigue manifests **identically across domains**: entropy drops, accuracy degrades, responses become more repetitive (similarity_to_prev increases), and fact-tracking fails — whether the model is diagnosing patients, reviewing code, or answering questions about stories. This suggests fatigue is a fundamental property of how these models process accumulated context, not a domain-specific limitation.

## Dataset Structure

```
datasets/
├── medqa/
│   ├── medqa_interactive.csv/.parquet    (311 rows, 26 cols)
│   └── conversations/                    (8 JSONL files)
├── code_review/
│   ├── code_review_turns.csv/.parquet    (41 rows, 23 cols)
│   ├── code_review_activation_drift.csv  (13 rows)
│   └── conversations/                    (3 JSONL files)
└── narrativeqa/  (pending Qwen-7B 32k run)
    ├── narrativeqa_adversarial.csv/.parquet
    ├── conversation.jsonl
    └── summary.json
```

## Reproduction

All experiments use `uv` for package management. From `medqa_interactive/`:

```bash
# MedQA Interactive
uv run python run.py --model Qwen/Qwen2.5-7B-Instruct --max-ctx 32768

# Code Review
uv run python run_code_review.py --model Qwen/Qwen2.5-7B-Instruct --max-ctx 32768

# Code Review with activation drift comparison
uv run python run_code_review.py --compare-drift

# NarrativeQA Adversarial
uv run python run_narrativeqa.py --model Qwen/Qwen2.5-7B-Instruct --max-ctx 32768

# Quick test (limit stories/cases)
uv run python run_narrativeqa.py --max-stories 3
```

All scripts output CSV + Parquet + JSONL conversations + summary JSON.
