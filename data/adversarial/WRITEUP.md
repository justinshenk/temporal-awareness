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
- **Qwen-7B (32k):** 41.7% modification quiz accuracy, 20.3% re-ask update rate, 6.8% revert rate. With a lightweight prompt forcing "Noted." replies on corrections (preventing echo), update rate stays low — the model wasn't truly integrating corrections, just echoing them.

When models fail to update after a correction, they mostly don't revert to the original answer (6.8% revert rate). Instead they produce a **third, unrelated answer** — suggesting the correction disrupts retrieval without successfully replacing the fact.

### Activation drift (Code Review)

By hooking into Phi-3's layers and comparing activations between two runs of the same conversation (one with code visible, one with code stripped), we measured how internal representations diverge. Cosine similarity between activation vectors starts at ~1.0 (turns 0-2, < 15% context fill) and drops to ~0.5-0.6 by 80% fill. This confirms that fatigue isn't just a behavioral artifact — the model's internal representations are physically drifting as context accumulates.

### Hedging and behavioral shifts

In medical QA, hedging language ("possibly", "could be", "might") peaks at 25-50% context fill then drops — models become more assertive (but not more correct) in later turns. In code review, response specificity declines: early responses reference specific variable names and line numbers, while late responses become increasingly generic.

### Two modes of fatigue: template lock-in vs. destabilization

A critical finding is that entropy moves in **opposite directions** depending on the task structure:

In **MedQA and Code Review**, where the model encounters repeated similar tasks (patient after patient, code snippet after code snippet), entropy **decreases** as context fills (Phi-3 NarrativeQA old-style: 0.70 → 0.08). The model locks into a response template — it stops attending to task-specific details and produces increasingly confident, formulaic answers. It "sounds sure" but accuracy degrades. Responses stay short and repetitive.

In **NarrativeQA adversarial**, where each story introduces unique content and corrections that contradict prior facts, entropy **increases** (Qwen-7B: 0.08 → 0.28 on per-story entropy). The model can't settle into a template because corrections actively destabilize its representations. Responses become verbose — mean followup length nearly doubles from 626 to 1,194 characters at 50-75% fill — as the model compensates for uncertainty by generating more text.

These are two distinct failure modes of the same underlying phenomenon:

- **Template lock-in** (repetitive tasks): model over-compresses, ignoring new information. Entropy drops, confidence rises, accuracy falls silently.
- **Destabilization** (contradictory information): model under-compresses, unable to resolve conflicts between accumulated facts. Entropy rises, responses become verbose and hedged, the model "talks more but says less."

Both represent fatigue — the model's inability to maintain consistent, calibrated performance as context accumulates. The direction of entropy change depends on whether the accumulated context reinforces patterns (lock-in) or introduces contradictions (destabilization).

### Recall and the shrinking attention window

The recall phase (asking the model to summarize each story after all stories are processed) reveals how attention distributes across context at high fill. At 86-91% context fill with 13 stories accumulated:

- **Early stories (S0-S2):** recalled with low entropy (0.23-0.25), relatively confident
- **Mid stories (S5-S8):** recalled with high entropy (0.42-0.50), most uncertain
- **Late stories (S10-S12):** entropy drops back slightly (0.30-0.41)

This U-shaped pattern suggests the model doesn't simply "forget" early stories via recency bias. Instead, early stories benefit from having been processed with a clean, uncluttered context. Mid-conversation stories sit in the worst position: enough prior context to create interference, but not recent enough to benefit from recency bias. This aligns with the 50% fill degradation peak observed in the main experiment.

## Dataset Structure

```
data/adversarial/
├── medqa/
│   ├── medqa_interactive.csv/.parquet       (311 rows, 26 cols)
│   └── conversations/                       (8 JSONL files)
├── code_review/
│   ├── code_review_turns.csv/.parquet       (41 rows, 23 cols)
│   ├── code_review_activation_drift.csv     (13 rows)
│   └── conversations/                       (3 JSONL files)
└── narrativeqa/
    ├── narrativeqa_adversarial.csv/.parquet  (325 rows, 35 cols)
    ├── narrativeqa_modifications.json        (187 stories, 748 modifications)
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
