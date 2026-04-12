# Temporal Representation Dataset for Activation Patching on Qwen3-4B

## Overview

200 prompt pairs for activation patching to locate where Qwen3-4B encodes temporal duration (short-term vs. long-term). A filtered subset of 135 pairs that Qwen3-4B correctly classifies is also available, along with a further-filtered 57-pair high-confidence subset (see [Model Evaluation](#model-evaluation)). Methodology follows Neel Nanda's attribution patching approach from "Attribution Patching: Activation Patching At Industrial Scale."

Each pair consists of a clean prompt (answer: " short") and a corrupted prompt (answer: " long"). Patching activations from corrupted into clean at specific model components reveals which components encode the temporal representation.

## Prompt Template

```
"The goal is to <goal>. Is this a <short-term or long-term / long-term or short-term> goal? The answer is:"
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 200 |
| Question order | 100 SL / 100 LS |
| Word count alignment | 96% equal |
| Clean verb diversity | 124 unique |
| Corrupted verb diversity | 109 unique |
| Clean avg chars/word | 3.69 |
| Corrupted avg chars/word | 4.47 |
| Exact duplicates | 0 |
| Near-duplicates (Jaccard ≥ 0.50) | 1 pair (7/13, accepted) |
| All syntactic templates | < 10% each |

## Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Growth (G) | 80 | 40% | Transforming something small into something large/established |
| Career/Mastery (C) | 77 | 38.5% | Achieving elite status or deep expertise |
| Accumulation (A) | 43 | 21.5% | Exhaustive scope requiring years of sustained effort |

## Domain Distribution

25 domains, 4–13 pairs each (mean 8.0):

| Domain | Pairs | | Domain | Pairs |
|--------|-------|-|--------|-------|
| photography | 13 | | programming | 8 |
| gardening | 12 | | writing | 7 |
| boating | 11 | | visual art | 7 |
| cooking | 10 | | woodworking | 7 |
| languages | 9 | | pets | 7 |
| riding | 9 | | stargazing | 7 |
| board games | 9 | | ceramics | 5 |
| dance | 9 | | beekeeping | 4 |
| team sports | 9 | | collecting | 4 |
| combat sports | 9 | | birding | 4 |
| fitness | 8 | | | |
| music | 8 | | | |
| swimming | 8 | | | |
| singing | 8 | | | |
| hiking | 8 | | | |

## Model Evaluation

### Qwen3-4B

Qwen3-4B was evaluated on all 200 pairs. It correctly classified 135/200 pairs (67.5% accuracy). The 65 failing pairs are recorded in `Qwen3_4B_failing_pairs.json`.

A filtered dataset of the 135 surviving pairs is available in `dataset_135.json` (original IDs preserved). 12 pairs had their question order flipped from SL to LS to rebalance the subset.

#### Surviving Dataset Statistics (135 pairs)

| Metric | Value |
|--------|-------|
| Total pairs | 135 |
| Question order | 68 SL / 67 LS |

#### Surviving Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Growth (G) | 38 | 28.1% | Transforming something small into something large/established |
| Career/Mastery (C) | 66 | 48.9% | Achieving elite status or deep expertise |
| Accumulation (A) | 31 | 23.0% | Exhaustive scope requiring years of sustained effort |

#### Surviving Domain Distribution

25 domains, 1–8 pairs each (mean 5.4):

| Domain | Pairs | | Domain | Pairs |
|--------|-------|-|--------|-------|
| gardening | 8 | | singing | 5 |
| board games | 8 | | team sports | 5 |
| programming | 8 | | combat sports | 5 |
| boating | 8 | | stargazing | 5 |
| cooking | 7 | | languages | 4 |
| writing | 7 | | dance | 4 |
| photography | 7 | | beekeeping | 3 |
| music | 6 | | birding | 3 |
| woodworking | 6 | | collecting | 2 |
| riding | 6 | | ceramics | 1 |
| hiking | 6 | | | |
| pets | 6 | | | |
| fitness | 5 | | | |
| visual art | 5 | | | |
| swimming | 5 | | | |

### Logit-Diff Filtering (57 pairs)

Many of the 135 correct-first-token pairs have weak logit differences — the model barely favors the right answer. 23 pairs even have the wrong sign on the corrupted side due to different settings in classification validation and activation patching experiments. These weak pairs add noise to activation patching.

`dataset_57.json` contains 57 high-confidence pairs where:
- Clean logit_diff (logit "short" − logit "long") > 1.0
- Corrupted logit_diff (logit "short" − logit "long") < −1.0

See `logit_filtering.md` for full details. Original IDs preserved; 16 pairs had their question order flipped from SL to LS to rebalance the subset.

#### Strong Dataset Statistics (57 pairs)

| Metric | Value |
|--------|-------|
| Total pairs | 57 |
| Question order | 28 SL / 29 LS |
| Clean mean logit_diff | +3.76 |
| Corrupted mean logit_diff | −2.73 |

#### Strong Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Growth (G) | 11 | 19.3% | Transforming something small into something large/established |
| Career/Mastery (C) | 38 | 66.7% | Achieving elite status or deep expertise |
| Accumulation (A) | 8 | 14.0% | Exhaustive scope requiring years of sustained effort |

#### Strong Domain Distribution

22 domains, 1–5 pairs each (mean 2.6):

| Domain | Pairs | | Domain | Pairs |
|--------|-------|-|--------|-------|
| writing | 5 | | gardening | 3 |
| programming | 5 | | photography | 3 |
| pets | 5 | | woodworking | 3 |
| hiking | 4 | | riding | 3 |
| cooking | 3 | | board games | 3 |
| music | 3 | | team sports | 3 |
| fitness | 2 | | languages | 2 |
| swimming | 2 | | singing | 1 |
| combat sports | 2 | | dance | 1 |
| visual art | 1 | | beekeeping | 1 |
| stargazing | 1 | | boating | 1 |

## Design Constraints

1. **Semantic overlap within pairs.** Clean and corrupted share the same domain. Only temporal horizon differs. Both must be on the same life continuum ("would the same person do both?").

2. **No explicit temporal keywords.** No "daily," "weekly," "years," "quick," "soon," "lifetime," "generational." Temporal horizon inferred from world knowledge only.

3. **Unambiguous horizons.** Short-term: completable in hours or a single sitting. Long-term: requires years of sustained effort. Clarity test applied: "Could a reasonable person achieve this in under a month?"

4. **Semantic coherence.** All prompts describe goals a real person would say out loud.

5. **Token alignment.** All 400 prompts target the same token length under Qwen3-4B.

6. **Question order shuffle.** 50/50 split, randomly assigned. Both prompts in a pair share the same order.

7. **No exact duplicate goals.** No two pairs share an identical clean or corrupted goal.

8. **Lexical diversity.** No content word exclusively on one side more than ~6 times (with two exceptions noted in limitations). No single corrupted verb more than 6 uses. No syntactic template in more than 10% of corrupted prompts.

## Known Limitations

### 1. Vocabulary Complexity Gap (+0.77 chars/word)
Corrupted prompts naturally use longer, rarer words ("professional," "championship") while clean prompts use shorter, domestic words ("cook," "brush," "swim"). The model could partially distinguish clean from corrupted via word-level complexity features rather than temporal semantics. Inherent to the task — long-term goals have more complex descriptions in natural language.

### 2. One-Sided Scope Markers
Several scope words appear exclusively in corrupted prompts: "whole" (12), "county" (8), ... The model could use these as surface cues. Mitigated by the fact that no single word dominates the full dataset — "whole" at 12/200 (6%) is the highest, and 188 pairs do not contain it.

### 3. Scale/Scope Confound
Growth (G) and Accumulation (A) cue types signal temporal duration indirectly through scale — "for the whole region" implies years because building a regional operation takes time. The model might learn a "scale" circuit rather than a "temporal duration" circuit. **Mitigation:** split patching results by cue type (C vs. A vs. G). If the same components activate across all three, the circuit is genuinely temporal. If different components activate for C (mastery) vs. A/G (scale), the model uses separate mechanisms.

### 4. Ambition-Level Confound
Most short-term goals are domestic/mundane while most long-term goals are aspirational. The model might distinguish ambition level rather than temporal duration. **Mitigation:** build a separate validation dataset (30–50 pairs) with ambition-decoupled pairs — short-term but ambitious ("perform tonight's solo at the concert") and long-term but mundane ("pay off the household mortgage").

## Recommended Analysis from Claude:

1. **Run activation patching** across all 200 pairs and average.
2. **Split-half correlation:** run on two random halves and verify top components agree.
3. **Split by cue type:** compare C vs. A vs. G to check if the same circuit handles all three temporal cue types.
4. **Validation dataset:** test discovered circuits on ambition-decoupled pairs.
5. **Token-position analysis:** 96% of pairs are word-count-aligned, enabling position-level patching comparison.
