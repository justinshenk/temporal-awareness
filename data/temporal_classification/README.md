# Temporal Representation Dataset for Activation Patching on Qwen3-4B

## Overview

200 prompt pairs for activation patching to locate where Qwen3-4B encodes temporal duration (short-term vs. long-term). A filtered subset of 116 pairs that Qwen3-4B correctly classifies and creates correct logit differences for is also available. Two main datasets that should be used in patching experiments are further-filtered subsets of 45 high-confidence pairs and 24 asymmetry-analysis pairs (see [Model Evaluation](#model-evaluation)). Methodology follows Neel Nanda's attribution patching approach from "Attribution Patching: Activation Patching At Industrial Scale."

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
| Career/Mastery (C) | 77 | 38.5% | Achieving elite status or deep expertise |
| Growth (G) | 80 | 40% | Transforming something small into something large/established |
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

Qwen3-4B was evaluated on all 200 pairs. It correctly classified and created correct logit differences on 116/200 pairs (58% accuracy). However, per prompt (not per pair) validation shown 83.5% accuracy.
A filtered dataset of the 116 surviving pairs is available in `datasets/survived_dataset_116.json` (original IDs preserved). SL/LS question order isn't balanced.

#### Surviving Dataset Statistics (116 pairs)

| Metric | Value |
|--------|-------|
| Total pairs | 116 |
| Question order | 72 SL / 44 LS |

#### Surviving Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Career/Mastery (C) | 63 | 54.3% | Achieving elite status or deep expertise |
| Growth (G) | 28 | 24.1% | Transforming something small into something large/established |
| Accumulation (A) | 25 | 21.6% | Exhaustive scope requiring years of sustained effort |

### Logit-Diff Filtering (45 pairs)

Many of the 116 correctly classified pairs have weak logit differences - the model barely favors the right answer. This adds noise to activation patching.

`datasets/strong_dataset_45.json` contains 45 high-confidence pairs where:
- Clean logit_diff (logit "short" − logit "long") > 1.0
- Corrupted logit_diff (logit "short" − logit "long") < −1.0

#### Strong Dataset Statistics (45 pairs)

| Metric | Value |
|--------|-------|
| Total pairs | 45 |
| Question order | 24 SL / 21 LS |
| Clean mean logit_diff | +3.98 |
| Corrupted mean logit_diff | −2.50 |
| Baseline ratio | 1.59 |

#### Strong Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Career/Mastery (C) | 31 | 68.9% | Achieving elite status or deep expertise |
| Growth (G) | 9 | 20.0% | Transforming something small into something large/established |
| Accumulation (A) | 5 | 11.1% | Exhaustive scope requiring years of sustained effort |

#### Strong Domain Distribution

22 domains, 1–4 pairs each (mean 2.0):

| Domain | Pairs | | Domain | Pairs |
|--------|-------|-|--------|-------|
| pets | 4 | | programming | 2 |
| cooking | 3 | | hiking | 2 |
| music | 3 | | singing | 1 |
| woodworking | 3 | | dance | 1 |
| riding | 3 | | visual art | 1 |
| board games | 3 | | beekeeping | 1 |
| team sports | 3 | | languages | 1 |
| writing | 3 | | stargazing | 1 |
| fitness | 2 | | boating | 1 |
| gardening | 2 | | combat sports | 1 |
| swimming | 2 | | photography | 2 |

### Asymmetry Analysis Set (24 pairs)

For testing whether layer importance differs between short→long and long→short patching directions, a stricter threshold is needed to ensure both sides have strong baselines. Otherwise, raw patching effects are dominated by whichever side has the larger logit_diff, producing spurious asymmetry.

`datasets/asymmetry_dataset_24.json` contains 24 pairs with stricter thresholds:
- Clean logit_diff (logit "short" − logit "long") > 2.0
- Corrupted logit_diff (logit "short" − logit "long") < −2.0

#### Asymmetry Dataset Statistics (24 pairs)

| Metric | Value |
|--------|-------|
| Total pairs | 24 |
| Question order | 14 SL / 10 LS |
| Clean mean logit_diff | +4.60 |
| Corrupted mean logit_diff | −3.45 |
| Baseline ratio | 1.33 |

#### Asymmetry Temporal Cue Types

| Type | Count | % | Description |
|------|-------|---|-------------|
| Career/Mastery (C) | 20 | 83.3% | Achieving elite status or deep expertise |
| Growth (G) | 3 | 12.5% | Transforming something small into something large/established |
| Accumulation (A) | 1 | 4.2% | Exhaustive scope requiring years of sustained effort |

#### Asymmetry Domain Distribution

18 domains, 1–3 pairs each (mean 1.3):

| Domain | Pairs | | Domain | Pairs |
|--------|-------|-|--------|-------|
| riding | 3 | | board games | 1 |
| swimming | 2 | | singing | 1 |
| woodworking | 2 | | programming | 1 |
| cooking | 2 | | dance | 1 |
| pets | 2 | | team sports | 1 |
| fitness | 1 | | writing | 1 |
| music | 1 | | visual art | 1 |
| photography | 1 | | hiking | 1 |
| gardening | 1 | | boating | 1 |

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
3. **Validation dataset:** test discovered circuits on ambition-decoupled pairs.
4. **Token-position analysis:** 96% of pairs are word-count-aligned, enabling position-level patching comparison.
