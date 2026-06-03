# WildChat partitioned by its own homogeneity: entropy↔heterogeneity, dilution↔length

Splitting WildChat by its **own** inter-turn homogeneity (format and dataset held
constant) isolates heterogeneity as the driver of the entropy signature — and separates
it cleanly from the attention signature, which tracks length instead. Both effects are
**modest but in the predicted direction**, significant at n=400, and the entropy effect
survives (indeed reverses) a length control.

## Design
400 English WildChat conversations (≥6 user turns, ≤16k tokens), Qwen2.5-7B-Instruct.
Per conversation: **homogeneity** = mean pairwise TF-IDF cosine across user turns
(high = user repeating similar tasks; low = topic-switching; independent of the model
under test). Per user→assistant boundary, one `generate` call captures own-confidence
entropy + per-head last-token attention (the same extraction answers both questions).
Within-conversation trajectories are the unit of analysis, so homogeneity cannot proxy
length. Driver: `run_wildchat_dynamics.py`; analysis: `analyze_wildchat_homogeneity.py`;
homogeneity: `src/probes/context_fatigue/wildchat_homogeneity.py`.

## Q1 — entropy collapse tracks homogeneity (output signature)

| | value |
|---|---|
| Pearson(homogeneity, entropy_slope) | **−0.141** (p = 0.005) |
| Spearman | −0.103 (p = 0.039) |
| Partial corr controlling tokens | **−0.151** |
| corr(homogeneity, max_tokens) | **+0.155** — homogeneous convs are *longer*, not shorter |

Tertile contrast (within-conversation late-vs-early entropy):

| group | n | mean homogeneity | median entropy_slope | median late/early entropy | median max_tokens |
|---|--:|--:|--:|--:|--:|
| **homogeneous** (top ⅓) | 134 | 0.220 | −0.120 | **0.897** (collapses ~10%) | 3127 |
| **heterogeneous** (bot ⅓) | 134 | 0.032 | −0.030 | **1.001** (flat) | 2566 |

Homogeneous conversations show a within-conversation entropy **collapse** (late entropy
~90% of early); heterogeneous ones are **flat**. The relationship survives a length
control — and since homogeneous conversations are actually *longer*, length works
**against** the effect, not for it. This isolates **heterogeneity** as the entropy driver
with format and dataset held perfectly constant — strictly stronger than the
format-confounded DDXPlus contrast (`WILDCHAT_DYNAMICS.md`).

The magnitude is mild because even "homogeneous" WildChat (TF-IDF ≈ 0.22) is far less
homogeneous than DDXPlus (literally one repeated task). The picture is monotone across
the whole spectrum: heterogeneous WildChat flat → homogeneous WildChat mild collapse →
DDXPlus strong (3–4×) collapse. Same axis, increasing dose.

## Q2 — current-query dilution tracks length, not homogeneity (attention signature)

| | value |
|---|---|
| pooled corr(frac_current, fill), L14, depth ≥ 2 | **−0.144** |
| corr(homogeneity, per-conv dilution_slope) | **−0.019** (≈ 0) |
| per-layer dilution (depth ≥ 2) | L0 −0.09, L7 +0.08, L14 −0.14, L21 −0.05, L27 −0.07 |

Current-turn attention falls as the context fills (modest, strongest at L14), and this
dilution is **independent of homogeneity** (−0.02). It is a length/fill effect, present
in both homogeneous and heterogeneous chats alike. (Depth 0–1 excluded throughout, since
`frac_current` is mechanically ≈ 1 with no prior context — the artifact from the earlier
run.)

## The dissociation (one dataset, format constant)
- **Output signature (entropy)** is driven by **heterogeneity** (corr −0.15, survives
  length control), not length.
- **Attention signature (dilution)** is driven by **length/fill** (corr −0.14), not
  heterogeneity (−0.02).

Two signatures, two distinct drivers, separated cleanly within a single in-distribution
dataset. "Entropy collapse" and "attention dilution" are **not the same phenomenon** and
should not be lumped under one "context fatigue" label: the model's *output confidence*
responds to whether it keeps seeing the same kind of task (an ICL-comfort effect), while
its *attention allocation* dilutes simply because there are more tokens to spread over.

## Caveats (honest magnitudes)
Effects are modest (real organic data): Pearson −0.14 / Spearman −0.10 for entropy;
dilution −0.05..−0.14, layer-dependent and weak outside L14. The consecutive-turn
homogeneity variant is only marginal (−0.095, p = 0.057); the all-pairs measure is the
stronger signal. One model (Qwen-7B); homogeneity is lexical (TF-IDF), not semantic.
Directions are as predicted and the length confound is ruled out.

## Reproduce
```bash
uv run python -m scripts.context_fatigue.run_wildchat_dynamics --min-turns 6 --n-convs 400 \
    --max-ctx 16384 --max-boundaries 16 --out-dir results/context_fatigue/wildchat_homogeneity
uv run python -m scripts.context_fatigue.analyze_wildchat_homogeneity \
    --in-dir results/context_fatigue/wildchat_homogeneity
```
Outputs `conversations.csv`, `turns.csv`, `attention.csv`, `attention_heads.parquet`,
`per_conversation.csv`, `homogeneity_analysis.json`. Tests:
`tests/probes/context_fatigue/test_wildchat_homogeneity.py`.
