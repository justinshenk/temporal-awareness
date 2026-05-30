# Context Rot in OLMo-2: Attention Dynamics vs Per-Case Performance

Does "context rot" — degradation as a conversation accumulates — show up as a
*measurable attention shift*, and does that shift track *per-case* failure? We
run DDXPlus medical MCQ accumulation on OLMo-2 7B, capturing, for every case,
where the last token's attention lands (system prompt / early cases / recent
cases / current query), its entropy, and its peak, at layers {0, 8, 16, 24, 31};
each case is paired with whether the model answered it correctly.

Attention is replicated exactly for OLMo-2 (q_norm(q_proj)→reshape→RoPE, last
query token only) and **validated against the model's own `output_attentions`
to max|Δ| ≈ 1.4e-3** (bf16 rounding). Because a 4096-token context only holds
~11 cases, we pool many independent accumulation **sessions** (different case and
option orders) to power the per-case analysis.

Script: `scripts/context_fatigue/run_olmo_attention.py` (re-analyze a saved CSV
with `analyze_attention.py`). Data: `results/olmo_attention_{instruct,dpo,sft}/`.

---

## 1. Context rot is a strong, monotonic attention shift

OLMo-2-Instruct, attention mass by context fill (mean over heads), layer 24:

| context fill | system | early cases | recent cases | current query | attn entropy |
|---|---:|---:|---:|---:|---:|
| 0–25%   | 0.295 | 0.200 | 0.145 | 0.360 | 2.99 |
| 25–50%  | 0.130 | 0.399 | 0.283 | 0.189 | 3.62 |
| 50–75%  | 0.057 | 0.376 | 0.391 | 0.175 | 3.97 |
| 75–100% | 0.035 | 0.326 | 0.477 | 0.162 | 4.19 |

All four signatures the writeup reported for Qwen replicate in OLMo, as strong
per-case correlations with context fill (pooled, **n = 115 cases, 10 sessions**):

| layer | system↔fill | recent↔fill | current↔fill | entropy↔fill |
|---|---:|---:|---:|---:|
| 0  | −0.70 | +0.79 | −0.77 | +0.93 |
| 8  | −0.86 | +0.84 | −0.68 | +0.96 |
| 16 | −0.92 | +0.93 | −0.80 | +0.89 |
| 24 | −0.93 | +0.89 | −0.71 | +0.95 |
| 31 | −0.93 | +0.93 | −0.67 | +0.96 |

- **System-prompt erosion** (system↔fill ≈ −0.9 in deep layers): final-layer
  system attention falls ~0.28 → 0.02 as context fills. The model progressively
  stops attending to its instructions.
- **Recency bias** (recent↔fill ≈ +0.9): attention to recent prior cases more
  than triples (0.12 → 0.49).
- **Current-query neglect** (current↔fill ≈ −0.7): attention to the actual
  question being answered roughly halves.
- **Attention diffuses** (entropy↔fill ≈ +0.95): last-token attention spreads out
  as context grows.

---

## 2. The per-case performance link inverts the naive hypothesis

Naive guess: the model errs when it *neglects* the current query. The data say
the opposite. Conditioning on context fill (so the result isn't just the fill
trend), at layer 24 the cases the model gets **wrong** have **higher** current-
query attention in every fill bin:

| fill bin | n (✓/✗) | current ✓ | current ✗ | Δ (✗−✓) |
|---|---:|---:|---:|---:|
| 0–25%   | 8/10 | 0.307 | 0.402 | **+0.095** |
| 25–50%  | 12/5 | 0.185 | 0.197 | +0.011 |
| 50–75%  | 11/5 | 0.169 | 0.188 | +0.019 |
| 75–100% | 6/3  | 0.153 | 0.180 | +0.027 |

And wrong cases have **lower, more peaked** attention entropy (layer 24:
3.74 correct → 3.40 wrong; same direction at every probed layer).

**Interpretation.** Errors are not driven by ignoring the query — they are
associated with *fixating* on it. The cases OLMo answers **correctly** are the
ones where attention spreads off the current query and onto the accumulated
prior cases (early + recent) — i.e. where it actually *uses the in-context
examples*. When the model instead stares narrowly at the current query (high
current-query mass, low-entropy/peaked attention), it is more likely to be
wrong. This is the per-case mechanism behind the writeup's macro observation
that accumulated context *helps* accuracy via in-context learning: the benefit
shows up case-by-case as attention allocated to context rather than to the
query in isolation.

So the two halves of "context rot" pull in opposite directions:
- the **global drift** (system erosion, recency bias, diffusion) is genuine rot;
- but the **within-context predictor of a correct answer** is *more* contextual
  attention, not more query focus.

---

## 3. Across the post-training gradient

Pooled DPO (n=115) is **indistinguishable** from Instruct — layer-24 correlations
sys/recent/current/entropy ↔ fill = −0.93 / +0.89 / −0.71 / +0.95 for *both*, and
the same per-case performance inversion (current ✓→✗ 0.218→0.269). The attention
rot is therefore present from the SFT/DPO stage and **not introduced by the final
RLVR step** — consistent with the dose-response finding that SFT installs most of
the behavioral change.

## 4. Accuracy is stable — the rot does not lower the score

Critically, the attention rot does **not** produce falling accuracy. Per-case
accuracy by context fill (OLMo-2-Instruct, n=115):

| context fill | accuracy | n |
|---|---:|---:|
| 0–20%   | 0.50 | 28 |
| 20–40%  | 0.70 | 27 |
| 40–60%  | 0.73 | 26 |
| 60–80%  | 0.61 | 23 |
| 80–100% | 0.73 | 11 |

`corr(correct, fill) = +0.11` (Instruct), `+0.07` (DPO) — slightly **positive**.
The model is *worst at the start* (cold, 0.50), warms up via in-context learning,
then plateaus ~0.70. So context rot manifests in **attention allocation and
calibration** (entropy collapses ~5× — see `RLHF_DOSE_RESPONSE.md`), **not in the
accuracy score**, because in-context learning compensates. The hazard is the
*confidently-wrong gap*: confidence runs ahead while accuracy holds flat.

> Caveat: the homogeneous single-task stream is exactly what lets ICL compensate.
> See `MIXED_TASK_ICL.md` for the interleaved medical+MMLU control that removes
> the consistent task pattern to test whether accuracy then degrades.

---

## Bottom line

Context rot in OLMo-2 is a real, strongly-measurable reallocation of attention
(instructions out, recency in, query focus down, attention diffusing) that grows
monotonically with context fill and is validated exactly against the model's own
attention. But tying it to per-case performance overturns the simple story:
within a given context level, the model fails on the cases where it *fixates on
the current query*, and succeeds where it *spreads attention onto the accumulated
context* — the per-case fingerprint of in-context learning.
