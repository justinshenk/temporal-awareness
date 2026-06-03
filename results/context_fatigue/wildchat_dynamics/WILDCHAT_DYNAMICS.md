# Do the context-fatigue signatures replicate on organic dialogue? (WildChat-1M)

**Short answer: no.** On 150 real multi-turn conversations, Qwen2.5-7B-Instruct shows
**no entropy collapse** and **no robust progressive current-turn neglect** — the two
"fatigue" signatures that looked strong on the synthetic repeating-task (DDXPlus) setup.
Both turn out to be artifacts of that setup (ICL-of-one-task comfort; the trivial
low-context geometry of early turns), not properties of long organic conversations.

## Why this is the decisive test
DDXPlus accumulates *one task repeating*, so entropy collapse there is confounded with
ICL competence (the model gets confident because it learned the one format). WildChat is
**organic, heterogeneous, topic-shifting** dialogue — no single task to in-context-learn.
If the signatures are real context-depth effects they should survive; if they are
ICL/setup artifacts they should vanish.

## Method
Stream WildChat-1M → 150 English conversations with ≥10 user turns whose full render fits
32k (Qwen2.5-7B-Instruct). Walk every user→assistant boundary in the **real** history
(teacher-forced; no generation drift). Per boundary, one `generate` call yields:
- **own-confidence entropy** — mean next-token Shannon entropy over an 8-token greedy
  probe of the model's *own* continuation (discarded), matching the DDXPlus metric;
- **last-query-token attention** split into first / middle / recent / current buckets
  (layers 0,7,14,21,27). 2039 boundaries total.
Driver: `scripts/context_fatigue/run_wildchat_dynamics.py`. Logic + tests:
`src/probes/context_fatigue/{wildchat_data,attention_capture}.py`.

## Result 1 — entropy does NOT collapse (the headline)

| | DDXPlus (repeating task) | WildChat (organic) |
|---|---|---|
| early→late entropy | 0.47 → 0.13 (**~3–4×**) | 0.39 → 0.32 |
| within-conversation late/early ratio | (collapse) | **0.99 median** (mean 1.09) |
| within-conv corr(entropy, depth) | strongly negative | **−0.02 median, −0.008 mean** |
| convs trending down | — | **52%** (coin-flip) |

Within-conversation entropy is **flat** with depth (only the rare 15+-turn bin dips, on
126/2039 obs). The 3–4× DDXPlus collapse **does not appear** on heterogeneous dialogue.

> **The entropy collapse is task-specific ICL comfort, not generic context fatigue.**
> This empirically settles the confound: confidence rose in DDXPlus because the model
> learned the one repeating task, not because accumulated context "tires" it.

## Result 2 — current-turn neglect is mostly a depth-0 artifact
Pooled, `frac_current` looks like it falls with depth (corr ≈ −0.43 to −0.53, all
layers). But **depth 0 has `frac_current = 1.0` by construction** (no prior context
exists to attend to). Excluding the trivial low-context turns (depth ≥ 2):

| layer | corr(frac_current, depth), depth≥0 | depth≥2 |
|---|---|---|
| L0  | −0.53 | −0.28 |
| L14 | −0.45 | **−0.11** |
| L27 | −0.43 | **−0.06** |

Once context is established, current-attention is roughly flat (~0.24–0.40 depending on
layer); the strong pooled correlation was driven by the early turns. What does shift with
depth is a redistribution from the opening tokens toward the growing middle — but that is
partly mechanical (the middle segment grows in size). So **progressive current-neglect
does not robustly replicate** in the wild.

## Interpretation
Combined with the flat-accuracy result and the instruction-adherence null, this is a
consistent story: **Qwen-7B does not behaviorally "fatigue" on real long conversations.**
The dramatic context-fatigue signatures live in the synthetic homogeneous setup and are
explained by (i) ICL comfort on one repeating task (entropy) and (ii) the trivial
geometry of early low-context turns (attention). The genuine deployment hazard, if any,
is calibration on repeated *identical* tasks — not organic multi-turn use.

## Caveats
One model (Qwen-7B); entropy measured via an 8-token own-confidence probe; attention
buckets use approximate token offsets from partial chat-template renders; WildChat
assistant turns were authored by other models (history content is real but not Qwen's).
The depth-0 artifact is a reminder to exclude mechanically-determined low-context turns
before reading any depth correlation.

## Reproduce
```bash
uv run python -m scripts.context_fatigue.run_wildchat_dynamics \
    --model Qwen/Qwen2.5-7B-Instruct --max-ctx 32768 \
    --min-turns 10 --n-convs 150 --max-boundaries 20 --probe-k 8
```
Outputs `turns.csv`, `attention.csv`, `summary.json` here; tests in
`tests/probes/context_fatigue/test_wildchat_data.py`.
