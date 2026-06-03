# Current task — WildChat homogeneity partition (entropy vs attention dissociation)

Stronger replacement for the DDXPlus-vs-WildChat contrast: split WildChat by its OWN
homogeneity, holding format + dataset constant, to isolate heterogeneity as the driver.

## Two questions, one extraction
- **Q1 (output):** does the own-confidence entropy collapse track *homogeneity*?
  Prediction: homogeneous convs collapse (entropy_slope < 0), heterogeneous flat →
  corr(homogeneity, entropy_slope) < 0, surviving a length control (partial corr on tokens).
- **Q2 (attention):** does current-query dilution track *length* independent of homogeneity?
  Prediction: frac_current falls with fill regardless of homogeneity → pooled dilution < 0
  but corr(homogeneity, dilution_slope) ≈ 0.
- **Dissociation:** if entropy collapse tracks homogeneity while dilution tracks length,
  the output and attention signatures separate cleanly within one dataset.

## Homogeneity = TF-IDF cosine between user turns (sklearn; no deps; independent of Qwen)
High = user repeating similar tasks (15 translations); low = topic-switching.
Length control: per-conversation within-trajectory slopes + partial corr on tokens
(homogeneous chats may be shorter — must not let homogeneity proxy length).

## Build (all done, run in flight)
- `src/probes/context_fatigue/wildchat_homogeneity.py` (+6 tests) — homogeneity scoring.
- `scripts/context_fatigue/run_wildchat_dynamics.py` — EXTENDED: per-conv homogeneity,
  per-head attention (attention_heads.parquet), conversations.csv.
- `scripts/context_fatigue/analyze_wildchat_homogeneity.py` — partition analysis (Q1/Q2).
- Run: 400 convs, ≥6 turns, 16k ctx → `results/context_fatigue/wildchat_homogeneity/`.

## Prior context
- WildChat vs DDXPlus result (format-confounded, weaker): `WILDCHAT_DYNAMICS.md` —
  entropy ratio ≈1.0 (vs DDXPlus 3–4×); current-neglect mostly a depth-0 artifact.
- Parked: route-dependent safety sweep (`tasks/route_dependent_safety.md`), step-0 done
  (Qwen refusal headroom 0.983), trainer dose-overrides added. Resume after this.
