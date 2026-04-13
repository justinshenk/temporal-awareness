# RQ4: Lookahead Planning Detection in Language Models

**Branch:** `psycoplankton/rq4-lookahead-planning`  
**Author:** Tejas Dahiya (tdahiya2@wisc.edu, UW-Madison)  
**Supervisor:** Justin Shenk  
**Project:** SPAR — Temporal Awareness in LLMs  

---

## Summary

We investigate whether language models plan ahead by committing to future structure before producing tokens. Through experiments across **12 models** (including GPT-J-6B), **4 architecture families**, and **8+ baselines**, we find:

1. **Code domain (11 models, 124M–7B):** No evidence of planning. A `name+params` baseline (87–95%) beats the probe (80–91%) at every layer in every model (0/66 significant after FDR correction).

2. **Natural text (GPT-J-6B):** Probe genuinely beats fair embedding baselines (K=1: +25%, K=5: +5%), replicating Pal et al.'s Future Lens. However, the signal decays rapidly with distance — statistical continuation, not long-range planning.

3. **Methodology:** We propose a **baseline staircase protocol** that distinguishes domains where model processing adds information from domains where surface features explain everything.

## Code Domain Results

| Model | Params | Behavioral | Probe | Name+Params | Gap |
|-------|--------|-----------|-------|-------------|-----|
| GPT-2 Small | 124M | 0.0% | 80.3% | 84.6% | −3.9% |
| GPT-2 Medium | 345M | 0.0% | 78.7% | 87.6% | −8.9% |
| GPT-2 XL | 1.5B | 0.0% | 80.7% | 87.4% | −5.5% |
| Pythia-410M | 410M | 26.7% | 80.9% | 89.2% | −8.5% |
| Pythia-1B | 1.0B | 32.0% | 82.3% | 90.1% | −7.8% |
| Pythia-1.4B | 1.4B | 38.7% | 83.2% | 90.0% | −6.8% |
| Pythia-2.8B | 2.8B | 32.7% | 83.2% | 91.1% | −7.3% |
| SantaCoder | 1.1B | 41.3% | 91.1% | 93.1% | −2.0% |
| CodeLlama-7B | 7B | 42.7% | 88.6% | 94.5% | −5.9% |
| Llama-3.2-1B | 1.2B | 34.0% | 86.6% | 93.7% | −7.1% |
| Llama-3.2-1B-Inst | 1.2B | 34.0% | 83.6% | 94.3% | −10.7% |

*0/66 significant after Benjamini-Hochberg FDR correction*

## Future Lens Replication (GPT-J-6B)

Predicting exact token at position N+K from intermediate layer activations. Fair baselines use same-dimensionality embeddings. Proper train/test split (no data leakage).

| K | Chance | Trigram | Context Emb | Probe | Gap (P−Ctx) |
|---|--------|--------|-------------|-------|-------------|
| 1 | 11.8% | 16.0% | 59.0% | **84.4%** | **+25.4%** |
| 2 | 11.9% | 14.8% | 49.4% | **65.5%** | **+16.1%** |
| 3 | 12.2% | 11.1% | 45.6% | **53.7%** | **+8.1%** |
| 5 | 12.2% | 11.0% | 41.8% | **46.9%** | **+5.0%** |

The probe genuinely beats fair baselines (replicating Pal et al.), but the gap shrinks from +25% to +5% with distance — short-range statistical continuation, not long-range planning.

## Experiments

### Code Domain
1. **Main Probing** (500 examples × 11 models) — Name+params explains all signal
2. **Misleading Names** (50 examples × 3 models) — Models follow params 56%, name 22–34%
3. **Fair Subset Analysis** (3 models) — Probe 47–74% vs name+params 86–93% on hard examples
4. **Generation-Time Commitment** (20 steps × 4 models) — Probe decays during generation (CodeLlama: 95.3% → 62.9%)
5. **Fixed-Position Probing** (2 models) — `def` token has ~25% accuracy (chance)
6. **Mean-Pooling Confound** (3 models) — Confirms original gap was information quantity
7. **Acrostic Behavioral** (6 models) — 4–20% accuracy (near chance)
8. **Additional:** FP32 steering (10 models), causal patching, cross-task transfer, base vs instruct

### Natural Text (Future Lens)
9. **Future Lens Replication** (GPT-J-6B) — Probe beats embedding baselines, confirms Pal et al.
10. **Distance Decay** — Gap shrinks from +25% (K=1) to +5% (K=5)
11. **N-gram Baseline** — Trigram without data leakage: only 11–16% (no shortcut)

## Statistical Analysis

**FDR (Benjamini-Hochberg):** Probe vs N+P: 0/66 significant. Probe vs name-only: 55/66 significant.

**Spearman:** Behavioral vs Probe ρ=+0.940 (p<0.001). Behavioral vs Gap ρ=−0.078 (p=0.819). No link between capability and planning.

## Hypotheses

| H# | Hypothesis | Outcome |
|----|-----------|---------|
| H1 | Commit ≥3 tokens before | Signal exists but = surface features |
| H2 | Mid-to-late layers | Best probe at mid-late but = name+params |
| H3 | Patching flips >50% | Sufficiency confirmed but surface cues |
| H4 | Pre-commitment easier | Steering works but reflects name/params |
| H5 | Cross-task transfer >60% | **Failed** — random on rhyme |

## Repo Structure
src/lookahead/
├── datasets/           # code_return.py, rhyme.py, acrostic.py
├── probing/            # activation_extraction, baselines, probes
├── patching/           # causal_patching.py
└── utils/              # types.py (PlanningExample, TaskType)
tests/lookahead/        # 49/49 passing
scripts/lookahead/
├── experiments/        # All experiment scripts (16 scripts)
├── run_comprehensive.py
└── run_phase1_commitment_curves.py
results/lookahead/
├── complete/           # Initial runs (150 examples)
└── final/              # Final runs (500 examples)
├── final.json    # Per-model results (11 models)
├── gentime.json  # Generation-time curves
├── fixes_.json    # Mean-pool + def-position
├── future_lens_.json  # Future Lens replication
├── figures/        # 6 figures (PNG + PDF)
└── stats/          # FDR + Spearman JSON

## Key Experiment Scripts

| Script | Purpose | GPU |
|--------|---------|-----|
| `run_rq4_final.py` | Main 500ex, 11 models | ~10h |
| `run_rq4_gentime.py` | Generation-time curves | ~4h |
| `run_rq4_critfixes.py` | Mean-pool + def-position | ~2h |
| `run_rq4_futurelens_v4_fix.py` | Future Lens (GPT-J-6B, no leakage) | ~1h |
| `run_rq4_acrostic.py` | Acrostic behavioral | ~2h |
| `run_rq4_analysis.py` | Stats + figures (no GPU) | ~1m |

## Reproduction

```bash
git clone https://github.com/justinshenk/temporal-awareness.git
cd temporal-awareness && git checkout psycoplankton/rq4-lookahead-planning
pip install "transformer-lens==2.11.0" "transformers==4.44.0" scikit-learn scipy matplotlib
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 scripts/lookahead/experiments/run_rq4_final.py           # Code (GPU, ~10h)
python3 scripts/lookahead/experiments/run_rq4_futurelens_v4_fix.py  # Future Lens (GPU, ~1h)
python3 scripts/lookahead/experiments/run_rq4_analysis.py        # Stats + figures (no GPU)
python3 -m pytest tests/lookahead/ -v
```

## Compute

| Resource | Details |
|----------|---------|
| Code experiments | RTX 5070 Ti 16GB, ~$3 total |
| Future Lens | RTX 5090 32GB, ~$1 total |
| Total cost | **~$4** |

## Paper

**Title:** *Knowledge Without Planning: How Probing Baselines Explain Away Apparent Lookahead in Code Generation*

**Contributions:**
1. Baseline staircase protocol that distinguishes genuine model processing from surface features
2. Empirical evidence across 11 models that code probing signal = surface features
3. Future Lens replication showing short-range statistical continuation, not planning
4. Generation-time decay: models lose structural info during generation

## Contact

Tejas Dahiya — tdahiya2@wisc.edu — UW-Madison  
Justin Shenk — github.com/justinshenk
