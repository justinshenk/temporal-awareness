# RQ4: Lookahead Planning Detection in Language Models

**Branch:** `psycoplankton/rq4-lookahead-planning`  
**Author:** Tejas Dahiya (tdahiya2@wisc.edu, UW-Madison)  
**Supervisor:** Justin Shenk  
**Project:** SPAR — Temporal Awareness in LLMs  

---

## Summary

We investigate whether language models plan ahead by committing to future structure before producing tokens. Across **11 models**, **4 architecture families**, and **8+ baselines**, we find **no evidence of lookahead planning** (124M–7B parameters). What appears as planning signal (probe accuracy 80–91%) is fully explained by a **name+params baseline** (87–95%), which beats the probe at every layer in every model (0/66 significant after FDR correction).

## Core Discovery

```
Naive probe:       80-91%  →  "Planning detected!"
Name-only:         63-75%  →  Name explains most signal
Name+Params:       84-95%  →  Explains ALL signal (beats probe everywhere)
Gap (Probe - N+P): -2% to -11%  →  Zero residual planning
```

## Results

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

*0/66 layer-model combinations significant after Benjamini-Hochberg FDR correction*

## Experiments

### 1. Main Probing (500 examples × 11 models)
Predict return type (int/str/bool/list/float) from residual stream activations. Name+params baseline explains all signal.

### 2. Misleading Names (50 examples × 3 models)
Functions where name contradicts params (e.g., `def greet(numbers):`). Models follow params 56%, name 22–34%. Surface matching, not understanding.

### 3. Fair Subset Analysis (3 models)
Train on easy examples, test on hard. Probe: 47–74% vs name+params: 86–93%. Probe cannot generalize.

### 4. Generation-Time Commitment (20 steps × 4 models)
Probe at each autoregressive step. Accuracy **decays** during generation (CodeLlama: 95.3% → 62.9%). Models forget, not plan.

### 5. Fixed-Position Probing (2 models)
Probe at `def` token during generation: ~25% (chance). Type info localized at name+params tokens only.

### 6. Mean-Pooling Confound (3 models)
Mean-pooled probe = name+params (identical input). Original gap was information quantity, not planning.

### 7. Acrostic Behavioral (6 models)
Can models produce acrostics? 4–20% (near chance). Cannot do structural planning.

### 8. Additional
FP32 steering (10 models), causal patching (GPT-2), cross-task transfer (code→rhyme: fails), base vs instruct, nonsense names.

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

```
src/lookahead/
├── datasets/           # code_return.py, rhyme.py, acrostic.py
├── probing/            # activation_extraction, baselines, probes
├── patching/           # causal_patching.py
└── utils/              # types.py (PlanningExample, TaskType)

tests/lookahead/        # 49/49 passing

scripts/lookahead/
├── experiments/        # All GPU experiment scripts
├── run_comprehensive.py
└── run_phase1_commitment_curves.py

results/lookahead/
├── complete/           # Initial runs (150 examples)
└── final/              # Final runs (500 examples)
    ├── *_final.json    # Per-model results
    ├── gentime_*.json  # Generation-time curves
    ├── fixes_*.json    # Mean-pool + def-position
    ├── figures/        # 6 figures (PNG + PDF)
    └── stats/          # FDR + Spearman JSON
```

## Experiment Scripts

| Script | Purpose | GPU |
|--------|---------|-----|
| `run_rq4_final.py` | Main 500ex, 11 models | ~10h |
| `run_rq4_gentime.py` | Generation-time curves, 4 models | ~4h |
| `run_rq4_critfixes.py` | Mean-pool + def-position fixes | ~2h |
| `run_rq4_acrostic.py` | Acrostic behavioral, 6 models | ~2h |
| `run_rq4_subset_v2.py` | Fair subset analysis, 3 models | ~20m |
| `run_rq4_analysis.py` | Stats + figures (no GPU) | ~1m |

## Reproduction

```bash
git clone https://github.com/justinshenk/temporal-awareness.git
cd temporal-awareness && git checkout psycoplankton/rq4-lookahead-planning
pip install "transformer-lens==2.11.0" "transformers==4.44.0" scikit-learn scipy matplotlib
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 scripts/lookahead/experiments/run_rq4_final.py  # GPU required
python3 scripts/lookahead/experiments/run_rq4_analysis.py  # No GPU
python3 -m pytest tests/lookahead/ -v
```

## Compute

GPU: RTX 5070 Ti 16GB (Vast.ai, $0.115/hr). Total: ~$3 / ~26 hours.

## Paper

**Title:** *Knowledge Without Planning: How Probing Baselines Explain Away Apparent Lookahead in Code Generation*

**Contributions:** (1) Baseline staircase protocol as reusable methodology, (2) Empirical evidence across 11 models, (3) Generation-time decay finding, (4) Misleading names diagnostic.

## Contact

Tejas Dahiya — tdahiya2@wisc.edu — UW-Madison  
Justin Shenk — github.com/justinshenk
