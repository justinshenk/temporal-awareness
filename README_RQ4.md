# RQ4: Lookahead Planning Detection in Language Models

**Branch:** `psycoplankton/rq4-lookahead-planning`  
**Author:** Tejas Dahiya (tdahiya2@wisc.edu, UW-Madison)  
**Supervisor:** Justin Shenk  
**Project:** SPAR — Temporal Awareness in LLMs  

---

## Summary

We investigate whether language models plan ahead by committing to future structure before producing tokens. Through experiments across **14 models**, **5 architecture families**, **5 text domains**, and **8+ baselines**, we find:

1. **Code domain (11 models, 124M–7B):** No evidence of planning. A `name+params` baseline (87–95%) beats the probe (80–91%) at every layer in every model. 0/66 significant after FDR correction.

2. **Natural text (3 models with bootstrap CIs):** Probe genuinely beats fair embedding baselines, replicating Pal et al.'s Future Lens. But the gap decays rapidly with prediction distance (K=1: +19–51%, K=5: near zero) — short-range statistical continuation, not long-range planning.

3. **Intermediate domains (5 domains):** Chain-of-thought math maintains the strongest planning-like signal (+28% at K=3). Poetry collapses to zero. The baseline staircase is a general diagnostic tool.

4. **Methodology:** We propose a **baseline staircase protocol** extending Hewitt & Liang (2019) control tasks from binary (random vs real) to progressive (chance → BoW → name → name+params).

---

## Operationalization of Planning

We define "lookahead planning" as internal representations satisfying three criteria:

- **Beyond-surface (B):** Probe accuracy significantly exceeds the best surface-feature baseline after FDR correction
- **Persistence (P):** Signal maintains or strengthens during autoregressive generation
- **Transfer (T):** Representation transfers to structurally analogous tasks

| Domain | Criterion B | Criterion P | Criterion T | Verdict |
|--------|------------|------------|------------|---------|
| Code return types | ✗ (0/66 sig.) | ✗ (decays 95→63%) | ✗ (fails on rhyme) | **No planning** |
| Natural text (K=1) | ✓ (+19–51%) | ✗ (decays to ~0% at K=5) | Untested | **Statistical continuation** |
| Chain-of-thought | ✓ (+28% at K=3) | Partial (maintains gap) | Untested | **Strongest candidate** |

---

## Part I: Code Domain Results

### Core Finding: Baseline Staircase

```
Naive probe:       80-91%  →  "Planning detected!"
Name-only:         63-75%  →  Name explains most signal
Name+Params:       84-95%  →  Explains ALL signal (beats probe everywhere)
Gap (Probe - N+P): -2% to -11%  →  Zero residual planning
```

### Results Table (11 Models + 3 Independent Qwen Replications)

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
| *Qwen2.5-1.5B* | *1.5B* | *41.6%* | *86.6%* | *93.7%* | *−7.1%* |
| *Qwen2.5-7B* | *7B* | — | — | — | *Same pattern* |
| *Qwen2.5-14B* | *14B* | — | — | — | *Same pattern* |

*Qwen models independently replicated by Justin Shenk. 0/66 layer-model combinations significant after Benjamini-Hochberg FDR correction.*

### Statistical Analysis

**FDR (Benjamini-Hochberg):** Probe vs name+params: **0/66 significant.** Probe vs name-only: 55/66 significant.

**Spearman correlations (the cleanest thesis statement):**
- Behavioral accuracy vs Probe accuracy: **ρ = +0.940** (p < 0.001) — capability predicts probe signal
- Behavioral accuracy vs Gap (probe − N+P): **ρ = −0.078** (p = 0.819) — capability does NOT predict planning

---

## Part II: Future Lens Replication (3 Models, Bootstrap CIs)

Predicting exact token at position N+K from layer activations. Fair baselines use same-dimensionality embeddings. Proper 50/50 train/test split, 300 bootstrap samples, 3 probe seeds.

### Multi-Model K Decay (Gap = Probe − Context Embedding)

| Model | K=1 Gap | K=2 Gap | K=3 Gap | K=5 Gap |
|-------|---------|---------|---------|---------|
| Pythia-2.8B | **+18.7%** [.873,.940]* | +10.3% | +6.0% | −1.2% |
| Qwen2.5-1.5B | **+51.3%** [.832,.910]* | +24.5% | +11.1% | +2.7% |
| GPT-J-6B | **+30.9%** [.830,.896]* | +15.8% | +5.2% | +4.9% |

*\* = statistically significant (probe CI_lo > context CI_hi)*

**Finding:** K decay is **universal across architectures**. Models encode short-range statistical continuation that dissipates within ~5 tokens.

### Comparison to Pal et al.'s Actual Method

| Method | K=1 | K=3 | K=5 |
|--------|-----|-----|-----|
| Pal et al. linear mapping (PCA source) | 4.8% | 4.6% | 4.1% |
| Pal et al. linear mapping (full 4096-dim) | 2.0% | 2.7% | 2.5% |
| **Our classification probe** | **85.4%** | **44.4%** | **34.0%** |

Linear hidden-state mapping requires orders of magnitude more training data. Our classification probe is more practical for baseline staircase comparisons.

---

## Part III: Intermediate Domains (5 Domains, GPT-J-6B)

### K=3 — The Diagnostic Test

| Domain | Context Emb | Probe | Gap | Interpretation |
|--------|-------------|-------|-----|----------------|
| **Chain-of-thought** | 59.3% | 87.3% | **+28.0%** | Strongest planning-like signal |
| Free prose | 44.5% | 63.4% | +18.9% | Moderate continuation |
| Structured prose | 62.0% | 73.0% | +11.0% | Template following |
| Code | 74.5% | 85.5% | +11.0% | Syntactic patterns |
| **Poetry** | 42.0% | 40.8% | **−1.2%** | Collapses to zero |

Chain-of-thought maintains the gap where poetry collapses. The staircase distinguishes domains where model processing adds genuine multi-step information.

---

## Part IV: Supporting Experiments

### Misleading Names (50 examples × 3 models)
Functions where name contradicts params (e.g., `def greet(numbers):`). Models follow params 56%, name 22–34%.

### Fair Subset Analysis (3 models)
Train easy / test hard. Probe: 47–74% vs name+params: 86–93%.

### Generation-Time Commitment (20 steps × 4 models)
Probe accuracy **decays** during generation (CodeLlama: 95.3% → 62.9%). Fails Persistence criterion.

### Fixed-Position Probing (100 signatures, GPT-J-6B)

| Position | What it is | Accuracy |
|----------|-----------|----------|
| pos=0 (BOS) | Start token | 22% (chance) |
| pos=1 (def) | Keyword | 22% (chance) |
| pos=2 (name) | Function name | **81%** |
| pos=3 | After name | 80% |
| pos=5 (params) | Parameters | 80% |
| pos=last | Full signature | **94%** |

Info lives in the function name — exactly where surface features predict.

### Mean-Pooling Confound (3 models)
Mean-pooled probe = name+params accuracy. Original gap was information quantity, not planning.

### Acrostic Behavioral (6 models)
4–20% accuracy (near chance). Models cannot plan structurally.

---

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
    ├── *_final.json              # Per-model code results (11 models)
    ├── gentime_*.json            # Generation-time curves
    ├── fixes_*.json              # Mean-pool + def-position
    ├── future_lens_multimodel.json  # 3-model Future Lens with CIs
    ├── future_lens_v4_fixed.json    # GPT-J with proper train/test
    ├── future_lens_fixed_method.json # Pal et al. actual method
    ├── intermediate_domains.json    # 5-domain experiment
    ├── reviewer_fixes.json          # Larger scale (100 prompts)
    ├── fix5_100sigs.json            # 100-sig fixed positions
    ├── figures/                     # Paper figures (PNG + PDF)
    └── stats/                       # FDR + Spearman JSON
```

---

## Key Experiment Scripts

| Script | Purpose | GPU | Time |
|--------|---------|-----|------|
| `run_rq4_final.py` | Main 500ex, 11 models | Yes | ~10h |
| `run_rq4_gentime.py` | Generation-time curves | Yes | ~4h |
| `run_rq4_futurelens_multimodel.py` | Future Lens (3 models, CIs) | Yes | ~5h |
| `run_rq4_domains.py` | Intermediate domains | Yes | ~1h |
| `run_rq4_fl_fixed.py` | Pal et al. method comparison | Yes | ~1h |
| `run_fix5_fast.py` | 100-sig fixed positions | Yes | ~10m |
| `run_rq4_reviewer_fixes.py` | Larger scale (100 prompts) | Yes | ~3h |
| `run_rq4_analysis.py` | Stats + figures (no GPU) | No | ~1m |

---

## Reproduction

```bash
git clone https://github.com/justinshenk/temporal-awareness.git
cd temporal-awareness && git checkout psycoplankton/rq4-lookahead-planning
pip install "transformer-lens==2.11.0" "transformers==4.44.0" scikit-learn scipy matplotlib
export PYTHONPATH=$(pwd):$PYTHONPATH

# Code domain (GPU, ~10h)
python3 scripts/lookahead/experiments/run_rq4_final.py

# Future Lens with CIs (GPU, ~5h)
python3 run_rq4_futurelens_multimodel.py

# Intermediate domains (GPU, ~1h)
python3 run_rq4_domains.py

# Stats + figures (no GPU)
python3 scripts/lookahead/experiments/run_rq4_analysis.py

# Tests
python3 -m pytest tests/lookahead/ -v
```

---

## Related Work

- **Hewitt & Liang (2019)** — Control tasks for probes. We extend from binary to progressive staircase.
- **Pal et al. (2023)** — Future Lens. We replicate with fair baselines, show universal K decay.
- **Belrose et al. (2023)** — Tuned Lens. Our layer-wise analysis aligns with iterative refinement.

---

## Paper

**Title:** *Knowledge Without Planning: How Probing Baselines Explain Away Apparent Lookahead in Code Generation*

**Contributions:**
1. Baseline staircase protocol extending Hewitt & Liang (2019)
2. Code probing signal fully explained by surface features (14 models, 5 families)
3. Future Lens replication: universal K decay across 3 architectures
4. Intermediate domains: chain-of-thought maintains signal, poetry collapses
5. Generation-time decay refutes commitment hypothesis

---

## Contact

Tejas Dahiya — tdahiya2@wisc.edu — UW-Madison  
Justin Shenk — github.com/justinshenk
