# RQ4: Lookahead Planning and Commitment Detection

**Branch:** `psycoplankton/rq4-lookahead-planning`  
**Status:** Phase 1 codebase complete, ready for GPU experiments  
**Author:** psycoplankton (SPAR project with Justin Shenk)

## Research Question

Do autoregressive models plan ahead by committing to future structure (e.g., rhyme targets, return types) before those tokens are produced, and can we detect and intervene on that commitment from activations?

## What This Branch Contains

### Datasets (`src/lookahead/datasets/`)

Three task types with ground-truth planning targets:

| Task | What it tests | Example |
|------|--------------|---------|
| **Rhyme** | Does the model commit to a rhyme word before generating line 2? | "Roses are red, \n" → expects rhyme with "red" |
| **Acrostic** | As letters are revealed, when does the model lock in the word? | "Write an acrostic for DREAM" → next letter must be D |
| **Code return** | Does the model commit to a return type at the signature? | `def add(a, b) -> int:\n    ` → committed to int |

Each dataset includes:
- **Contrastive pairs** — same prompt structure, different targets (for computing steering vectors)
- **Controls** — non-planning variants (for baseline comparison)
- **Minimal pairs** — differ only in the planning-relevant feature

### Probing Pipeline (`src/lookahead/probing/`)

- **`activation_extraction.py`** — Extracts residual stream activations at *every* token position (not just last token). Also implements logit-lens analysis.
- **`commitment_probes.py`** — The core analysis. Trains a linear probe at each position to predict the future target. Key features:
  - **Out-of-fold predictions** for commitment curves (prevents overfitting artifacts)
  - **Shuffled-label baseline** (most important control)
  - **Permutation tests** for statistical significance
  - **Commitment point detection** with stability requirements (single spikes don't count)

### Causal Patching (`src/lookahead/patching/`)

- **`causal_patching.py`** — Necessity and sufficiency tests:
  - *Necessity:* Patch activations at commitment point → does output change?
  - *Sufficiency:* Inject commitment vector → does it steer to new target?
  - *Timing comparison:* Pre- vs at- vs post-commitment patching effectiveness

### Scripts (`scripts/lookahead/`)

- **`run_phase1_commitment_curves.py`** — Main experiment runner. Generates dataset, extracts activations, trains probes, computes commitment curves, runs baselines.

### Tests (`tests/lookahead/`)

- **`test_datasets.py`** — 25 tests validating dataset integrity (IDs, labels, structure, roundtrips)
- **`test_probing_pipeline.py`** — 13 tests using synthetic data with known commitment points:
  - Verifies probes detect signal at injection point
  - Verifies no-signal → chance accuracy
  - Verifies shuffled baseline stays at chance
  - Verifies commitment point detection accuracy
  - Verifies permutation test calibration

## Methodological Safeguards

This is designed to produce results that are NOT artifacts:

1. **Out-of-fold predictions** — Commitment curves use cross-validated held-out predictions, never in-sample. This prevents overfitting from creating illusory early commitment.

2. **Shuffled-label baseline** — Every experiment runs the same probes with randomized labels. If shuffled accuracy ≈ real accuracy, the signal is structural, not planning.

3. **Permutation test** — Formal p-values at each position via 1000-permutation null distribution.

4. **Stability window** — Commitment detection requires sustained above-threshold confidence (not single spikes).

5. **Contrastive pairs** — Same prompt structure, different targets. Controls for everything except the planning signal.

6. **Non-rhyming controls** — Same syntactic structure without planning constraint. Should show flat commitment curves.

## Quick Start

```bash
# Run all tests (no GPU needed)
python -m pytest tests/lookahead/ -v

# Generate dataset only
python -c "
from src.lookahead.datasets.rhyme import generate_rhyme_dataset, save_dataset
examples = generate_rhyme_dataset(n_per_rhyme_set=5)
save_dataset(examples, 'data/lookahead/raw/rhyme_dataset.json')
print(f'Generated {len(examples)} examples')
"

# Run Phase 1 experiment (needs GPU + TransformerLens)
python scripts/lookahead/run_phase1_commitment_curves.py \
    --model gpt2 \
    --task rhyme \
    --device cuda \
    --output results/lookahead/phase1_rhyme
```

## Experiment Plan

### Phase 1: Commitment Curves (THIS BRANCH — weeks 1-2)
- [x] Dataset generators for 3 task types
- [x] Activation extraction at all positions
- [x] Commitment probe training with OOF
- [x] Shuffled baseline + permutation tests
- [x] Synthetic data validation (38/38 tests)
- [ ] **Run on GPT-2** ← next step
- [ ] **Run on 7B model** (if GPT-2 shows signal)
- [ ] Commitment curve plots

### Phase 2: Causal Verification (weeks 3-4)
- [x] Necessity patching infrastructure
- [x] Sufficiency (steering vector) infrastructure
- [x] Timing comparison framework
- [ ] Run patching experiments at identified commitment points
- [ ] Pre- vs post-commitment intervention comparison

### Phase 3: Cross-Task Generalization (weeks 4-5)
- [ ] Train probe on rhyme → test on code return type
- [ ] Train probe on code → test on acrostic
- [ ] Transfer accuracy matrix across task types

### Phase 4: Intervention Demo (weeks 5-6)
- [ ] Real-time commitment detection monitor
- [ ] Code generation: detect insecure pattern commitment → steer
- [ ] Paper section draft

## File Structure

```
src/lookahead/
├── datasets/
│   ├── rhyme.py            # Couplet completion (15 rhyme families)
│   ├── acrostic.py         # Progressive letter reveal
│   └── code_return.py      # Function return type commitment
├── probing/
│   ├── activation_extraction.py  # All-position residual stream extraction
│   └── commitment_probes.py      # OOF probes + baselines + permutation tests
├── patching/
│   └── causal_patching.py        # Necessity/sufficiency/timing
├── analysis/               # (TBD: plotting, aggregation)
└── utils/
    └── types.py            # Core data structures
```

## Key Design Decision: OOF vs In-Sample

Early testing revealed that in-sample probe predictions create false early-commitment artifacts (the probe memorizes noise patterns in small datasets). Switching to `cross_val_predict` with out-of-fold predictions eliminated this. The synthetic data test (`test_commitment_point_detected_near_true_position`) validates that detected commitment points match the true injection point within ±3 tokens.

## Dependencies

- `transformer-lens` — Hook-based activation access
- `torch` — GPU computation
- `scikit-learn` — Linear probes, cross-validation
- `numpy`, `tqdm`, `pandas` — Standard tooling

## Related

- Parent issue: DRAFT: RQ4 in justinshenk/temporal-awareness
- Existing infrastructure reused: `src/activation_patching/`, `src/inference/backends/backend_transformerlens.py`
- External refs: Future Lens (Pal et al. 2023), Planning in LLMs (Valmeekam et al. 2023)
