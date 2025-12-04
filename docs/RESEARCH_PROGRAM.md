# Temporal Grounding: Research Overview

**Investigating how LLMs represent and reason about time**

---

## Overview

Without understanding how models represent time, we can't ensure AI systems reason responsibly about decisions that matter across time.

This research investigates how time horizons are encoded in LLMs, whether these encodings are robust, and what this means for alignment and ethics.

---

## Research Tracks

The program has two parallel tracks:

### Track A: Probe Infrastructure (Sequential)

```
#1 Dataset validation → Probe validation → #2 SPD → #4 SAEs
```

| Step | Issue | Status | Depends On |
|------|-------|--------|------------|
| 1 | #1 Dataset validation | ⬜ Pending | - |
| 2 | Probe validation | ⚠️ Preliminary | #1 |
| 3 | #2 SPD decomposition | 🔄 In progress | Probe validation |
| 4 | #4 Sparse autoencoders | ⬜ | #2 |

### Track B: Experiments (Parallelizable)

Each experiment requires its own purpose-built dataset, independent of Track A.

| Issue | Title | Dataset Needed |
|-------|-------|----------------|
| #3 | Explicit temporal context | Ground-truth temporal distributions |
| #5 | Eval awareness detection | Scenarios with hidden long-term planning |
| #6 | Uncertainty scaling | Prompts at varying time horizons |
| #7 | Consistency failures | Near/far matched pairs |
| #8 | Value alignment | User profiles + temporal scenarios |
| #9 | Register bias | Formal/casual matched pairs |
| #10 | Coconut latent reasoning | Temporal decision tasks |
| #11 | Active inference/phenomenology | Structured temporal cognition prompts |
| #12 | Temporal defabrication (MPE) | Recursive generation + contemplative prompts |
| #13 | Intertemporal preference framework | Reward/time tradeoff scenarios by domain |

---

## Research Thrusts

### Thrust 1: Foundations

**Core Question**: Do probes detect genuine temporal semantics, or lexical shortcuts?

- **#1 Dataset validation**: Align datasets with CAA methodology
- **#3 Explicit temporal context**: Can models output calibrated temporal distributions?
- **Probe validation**: Train on keywords, test on semantic-only data

### Thrust 2: Mechanistic Interpretability

**Core Question**: What circuits and features implement temporal reasoning?

- **#2 SPD decomposition**: Identify temporal components via Stochastic Parameter Decomposition
- **#4 Sparse autoencoders**: Find monosemantic temporal neurons

### Thrust 3: Robustness & Failure Modes

**Core Question**: When do temporal representations fail, and what does this reveal?

| Issue | Question |
|-------|----------|
| #5 Eval awareness | Can temporal signatures detect hidden long-term planning? |
| #6 Uncertainty scaling | Do hedges increase with temporal distance? |
| #7 Consistency failures | Do models treat near/far futures inconsistently? |
| #8 Value alignment | Do temporal preferences reflect user values? |
| #9 Register bias | Does writing style shift advice horizons? |

### Thrust 4: Theoretical Bridges

**Core Question**: What frameworks from cognitive science illuminate LLM temporal cognition?

| Issue | Framework | Key Concept |
|-------|-----------|-------------|
| #10 | Coconut | Continuous latent space enables richer temporal gradients |
| #11 | Active inference | Precision decay, temporal depth, counterfactual exploration |
| #11 | Phenomenology | Retention/protention, past/present/future modes |
| #12 | MPE/Defabrication | Recursive generation causes temporal representation collapse |
| #13 | Intertemporal preference | Reward-based discounting, internal vs. stated horizon |

---

## Cross-Cutting Themes

### A. Internal vs. External Alignment
Does what the model "thinks" match what it says?
- #5: Internal planning vs. stated focus
- #6: Internal confidence vs. verbal hedging
- #12: Internal differentiation vs. fluent output

### B. Robustness to Surface Features
Are temporal representations robust to surface variation?
- #7: Same question, different framing
- #9: Same question, different register
- Probe validation: Same meaning, no keywords

### C. Calibration and Uncertainty
Do models appropriately quantify temporal uncertainty?
- #3: Calibrated probability distributions
- #6: Hedges scaling with distance
- #11: Precision decay predictions

### D. Failure Modes as Probes
What do failures reveal about representations?
- #7: Inconsistencies expose shallow encoding
- #9: Register bias exposes training correlations
- #12: Defabrication exposes representational fragility

---

## Issue Index

| # | Title | Track | Thrust | Status |
|---|-------|-------|--------|--------|
| 1 | Dataset validation | A | Foundations | ⬜ Pending |
| 2 | SPD decomposition | A | Mechanisms | 🔄 In progress |
| 3 | Explicit temporal context | B | Foundations | ⬜ |
| 4 | Sparse autoencoders | A | Mechanisms | ⬜ |
| 5 | Eval awareness detection | B | Robustness | ⬜ |
| 6 | Uncertainty scaling | B | Robustness | ⬜ |
| 7 | Consistency failures | B | Robustness | ⬜ |
| 8 | Value alignment | B | Robustness | ⬜ |
| 9 | Register bias | B | Robustness | ⬜ |
| 10 | Coconut latent reasoning | B | Theory | ⬜ |
| 11 | Active inference/phenomenology | B | Theory | ⬜ |
| 12 | Temporal defabrication (MPE) | B | Theory | ⬜ |
| 13 | Intertemporal preference framework | B | Theory | ⬜ |
| - | Probe validation | A | Foundations | ⚠️ Preliminary |

---

## Links

- GitHub: [temporal-awareness](https://github.com/justinshenk/temporal-awareness)
- Issues: https://github.com/justinshenk/temporal-awareness/issues
