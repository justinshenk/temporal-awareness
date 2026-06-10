# Where in variance-space does the LoRA's capability live? (oracle PCA-band + complement injection, L20)

**Setup.** Llama-2-7B + metamath-LoRA, GSM8K. δ = LoRA−base residual at L20 on the base trajectory,
PCA'd over 25,424 CoT tokens (100 train problems). For each band we run the **lockstep oracle** —
inject `a + Π_band(δ_true)` at L20 every decode step — and measure closed-loop recovery on the
34-problem base-fails/LoRA-solves contrast set (20 used). Same apparatus as the lockstep oracle
(`2026-06-09-lockstep-oracle.md`); only the injected subspace changes.

## Result: capability is a wide, low-variance band — and the head is necessary too

| band | k | δ-energy | recovery |
|---|---:|---:|---:|
| top8 | 8 | 0.36 | 0.00 |
| top64 | 64 | 0.55 | 0.00 |
| top256 | 256 | 0.71 | 0.05 |
| **top512** | 512 | 0.80 | **0.45** |
| top1024 | 1024 | 0.88 | 0.65 |
| full | 4096 | 1.00 | 0.75 |
| **δ−top8** (complement) | 4088 | 0.64 | **0.00** |
| **δ−top64** (complement) | 4032 | 0.45 | **0.00** |

Figure: `results/figures/pca_band_recovery_L20.png`.

### 1. Energy and capability are dissociated (variance is the wrong objective)
The top-64 directions carry **55% of δ's energy but 0% of the recovery**. Recovery only switches on
between k=256 (0.05) and k=512 (0.45), in the moderate-variance band that an MSE / variance-chasing
fit structurally deprioritizes. This is the quantitative reason a ridge or MLP map (which chase
variance) recover ~0 while the oracle recovers 0.75: **the lever and the signal sit in different
parts of the spectrum.** It also answers the open 512 question — top256→top512 is the cliff (0.05 →
0.45); the "some recovery" previously seen at 256 was a single problem.

### 2. The complement injections: it's a fidelity threshold, not a low-variance locus
The tempting reading of (1) is "capability lives in the low-variance tail." The complement
injections refute it. Injecting δ with **only the top-8 directions removed** (keep 4088 dirs, 64%
of energy) recovers **0.00**; removing the top-64 (keep 4032 dirs, 45% energy) also **0.00**. So:

- top-8 alone = 0.00 **and** δ−top8 = 0.00 → the head is **necessary but not sufficient**.
- The same holds for the moderate band (top64 alone = 0; needs ≥512 to fire).

Capability is **conjunctive across a wide band**: it appears only when both the high-variance head
*and* the moderate-variance directions out to ~512 are reproduced faithfully. Dropping even a sliver
of the head destroys it. This is a **fidelity threshold**, not a small privileged subspace — which is
exactly why no low-rank or partial feed-forward map has cleared it.

## Consequence / next
Two readings were set up as maximally separated: (A) δ−top64 ≈ 0.75 (inert head) vs (B) δ−top64 ≈ 0
(fidelity threshold). **Result is B, in its strong form (even δ−top8 = 0).** The salvage is no longer
"reweight the variance" — it's whether the causal directions are **low-rank but low-variance**,
findable by a *task-loss* subspace search where variance fails. That is the DAS experiment
(`das_subspace_gsm8k.py`): learn an orthonormal R by behavioral loss, inject `a + Π_R(δ)`, compare
DAS-R@r vs PCA-top-r at matched rank. If DAS-R@64 ≫ PCA-top64 (=0), the directions are low-rank; if
DAS-R also needs r≈512, it's a genuine wide-band fidelity floor and the honest next move is the
task-loss-trained *map* (LoReFT upper bound, learning the injected value, not just the subspace).

## Reproduce
`scripts/attribution/lockstep_pca_band.py` (`--bands full,top8,top64,top256,top512,top1024,tail64`,
plus `--bands tail8,tail64` for the complements); plot via `scripts/attribution/plot_pca_band.py`.
JSON: `results/attribution/lockstep_pca_band_L20.json`. δ-PCA in `src/probes/attribution/delta_subspace.py`;
injection in `lockstep_oracle.projected_injection`.
