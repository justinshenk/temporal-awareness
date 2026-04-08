## Phase 1: Behavioral Metrics & SAE Feature Stability

### Summary
- Complete Phase 1 behavioral metrics pipeline: 8 models × 4 datasets × 10 repetition counts, with accuracy, format compliance, refusal rate, entropy, and repetition metrics
- Data preparation pipeline for 4 dataset tiers (low/medium-temporal/medium-code/high stakes) from AG News, TRAM Arithmetic, MBPP, and MedQA
- SAE feature stability experiment framework with TransformerLens + sae_lens integration
- Sherlock SLURM submission scripts for V100/L40S GPU clusters
- OOM fixes for high repetition counts (50/100) on larger models
- W&B integration for all experiment tracking (project: justinshenk-time/patience-degradation)

### Key Results
- **Llama-3.1-8B-Instruct**: Clear degradation (TRAM accuracy 25.8% → 12.6% by rep 5)
- **Qwen3-8B**: Warm-up effect, not degradation — accuracy improves with repetition
- **Qwen3-30B-A3B (MoE)**: Behaviorally stable across all repetition counts
- **DeepSeek-R1-Distill-Qwen-7B**: Reasoning-distilled, to be tested with thinking enabled vs disabled

### Models (Phase 2 final set)
1. Llama-3.1-8B-Instruct — degradation case
2. Qwen3-8B — same-scale comparison
3. Qwen3-30B-A3B — MoE architecture
4. DeepSeek-R1-Distill-Qwen-7B — reasoning-distilled

### Files Added
- `scripts/experiments/behavioral_metrics.py` — Main Phase 1 experiment script
- `scripts/experiments/patience_degradation.py` — Activation extraction + SAE probing
- `scripts/experiments/sae_feature_stability.py` — SAE stability analysis
- `scripts/experiments/sequential_activation_tracking.py` — Sequential tracking
- `data/processed/patience_degradation/` — Processed datasets (4 tiers)
- `data/filtered_temporal/` — Filtered temporal datasets
- `phase1_completion_report.md` — Full Phase 1 report
- Sherlock submission scripts and requirements files

### Test Plan
- [ ] Phase 1 behavioral metrics reproduce on Sherlock
- [ ] W&B runs match reported results
- [ ] Dataset processing pipeline runs end-to-end
- [ ] Phase 2 activation extraction builds on this branch
