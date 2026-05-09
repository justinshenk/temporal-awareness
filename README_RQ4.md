# RQ4: Diagnosing Probing Claims About Future Tokens

All result files are in `results/lookahead/final/`.

## Paper Section → Data File Mapping

### Section 3.1: Position Baselines Match Code Probes
- 11-model staircase (Table 1): `*_final.json` (11 files)
- Spearman correlations (Fig 1): computed from `*_final.json`
- Opaque/adversarial names (Appendix G): `robust_s6_*.json`, `robust_s9_*.json`

### Section 3.2: Temporal Horizon Narrows During Training
- K decay, 17 checkpoints (Fig 2A): `mech_s3_kdecay_all.json`
- Fixed-text control (Fig 2B): `robust_s4_fixed_eval.json`
- Stronger baseline (Appendix A): `robust_s5_stronger_baselines.json`

### Section 3.3: Logit Lens (Fig 3)
- `mech_s1_logit_pythia_2_8b_deduped.json`
- `mech_s1_logit_pythia_410m_deduped.json`

### Section 3.4: Training Dynamics (Fig 4)
- `dynamics_p1_pythia_{2_8b,1b,410m}_deduped.json`
- `dynamics_p2_pythia_{2_8b,1b,410m}_deduped.json`

### Appendix
| Section | File |
|---------|------|
| A: Stronger baseline | `robust_s5_stronger_baselines.json` |
| B: Probe vs behavioral | `dynamics_p2_*.json` |
| C: Domain spectrum | `overnight_complete.json` |
| D: MLP vs attention | `overnight_phase1b_attnmlp.json` |
| E: Transfer matrix | `overnight_phase1c_transfer.json` |
| F: Fixed-text results | `robust_s4_fixed_eval.json` |
| G: Opaque names | `robust_s6_*.json`, `robust_s9_*.json` |
| H: Grouped CV | `robust_s8_grouped_cv.json` |
