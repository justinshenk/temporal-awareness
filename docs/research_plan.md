# Research Plan

## Goal

Detect and steer temporal reasoning (immediate vs long-term thinking) in LLM internal representations.

## Phases

### Phase 1: Dataset & Baseline (Current)
- [x] Create explicit temporal pairs (50)
- [x] Create implicit temporal pairs (50)
- [x] Train linear probes
- [x] Achieve 99% accuracy on explicit
- [ ] Expand to 200+ pairs

### Phase 2: Robustness
- [ ] Cross-model validation (GPT-2 → Pythia)
- [ ] Adversarial evaluation (paraphrased)
- [ ] Confound analysis

### Phase 3: Circuits
- [ ] Activation patching
- [ ] Identify key attention heads
- [ ] Ablation studies

### Phase 4: Applications
- [ ] Divergence detection (stated vs internal)
- [ ] Steering demonstrations
- [ ] Integration with latents library

## Key Questions

1. Is temporal reasoning linearly encoded?
2. Does it transfer across models?
3. Can we detect when stated horizon differs from internal?

## Related Work

See [RELATED_WORK.md](RELATED_WORK.md)
