# Register vs Procedure & Context Fatigue

Two active research projects sharing one codebase (Llama-2-7b attribution stack +
multi-model context-accumulation harness). Both target *Interpretability as a Science*
(NeurIPS 2026 workshop).

## Paper A — Register, Not Procedure

Whether a LoRA-installed behavior transports through a fitted pointwise activation map
depends on what the behavior *is*: a register (an output format/disposition) installs
through a single-layer linear map; a multi-step procedure (GSM8K arithmetic, MuSiQue
multi-hop composition) does not — its full-δ oracle recovers, but every pointwise rung
of the ladder stays near zero.

- Drivers: `scripts/attribution/` (+ `scripts/safety/` for the refusal arm)
- Library: `src/probes/attribution/`, `src/probes/safety/`
- Configs: `configs/attribution/`
- Artifacts: `results/attribution/` (gitignored provenance store; every paper number
  traces to a JSON there via `papers/register_vs_procedure/numbers.md`)
- Paper sources: `papers/register_vs_procedure/`

## Paper B — Context Fatigue

The "context fatigue" signatures (entropy collapse, attention drift) replicate across
Qwen-2.5-7B, Llama-3.1-8B, Gemma-2-9B, and the OLMo-2-7B post-training chain — but with
attention dilution structurally removed (individually localized tasks), there is no
performance cost. The real hazard is a widening confidently-wrong gap.

- Drivers: `scripts/context_fatigue/`
- Library: `src/probes/context_fatigue/`
- Artifacts: `results/context_fatigue/`
- Paper sources: `context_fatigue_paper/`

## Setup

```bash
uv sync
```

## Tests

```bash
make test   # CPU-only, no network, seeded
```

## Working docs

- `tasks/current_task.md` — live experiment ledger
- `tasks/lessons.md` — experimental-design and analysis lessons
- `docs/superpowers/specs/2026-08-07-workshop-papers-design.md` — the two-paper design spec
