# EXECUTION — cross-model LoRA capability transmission via linear maps (Qwen rung)

**Parent:** `tasks/lora_linear_map_transfer_idea.md` (design agreed in chat). **Started
2026-08-21.** User: run autonomously, report on request.

**Question:** does the DDXPlus capability of a LoRA trained on Qwen2.5-7B-Instruct transmit to
untrained Qwen2.5-1.5B-Instruct through per-layer linear maps between their activation spaces?

## Assets / facts

- Adapters lost with the box — **retraining** from the committed configs (seeded, nested
  slices): `configs/lora_icl/ddxplus_qwen_lora.yaml` (7B, n=600),
  `ddxplus_qwen1.5b_lora.yaml` (1.5B ceiling). Plus 7B `--shuffle-labels` control.
- Both models 28 decoder layers (7B d=3584, 1.5B d=1536) → 1:1 layer pairing, maps 3584→1536.
- Reuse: `train_ddxplus_lora.py`, `PerTokenResidualCapture` + `last_token_residual` +
  `stack_shift_set`, `AdditionSteeringHook` (`decode_time`), `build_cases`/`disjoint_split`
  (seed 42 → identical eval panel of 100 cases everywhere; prompts are model-independent text).

## Design

1. **Map corpus:** 400 train-slice DDXPlus clean prompts (disjoint from eval), both models,
   final-token residuals at all 28 layers. Fit per-layer ridge maps A→B; report held-out R²
   (80/20 split) so a transfer null is attributable.
2. **δ_A:** mean of `lora_shift_real` = resid(7B+LoRA, clean) − resid(7B, clean) over the 100
   eval cases, per layer. Same for the shuffled adapter.
3. **Donor sanity (7B):** floor, ceiling (adapter), and self-steer (base + δ_A at its own
   layer, no map) — if self-steer does not move the donor's own accuracy, mean-shift
   transmission is dead at home and the run stops there.
4. **Recipient arms (1.5B, decode_time steering):** floor, ceiling (own adapter),
   transfer_real = M_L·δ_A at L ∈ {14, 18, 21} × α ∈ {1, 2}, norm-matched random control per
   (L, α) [dose-matched-controls lesson], transfer_shuffled at the best (L, α).
5. **Readout:** recovered fraction = (transfer − floor) / (ceiling − floor), paired bootstrap
   over the shared 100-case panel.

## Falsification / void

- Self-steer null on the donor → premise dead (report, stop).
- Random control ≈ real at every dose → instrument, not transfer.
- Held-out map R² ≈ 0 → null uninformative (report R² regardless).
- Ceiling ≈ floor on 1.5B → no headroom, redesign.

## Files

- `src/probes/lora_icl/linear_map_transfer.py` (test-first: ridge fit/R², transfer vector,
  norm-matched control) + tests.
- `scripts/lora_icl/run_lora_map_transfer.py` — subcommands `capture-donor`, `fit-maps`,
  `run-recipient`.
- Artifacts → `results/lora_icl/map_transfer/`; report `results/lora_icl/MAP_TRANSFER.md`.

## OUTCOME (2026-08-21) — NULL, maximally attributed. Report: results/lora_icl/MAP_TRANSFER.md

Self-steer works in BOTH models (donor 0.73, recipient 0.67 vs floors 0.13/0.14, ceilings
0.97/0.90); maps R2 0.72–0.82; transfer zero everywhere; mapped vectors direction-specifically
destroy generation at 2x dose while norm-matched random is harmless; closing geometry:
cos(mapped donor shift, recipient's own shift) = −0.19..+0.18 → the map connects states, not
function. Fifth driver phase `recipient-selfsteer` added mid-run to attribute the null. Next
rungs if resumed: subspace (top-k) transfer, cross-layer pairing search, Procrustes maps,
Qwen→Gemma at matched width.
