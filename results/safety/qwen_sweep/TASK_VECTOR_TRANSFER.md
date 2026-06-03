# Transferring the ICL task gain to a fresh chat (Qwen, DDXPlus)

Can the performance gain from many-shot ICL be delivered to a FRESH zero-shot chat
without paying the long-context cost — and does representation drift across context fill
break it? Held-out eval n≈40.

## Result — yes, via a reusable task vector (additive steering)

| method | DDXPlus acc | reusable / cheap? |
|---|--:|---|
| zero-shot (fresh chat) | 0.14 | — |
| 4-shot ICL (gain to transfer) | 0.76 | no — pays context cost |
| **steer `d_icl_many`, α=0.5** | **0.70** | **yes — one fixed vector, no demos** |
| steer `d_icl_few`, α=0.5 | 0.65 | yes |
| steer `d_lora`, α=0.5 (oracle) | 0.68 | yes (needs adapter) |
| last-token patch (transplant) | 0.795 | no — needs the case's own 48-shot forward |

Additive steering of a single fit-set task vector takes a fresh prompt **0.14 → 0.70**,
~92% of the ICL gain, for the cost of adding a vector. Clear **Goldilocks** magnitude:
α≈0.5 best, α=1 degrades, α=2 over-steers to collapse (`d_lora` → 0.00) — consistent with
the project's other steering results.

## On representation drift (the motivating worry)
- **For steering: moot.** Adding a direction preserves the fresh chat's own state, so the
  fill-drift offset in the raw shift doesn't transplant. Notably the **many-shot** direction
  (most drift-contaminated) transferred *as well as* few-shot (0.70 vs 0.65) — the mean
  task component dominates the additive steer.
- **For patching: it depends on the gap, and here it did NOT break.** The earlier
  clean→fatigued patch degenerated because it bridged a huge positional gap (200-tok →
  25k). This patch is long→short, last-token-only, 5 layers, position-matched (both prompts
  end at "…Answer:"), so the transplant decodes fine (0.795). But it still requires the
  case's own 48-shot forward, so it does **not** save the context cost — it's an
  oracle/upper-bound, not a practical transfer.

## Takeaway
The deployable recipe is **additive task-vector steering**: extract one DDXPlus task
direction once (from ICL shifts — no finetuning needed), add it at α≈0.5 to any fresh
zero-shot prompt, recover ~92% of the ICL accuracy gain with no demonstrations and no
long-context cost. Drift does not block it. (Earlier we showed this same task direction is
shared with the LoRA route and is off the refusal axis — so this transfer carries the task
benefit without the weight route's safety side-effect.)

## Safety of the steering method (task vector added to every prompt)

| condition | task_acc | refusal |
|---|--:|--:|
| base | 0.139 | 1.000 |
| steer α=0.5 | 0.650 | 0.980 |
| steer α=1.0 | 0.650 | 0.140 ← refusal-axis leak |
| **steer_orth α=0.5** | **0.700** | **1.000** |
| **steer_orth α=1.0** | 0.650 | **0.960** |
| steer_orth α=2.0 | 0.225 | 0.000 |
| LoRA finetune (600 ex) | 0.975 | 0.000 |

The naive task vector is a safety-preserving finetuning alternative **only in a Goldilocks
band**: at α=0.5 task 0.14→0.65 with refusal intact (0.98), but at α=1.0 refusal collapses
to 0.14 (no extra task gain) — the vector has a small refusal-axis leak
(`cos(u,r) ≈ −0.16..−0.26` at mid layers) that bites when over-steered.

**Orthogonalizing the task vector against r removes the leak → robustly safe.** With the
r-component projected out: α=0.5 gives acc **0.70 / refusal 1.00** (strictly better than
naive), and α=1.0 holds refusal at **0.96** where the naive version had collapsed to 0.14.
So across the useful range (α≈0.5–1.0) it confers DDXPlus skill (0.14→0.70) with refusal
fully preserved — vs the LoRA finetune (0.975 acc but refusal → 0.00). (Pathological α=2.0
collapses both task and refusal, but task acc is already gone there.) Script:
`scripts/safety/run_steering_safety.py` → `steering_safety.json`.

## Caveats
One model (Qwen-7B), n≈40, single seed; steering adds to all positions (last-token-only
FV-style steering untested here); the 48-shot ICL accuracy baseline was `nan` (length
guard), so the gain reference is the 4-shot ICL (0.76).

## Reproduce
`scripts/safety/run_task_vector_transfer.py` with `configs/safety/route_safety_qwen.yaml`;
`AdditionSteeringHook` in `src/probes/safety/steering_hook.py`. JSON:
`task_vector_transfer.json`.
