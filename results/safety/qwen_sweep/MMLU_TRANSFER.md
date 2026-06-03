# Domain shift: does task-vector steering generalize to MMLU, and stay safe? (Qwen)

Stress-test of the steering-as-finetuning-alternative on MMLU (general knowledge), incl.
cross-domain transfer of the DDXPlus medical vector. n_eval=50 MMLU, n_harmful=25.

## 1. No headroom on MMLU → nothing to transfer
| | acc |
|---|--:|
| MMLU zero-shot | 0.681 |
| MMLU 5-shot ICL | 0.660 |

Qwen already knows MMLU zero-shot; few-shot ICL does **not** help (even slightly lower). So
there is no ICL gain to transfer, and steering with the MMLU-native vector only adds noise:

| steer (MMLU vector) | acc | refusal |
|---|--:|--:|
| all-position, α=0.5 | 0.653 | 1.00 |
| all-position, α=1.0 | 0.600 | 0.92 |
| last-token, α=1.0 | 0.620 | 1.00 |

**The steering-as-finetuning-alternative is specific to tasks the model lacks zero-shot**
(DDXPlus's niche format, 0.14→0.65). On knowledge it already has (MMLU), there's no gain;
steering mildly hurts.

## 2. Task vectors are domain-specific (cross-domain transfer fails)
The DDXPlus **medical** vector applied to MMLU is essentially inert at low α (0.680 = base
0.681) — the medical task direction is irrelevant to MMLU questions. So task vectors carry
domain/task content, not a generic "MCQ-answering" boost.

## 3. The safety hazard under domain shift (the important result)
| steer (DDXPlus medical vector) on MMLU | acc | refusal |
|---|--:|--:|
| all-position, α=0.5 | 0.680 | 1.00 |
| **all-position, α=1.0** | 0.640 | **0.000** |
| last-token, α=1.0 | 0.640 | 1.00 |

**The medical vector, deployed broadly (all-position, α=1), fully destroys refusal
(1.00→0.00) on harmful prompts — while doing nothing useful for MMLU.** A task vector carries
its refusal-axis leak into *any* context it is added to, independent of the target task; the
worst case is exactly this domain-shift setting (out-of-domain vector, no task benefit, full
safety cost). The MMLU-native vector leaks less (refusal 0.92 at α=1) — different vectors have
different leaks — but both are dangerous at full all-position strength.

**Last-token application removes the hazard for both vectors (refusal 1.00 at α=1).** This is
the strongest case yet that **last-token steering is the safety-robust application** — the
refusal-axis leak only lands on one position of harmful prompts instead of all.

## Takeaways
- Steering transfers a *task* only where the model lacks it zero-shot; MMLU has no headroom.
- Task vectors are domain-specific (medical vector inert on MMLU).
- **All-position steering of an out-of-domain vector at α=1 is a real safety hazard (refusal
  → 0); last-token steering eliminates it.** Confirms last-token as the safe default, and
  shows the danger is maximal under domain shift.

## Caveats
One model, n_eval=50 / n_harmful=25, single seed; MMLU sampled from `cais/mmlu` "all". The
absolute MMLU accuracies are typical for Qwen-7B; the safety numbers are the robust signal.

## Reproduce
`scripts/safety/run_mmlu_transfer.py` with `configs/safety/route_safety_qwen.yaml`. JSON:
`mmlu_transfer.json`.
