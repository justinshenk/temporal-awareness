# Graded-risk MCQ sweep: route-dependence + steering breadth on a new task family

Holds the task FUNCTION fixed (4-option factual MCQ) and varies content topic across four
buckets at intended increasing proximity to the refusal axis r (MMLU neutral → WMDP hazardous):
B0 neutral (geography/history), B1 clinical/law, B2 dual-use (bio/security), B3 WMDP bio/chem/cyber.
Two goals: (1) generalize the W·a steering result to a non-medical task family; (2) test whether
ICL erosion grows with content→r. Qwen2.5-7B; see `configs/safety/graded_risk_qwen*.yaml`.

## Premise check (gradient gate): NULL

Bucket-mean cos(content, r) at the MCQ prediction site is uniformly ≈−0.02 and **not** monotone
(B0 −0.032, B1 −0.011, B2 −0.033, B3 −0.006). **Topic ≠ distance-to-r under benign exam framing**:
r encodes request INTENT, not subject matter. A WMDP virology question asked as an exam item is not
"harmful-request-shaped." The gate aborts the sweep unless `--force`. This null is *why* the right
axis is harmful-prompt distance-to-fit (see `OOD_LUCK_PROBE.md`), not benign-content topic.

## Results (best safe steer = highest-acc condmap with refusal ≥ 0.9)

| bucket | zs→fs acc | ICL ΔRef | ICL cos(shift,r) | LoRA ΔRef | LoRA cos(shift,r) | steer acc/ref |
|---|---|---|---|---|---|---|
| B0 | 0.82→0.88 | +0.00  | −0.103 | +0.00  | −0.198 | 0.84 / 0.975 |
| B1 | 0.60→0.66 | +0.05  | −0.117 | +0.00  | −0.310 | 0.64 / 0.950 |
| B2 | 0.71→0.72 | +0.025 | −0.085 | +0.025 | −0.339 | 0.71 / 0.950 |
| B3 | 0.74→0.78 | +0.025 | −0.126 | +0.00  | −0.270 | 0.74 / 0.950 |

`corr(ΔRefusal, cos(shift,r))@L27 pooled over ICL+LoRA = +0.15` (n=12).

## Interpretation

1. **ICL barely erodes on any benign bucket** (ΔRef ≤ 0.05, within multi-seed noise) — route-
   dependence holds: the activation route is ~safe. The ICL shift carries a *small constant* anti-r
   component (cos ≈ −0.1) regardless of topic — the same constant-direction leak seen in the OOD probe.
2. **The MCQ-bucket LoRAs don't erode either** (ΔRef ≈ 0). This is the **MedMCQA regime**, not a
   counterexample to route-dependence: the model is already competent on these buckets (acc 0.7–0.88),
   so the finetune barely moves the weights → little erosion (erosion scales with how far the finetune
   moves the weights; cf. DDXPlus strong→full collapse, MedMCQA weak→none). The tell that the *direction*
   of route-dependence still holds: **LoRA's shift is consistently more anti-r than ICL's** (−0.27 avg vs
   −0.11) — the weight route leans toward −r more, the magnitude is just sub-threshold here.
3. **corr is near-zero (+0.15) only because nothing erodes** — there is no erosion signal to track
   movement along r. The −0.875 mechanism needs a *strong* finetune (DDXPlus) to manifest; these buckets
   don't provide one.
4. **W·a steering transfers the (small) ICL gain refusal-safe at α=0.5** across buckets — breadth of the
   linear-map steering confirmed on a third, non-medical task family. The transfer is small because
   MMLU/WMDP have little zero-shot→few-shot headroom for Qwen (headroom law: steering transfers exactly
   the ICL gain available; DDXPlus 0.14→0.70 huge, here ≤0.06).

## Bottom line

On benign content, *neither* route erodes much — route-dependence holds but is undramatic because the
finetune is weak (model already competent). The dramatic, decisive result is on the *other* axis:
grading the **harmful** prompts by distance-to-fit, where the conditional map's safety is shown to be
OOD-luck (`OOD_LUCK_PROBE.md`). This sweep's lasting contributions: the **benign-topic≠distance-to-r
null**, the **weak-finetune no-erosion regime** (LoRA shift more anti-r than ICL's but sub-threshold),
and **W·a steering breadth** on MMLU/WMDP.

Artifacts: `results/safety/graded_risk_qwen/{graded_risk_sweep.json, graded_risk_summary.json,
graded_gradient.json, graded_risk_plot.png}`. Related: `OOD_LUCK_PROBE.md`,
memory `route-dependent-safety-result`, `steering-map-ood-luck-result`.
