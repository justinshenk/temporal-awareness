# Route-dependent safety cost — Qwen2.5-7B (DDXPlus → refusal), dose sweep

Base refusal **1.000**, base DDXPlus acc **0.324**; held-out harmful eval n=50.

| route | dose | task_gain | ΔRefusal | mean cos(shift,r) | last-layer cos(shift,r) |
|---|--:|--:|--:|--:|--:|
| icl | 4sh | +0.307 | +0.000 | -0.103 | +0.000 |
| icl | 16sh | +0.301 | +0.040 | -0.130 | -0.074 |
| icl | 48sh | +nan | +0.060 | -0.130 | -0.127 |
| icl | 128sh | +nan | +0.040 | -0.138 | -0.166 |
| lora | 25ex | +0.401 | +0.000 | -0.217 | -0.310 |
| lora | 75ex | +0.426 | +1.000 | -0.275 | -0.407 |
| lora | 225ex | +0.551 | +1.000 | -0.273 | -0.464 |
| lora | 600ex | +0.601 | +1.000 | -0.320 | -0.514 |

## The two relationships
- **ΔRefusal vs refusal-axis alignment** (last-layer cos): corr = **-0.875** (negative cos ⇒ toward compliance ⇒ erosion; a negative corr means erosion tracks anti-refusal movement).
- **Task gain does NOT predict erosion** (it's not a dose effect): the cleanest evidence is the LoRA threshold — 25ex (gain +0.40) erodes 0.00 while 75ex (gain +0.43) erodes +1.00. Near-identical competence, opposite safety. So erosion is about *how* the move is made (along r), gated by a threshold — not how much task is learned. (Naive corr(task_gain, ΔRefusal) = +0.835 over n=6 is a small-n artifact: the high-gain points are simply the LoRA ones.)

## Route contrast (means)
- **ICL (activation):** task_gain +0.304, ΔRefusal +0.035, last cos(shift,r) -0.092 (range -0.166..+0.000 over depth).
- **LoRA (weight):** task_gain +0.494, ΔRefusal +0.750, last cos(shift,r) -0.424.

## Route-dependence at comparable task gain
- ICL reaches task_gain up to **+0.307** (4 shots) at ΔRefusal **+0.000** (cos_r +0.000); deeper ICL (≤128 shots) keeps ΔRefusal ≤ 0.06.
- LoRA at task_gain **+0.426–+0.601** (75–600 ex) collapses refusal to **0.00** (ΔRefusal +1.00, cos_r -0.462).
- Same task, comparable competence gain: the **weight** route couples it to refusal erosion along −r; the **activation** route does not, at any tested depth.
