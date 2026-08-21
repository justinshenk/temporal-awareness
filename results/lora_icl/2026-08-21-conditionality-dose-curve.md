# E-A3 — the bias is learned first: the conditionality index against training dose

**Verdict: the constant-carriable (register) component of the adapter is at full strength by
25 training examples and flat thereafter; everything additional data buys is input-conditional
refinement the constant cannot carry — so the index falls monotonically as the ceiling rises.**
At dose 25 the index exceeds 1: the mean vector steers *above its own adapter's ceiling*
(0.700 vs 0.680) — at low dose the adapter effectively *is* its mean shift, and averaging over
the eval panel even denoises it slightly.

Run 2026-08-21 · Qwen2.5-7B-Instruct · nested seeded train slices [:25] ⊂ [:75] ⊂ [:225] ⊂
[:600], each trained to its own adapter (r16 α32, identical recipe) · same 100-case eval panel
and protocol as every arm in this program (floor 0.130) · driver
`run_lora_map_transfer.py dose-curve` · artifacts `map_transfer/dose_curve_evals.json`,
`delta_d{25,75,225}.npy`.

| dose | ceiling | best self-steer | steerable accuracy (steer − floor) | index |
|---:|---:|---:|---:|---:|
| 25 | 0.680 | 0.700 (L18) | 0.570 | **1.04** |
| 75 | 0.840 | 0.770 (L21) | 0.640 | 0.90 |
| 225 | 0.880 | 0.720 (L18) | 0.590 | 0.79 |
| 600 | 0.970 | 0.730 (L18/L21) | 0.600 | 0.71 |

The two curves separate cleanly: the steerable amount is flat at ${\sim}0.57$–$0.64$ from the
smallest dose onward, while the ceiling climbs $0.68 \to 0.97$. Training data past ~25 examples
buys only the conditional part.

**Geometry rider — the functional subspace again.** The dose shifts are far from collinear
with the full-dose shift (cos to δ₆₀₀ at L18: 0.57 / 0.62 / 0.59; at L21: 0.76 / 0.82 / 0.82)
yet all steer to 0.70–0.77. As in E-A2 (ICL vector, cos 0.32, effect 0.68), behavioral
equivalence does not require directional agreement: the register lives in a subspace reachable
from many directions, and cosine to any particular working vector does not predict effect.

**Placement in the extended-paper argument.** With E-A1 and E-A2 this completes the
within-model program: the register component is (i) route-independent (weights or context),
(ii) learned essentially instantly (25 examples), (iii) carried by a subspace rather than a
line, and (iv) the *whole* story at the register pole and *none* of it at the computation pole
(GSM8K index 0.000). What data and rank buy beyond the first few examples is precisely the
input-conditional computation that also refuses to cross models (the map-transfer null).

Caveats: one seed per dose (nested slices share their prefix, which is the design, but dose
curves would tighten with seed replicates); index >1 at d25 is within eval noise of 1.0
(n=100); L18/L21 only.
