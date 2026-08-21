# E-A2 — the context route and the weight route install interchangeable biases

**Verdict: the register component is route-independent behaviorally, and only partially
shared geometrically.** A mean shift derived purely from in-context demonstrations, injected
into *clean* prompts at one layer, recovers nearly as much of the task as the LoRA-derived
shift does — 0.680 vs 0.730 at L18, 0.640 vs 0.730 at L21 (floor 0.140, adapter ceiling
0.970) — despite the two vectors sharing only cos +0.32 at L18. The cosine climbs with depth
(+0.07 → +0.12 → +0.32 → +0.66 at L7/14/18/21), mirroring Paper B's E6 shared-mode-axis
geometry (+0.75 at its L21 analog): the deeper the layer, the more the two installation routes
converge on one axis — but equal behavioral effect at partial alignment means the functional
region is a *subspace*, not a single line.

Run 2026-08-21 · Qwen2.5-7B-Instruct · same 100-case eval panel and protocol as the
map-transfer runs · ICL contexts: gold-answered filler cases to fill 0.85 of 4k (budget bug in
`icl_messages` fixed first — Qwen's BatchEncoding defeated the token count; regression-tested)
· driver `run_lora_map_transfer.py icl-route` · artifacts `map_transfer/icl_route_evals.json`,
`delta_icl.npy`, `icl_lora_cos_profile.json`.

| arm | accuracy | parse |
|---|---:|---:|
| floor (clean prompts) | 0.140 | 1.00 |
| ICL ceiling (demonstrations in context) | 0.760 | — |
| adapter ceiling | 0.970 | 1.00 |
| ICL-vector steer, clean prompts, norm-matched to the LoRA delta: L7 / L14 / L18 / L21 | 0.130 / 0.410 / **0.680** / **0.640** | 0.96–1.00 |
| (raw-norm variants) | 0.130 / 0.120 / 0.490 / 0.610 | |
| reference: LoRA-vector steer at the same layers | 0.140 / 0.480 / 0.730 / 0.730 | |

## Reading

1. **One bias, two delivery mechanisms.** Demonstrations-in-context and weight updates write
   residual shifts whose *mean* carries the same behavioral content: either vector, injected at
   a mid-depth layer, converts the constant-"B" collapse into per-case discrimination at
   ~0.65–0.73. The register component of this task is a property of the model's geometry, not
   of the route that installs it.
2. **Behavioral equivalence exceeds geometric alignment.** At L18 the two vectors agree at only
   cos 0.32 yet steer nearly identically. Combined with the map-transfer result (a single
   *mapped* direction, cos ≈ 0 to the recipient's own, carries nothing), the picture is: within
   a model there is a task-functional *subspace* reachable from many directions; across models
   no ridge-mapped direction lands in it.
3. **Dose is a real variable** (L14: 0.12 raw → 0.41 norm-matched); norm-matching to a vector
   of known effect is the right convention, and raw ICL shifts are mis-scaled at some layers.
4. **Depth convergence replicates across projects**: the icl/lora cosine profile rising to 0.66
   at L21 is the same shape as E6's mmlu/gsm8k mode-vector cosine (+0.75 at L21-analog) — deep
   layers hold the shared mode axis.

## Provenance note

The `icl_messages` fill budget was inert for Qwen tokenizers before today's fix (BatchEncoding
`len()` counts keys): any *Qwen* run through that helper used effectively unbounded contexts.
The June `lora_icl` artifacts predate the Qwen configs (Gemma era, list-returning tokenizer)
and should be unaffected, but re-verify before quoting them in the extended paper.
