# Does the LoRA finetune change attention vs base? (Qwen, DDXPlus→refusal)

Base vs LoRA-600 last-query-token attention on the same prompts; per head
`1 − cos(base_attn, lora_attn)` (same prompt ⇒ same key positions). n=30 each.

## Result — significant, but sparse and late-layer-concentrated

| | harmful | task |
|---|--:|--:|
| overall mean 1−cos | 0.100 (cos≈0.90) | 0.185 (cos≈0.82) |
| per-layer: early (L0) | 0.003 | 0.003 |
| per-layer: late (L21/L24/L27) | 0.19 / 0.17 / 0.22 | 0.29 / 0.33 / 0.30 |
| top changed heads | L19-H2 0.89, L19-H17 0.81, L21-H1 0.74 | L24-H21 0.97, L24-H27 0.97, L24-H26 0.96 |

- **Localized, not diffuse:** most heads barely move (overall mean 0.10–0.18); a handful of
  late-layer heads are almost fully rerouted (0.7–0.97).
- **Late-layer concentration:** L0–L3 ≈ identical; divergence grows through L9–L27 — the
  same region where the refusal-erosion direction lives (`cos(ŵ, r)` peaks L21/L27). The
  attention rerouting **co-localizes with the erosion**, a plausible upstream cause of the
  residual shift along ŵ.
- **Task > harmful** (0.185 vs 0.100): the LoRA reshapes attention most on its training
  task, but also reroutes specific late-layer heads on harmful prompts (L19-H2 0.89,
  L21-H1 0.74) — candidate heads for how it suppresses refusal.

## Causal test — the heads are a CORRELATE, not the route
Patching the base per-head attention outputs back into the LoRA at the top harmful heads
(n=40 harmful, `head_patch_causal.json`):

| condition | refusal |
|---|--:|
| base | 1.000 |
| LoRA (none) | 0.000 |
| + patch top-10 harmful heads | 0.000 |
| + patch random heads | 0.000 |
| + patch ALL heads at those layers | 0.000 |

Restoring the base attention — even all heads at the divergent layers — does **not** recover
refusal. So the attention rerouting does not *carry* the erosion. Together with the
weight-projection null and the partial success of activation-direction ablation, the picture
is: the erosion is a **distributed, emergent residual-stream shift along ŵ** — readable and
partially removable as a direction, but not localized to specific heads or a low-rank
weight write.

Caveat on the negative: patched only the last-token attention output during prefill, and
`o_proj` is itself LoRA-modified (so the head contribution wasn't fully reverted to base).
A stronger test patches the head's residual contribution (o_proj output) at all positions.

## Caveats (descriptive part)
Last-token attention only, n=30, one adapter (600ex), single seed; the harmful-prompt
change may be partly generic LoRA modification rather than refusal-specific.

## Reproduce
`scripts/safety/run_attention_base_vs_lora.py` with `configs/safety/route_safety_qwen.yaml`;
reuses `SelectiveAttentionCapture`. JSON: `attention_base_vs_lora.json`.
