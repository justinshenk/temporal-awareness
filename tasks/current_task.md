# Current Task: Attention Visualization Overhaul

## Status: In progress — iterating on quality

## What was done

### Architecture rewrite (attn analysis + visualization)
- **Data layer**: `AttnPairResult` now stores `dst_group_attention: dict[str, DstGroupAttention]` — label-aligned attention matrices for EVERY format_pos group as destination, with clean and corrupted sides properly aligned by canonical label (union of both frames).
- **Analysis** (`attn_analysis_run.py`): `run_attn_analysis` accepts both clean and corrupted mappings, uses ALL model layers (not just configured subset), builds per-dst-group attention for every format_pos present in either frame.
- **Per-pair viz** (`attn_pair_viz.py`): Clean module generating per-dst subfolders with 11 plot types each.
- **Aggregated viz** (`attn_viz.py`): Stream-loads per-pair JSONs from disk, averages by canonical label, generates same plot types as per-pair.
- **Memory**: `pop_heavy()` called after per-pair viz to drop heavy patterns.

### Plot types per dst format_pos group
- `attn_heatmaps/<dst>.png` — layer × position, mean over heads, per-layer normalized
- `attn_sidebyside/<dst>.png` — clean | corrupted layer-level
- `attn_diff/<dst>.png` — clean - corrupted layer-level
- `attn_consistency/<dst>.png` — per-(head, label) cosine similarity
- `attn_heads/<dst>.png` — heatmap: y=(layer.head), x=source format_pos groups
- `attn_flow/<dst>.png` — all-layer flow figure
- `source_bars/<dst>.png` — rows = src format_pos groups, top heads paired bars
- `top_attended/<dst>.png` — top-8 heads, y = group ticks
- `head_heatmaps/<dst>/L<layer>.png` — per-(dst, layer) clean head heatmap
- `head_diff/<dst>/L<layer>.png` — per-(dst, layer) clean - corrupted
- `heads_sidebyside/<dst>/L<layer>.png` — per-(dst, layer) clean | corrupted

### Coarse plot fixes (investment)
- `layer_position_heatmap.png`: densest position sweep, format_pos:rel_pos labels
- `noise_vs_denoise_per_component_layer.png`: single row of resid_post/attn_out/mlp_out, AND-like/NECESSARY + OR-like/SUFFICIENT labels
- `layer_component_interaction.png`: integer x-ticks, only attn_out/mlp_out/resid_post
- `position_component_interaction.png`: grouped by format_pos (mean across rel_pos), only attn_out/mlp_out/resid_post, attn_out drawn on top with dashed+x markers
- `marginal_contribution.png`: no L23 annotations, only resid_post on secondary axis (alpha 0.35), no y=0 line
- `marginal_contribution_var.png`: new — mean ± std across pairs
- `cumulative_recovery.png`: resid_post as filled area (absolute recovery, not cumsum), no Full Recovery line
- `attn_vs_mlp_paired.png`: only significant-movement layers plotted, simplified title
- `component_importance_ranked.png`: top 20, sorted by recovery + disruption combined

## Iterations completed (nano attn)
1. Initial rewrite — 38 dst groups, 4112 PNGs, clean≠corrupted verified
2. Sparse group x-ticks — 10/11 PASS
3. attn_heads → compact heatmap, source_bars height cap — 11/11 PASS
4. Union layer keys + zero-fill for one-frame-only dst groups
5. Aggregated attn plots added (stream from per-pair JSON, same plot types)

## Remaining
- Investment per-pair attn: only 11/71 pairs have attn/ (rest were from killed run). Aggregation is done from the old cached JSONs.
- Need to finish the remaining 60 per-pair investment attn runs when time allows.
