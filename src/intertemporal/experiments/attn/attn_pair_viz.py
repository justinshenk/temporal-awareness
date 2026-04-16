"""Per-pair attention visualizations.

Consumes ``AttnPairResult.dst_group_attention`` (label-aligned) and emits a
collection of plots, organized into per-dst-format_pos subfolders.

Output layout (under ``output_dir``):

    attn_heatmaps/<dst>.png        layer-level mean-over-heads, per-layer norm
    attn_sidebyside/<dst>.png      clean | corrupted
    attn_diff/<dst>.png            clean - corrupted
    attn_consistency/<dst>.png     per-(head, src_label) cosine
    attn_heads/<dst>.png           heatmap y=head x=src_format_pos
    attn_flow/<dst>.png            all-layer flow figure
    source_bars/<dst>.png          rows=src groups, top heads
    top_attended/<dst>.png         top-K heads, y=group ticks
    head_heatmaps/<dst>/L<l>.png   per-(dst, layer) clean head heatmap
    head_diff/<dst>/L<l>.png       per-(dst, layer) clean - corrupted
    heads_sidebyside/<dst>/L<l>.png per-(dst, layer) clean | corrupted
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from ....viz.plot_helpers import add_pair_label, save_figure
from .attn_analysis_results import AttnPairResult, DstGroupAttention


# ─── Style ──────────────────────────────────────────────────────────────────
DPI = 150
LAYER_COLORS = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#795548']
BAR_ALPHA = 0.85
GRID_ALPHA = 0.3
CLEAN_COLOR = '#2196F3'
CORRUPTED_COLOR = '#FF9800'
DIFF_CMAP = 'RdBu_r'


# ─── Helpers ────────────────────────────────────────────────────────────────


def _layer_keys_int(d: dict) -> list[int]:
    return sorted(int(k) for k in d.keys())


def _union_layer_keys(dga: DstGroupAttention) -> list[int]:
    keys: set[int] = set()
    keys.update(int(k) for k in dga.clean.keys())
    keys.update(int(k) for k in dga.corrupted.keys())
    return sorted(keys)


def _layer_get(d: dict, layer: int) -> "list[list[float]] | None":
    if layer in d:
        return d[layer]
    return d.get(str(layer))


def _zeros_like_other(dga: DstGroupAttention, layer: int) -> "list[list[float]] | None":
    n_labels = len(dga.canonical_labels)
    other = _layer_get(dga.clean, layer) or _layer_get(dga.corrupted, layer)
    if other is None:
        return None
    return [[0.0] * n_labels for _ in range(len(other))]


def _row_max_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    row_max = arr.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    return arr / row_max


def _layer_matrix(dga: DstGroupAttention, side: str, layers: list[int]) -> np.ndarray:
    n_labels = len(dga.canonical_labels)
    out = np.zeros((len(layers), n_labels), dtype=float)
    src = dga.clean if side == "clean" else dga.corrupted
    for i, layer in enumerate(layers):
        m = _layer_get(src, layer)
        if not m:
            continue
        arr = np.array(m, dtype=float)
        out[i, :] = arr.mean(axis=0)
    return out


def _group_canonical_labels_by_format_pos(
    canonical_labels: list[str],
) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for ci, full in enumerate(canonical_labels):
        grp = full.split(":", 1)[0]
        if grp not in groups:
            groups[grp] = []
            order.append(grp)
        groups[grp].append(ci)
    return [(g, groups[g]) for g in order]


def _group_tick_positions(
    canonical_labels: list[str],
) -> tuple[list[int], list[str], dict[str, list[int]]]:
    group_columns: dict[str, list[int]] = {}
    order: list[str] = []
    for ci, full in enumerate(canonical_labels):
        grp = full.split(":", 1)[0]
        if grp not in group_columns:
            group_columns[grp] = []
            order.append(grp)
        group_columns[grp].append(ci)
    tick_idx = [int(np.median(group_columns[g])) for g in order]
    return tick_idx, list(order), group_columns


def _set_group_xticks(ax: plt.Axes, labels: list[str], dst: str, fontsize: int = 7) -> None:
    tick_idx, tick_lbl, _ = _group_tick_positions(labels)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbl, rotation=60, ha='right', fontsize=fontsize)
    for label, name in zip(ax.get_xticklabels(), tick_lbl):
        if name == dst:
            label.set_fontweight("bold")


def _set_group_yticks(ax: plt.Axes, labels: list[str], dst: str, fontsize: int = 7) -> None:
    tick_idx, tick_lbl, _ = _group_tick_positions(labels)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_lbl, fontsize=fontsize)
    for label, name in zip(ax.get_yticklabels(), tick_lbl):
        if name == dst:
            label.set_fontweight("bold")


# ─── Main entry ─────────────────────────────────────────────────────────────


def visualize_attn_pair(
    result: AttnPairResult,
    output_dir: Path,
    pair_idx: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result.dst_group_attention:
        return
    n = 0
    for dst in sorted(result.dst_group_attention.keys()):
        dga = result.dst_group_attention[dst]
        n += _plot_layer_heatmap(dga, output_dir, result.pair_idx)
        n += _plot_layer_sidebyside(dga, output_dir, result.pair_idx)
        n += _plot_layer_diff(dga, output_dir, result.pair_idx)
        n += _plot_consistency(dga, output_dir, result.pair_idx)
        n += _plot_attn_heads(dga, output_dir, result.pair_idx)
        n += _plot_attn_flow(dga, output_dir, result.pair_idx)
        n += _plot_source_bars(dga, output_dir, result.pair_idx)
        n += _plot_top_attended(dga, output_dir, result.pair_idx)
        n += _plot_head_heatmaps_per_layer(dga, output_dir, result.pair_idx)
        n += _plot_head_diff_per_layer(dga, output_dir, result.pair_idx)
        n += _plot_heads_sidebyside_per_layer(dga, output_dir, result.pair_idx)


# ─── Layer-level plots ──────────────────────────────────────────────────────


def _plot_layer_heatmap(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    mat = _row_max_normalize(_layer_matrix(dga, "clean", layers))
    nl = len(dga.canonical_labels)
    sub = out / "attn_heatmaps"; sub.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(14, nl * 0.14), max(4, len(layers) * 0.30)))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.invert_yaxis()
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_ylabel("Layer"); _set_group_xticks(ax, dga.canonical_labels, dga.dst_label)
    ax.set_xlabel("format_pos")
    plt.colorbar(im, ax=ax, label="Attention (per-layer norm)", shrink=0.8)
    ax.set_title(f"Layer Heatmap (clean) — pair {pidx} — dst={dga.dst_label}", fontsize=10)
    plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


def _plot_layer_sidebyside(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    cmat = _row_max_normalize(_layer_matrix(dga, "clean", layers))
    kmat = _row_max_normalize(_layer_matrix(dga, "corrupted", layers))
    nl = len(dga.canonical_labels)
    sub = out / "attn_sidebyside"; sub.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(max(16, nl * 0.18), max(4, len(layers) * 0.30)), squeeze=False)
    for col, (m, nm) in enumerate([(cmat, "Clean"), (kmat, "Corrupted")]):
        ax = axes[0, col]; im = ax.imshow(m, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.invert_yaxis(); ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8); ax.set_ylabel("Layer")
        _set_group_xticks(ax, dga.canonical_labels, dga.dst_label); ax.set_xlabel("format_pos")
        ax.set_title(nm, fontsize=10); plt.colorbar(im, ax=ax, label="Attention (per-layer norm)", shrink=0.8)
    fig.suptitle(f"Layer Side-by-Side — pair {pidx} — dst={dga.dst_label}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); add_pair_label(fig, pidx)
    save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


def _plot_layer_diff(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    cmat = _row_max_normalize(_layer_matrix(dga, "clean", layers))
    kmat = _row_max_normalize(_layer_matrix(dga, "corrupted", layers))
    diff = cmat - kmat; vabs = max(abs(diff.min()), abs(diff.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    nl = len(dga.canonical_labels)
    sub = out / "attn_diff"; sub.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(14, nl * 0.14), max(4, len(layers) * 0.30)))
    im = ax.imshow(diff, aspect='auto', cmap=DIFF_CMAP, norm=norm); ax.invert_yaxis()
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_ylabel("Layer"); _set_group_xticks(ax, dga.canonical_labels, dga.dst_label)
    ax.set_xlabel("format_pos")
    plt.colorbar(im, ax=ax, label="Clean - Corrupted (per-layer norm)", shrink=0.8)
    ax.set_title(f"Layer Diff — pair {pidx} — dst={dga.dst_label}", fontsize=10)
    plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


# ─── Consistency ────────────────────────────────────────────────────────────


def _plot_consistency(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels)
    head_labels: list[str] = []; rows: list[list[float]] = []; WINDOW = 3
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        km = _layer_get(dga.corrupted, layer) or _zeros_like_other(dga, layer)
        if not cm or not km:
            continue
        ca, ka = np.array(cm, dtype=float), np.array(km, dtype=float)
        nh = min(ca.shape[0], ka.shape[0])
        for h in range(nh):
            row = []
            for ci in range(nl):
                lo, hi = max(0, ci - WINDOW), min(nl, ci + WINDOW + 1)
                a, b = ca[h, lo:hi], ka[h, lo:hi]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                row.append(float(np.clip(np.dot(a, b) / (na * nb), -1, 1)) if na > 1e-10 and nb > 1e-10 else 0.0)
            rows.append(row); head_labels.append(f"L{layer}.H{h}")
    if not rows:
        return 0
    mat = np.array(rows)
    sub = out / "attn_consistency"; sub.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, nl * 0.14), max(8, len(head_labels) * 0.07)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_yticks(range(len(head_labels))); ax.set_yticklabels(head_labels, fontsize=4)
    _set_group_xticks(ax, dga.canonical_labels, dga.dst_label); ax.set_xlabel("format_pos")
    ax.set_ylabel("Head"); plt.colorbar(im, ax=ax, label=f"Cosine sim (window=±{WINDOW})", shrink=0.8)
    ax.set_title(f"Consistency — pair {pidx} — dst={dga.dst_label}", fontsize=10)
    plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


# ─── Per-source-group breakdowns ────────────────────────────────────────────


def _plot_attn_heads(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    src_groups = _group_canonical_labels_by_format_pos(dga.canonical_labels)
    if not src_groups:
        return 0
    head_labels: list[str] = []; rows: list[np.ndarray] = []
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        if not cm:
            continue
        arr = np.array(cm, dtype=float)
        for h in range(arr.shape[0]):
            row = np.array([float(arr[h, ci].mean()) if ci else 0.0 for _, ci in src_groups])
            rows.append(row); head_labels.append(f"L{layer}.H{h}")
    if not rows:
        return 0
    mat = np.stack(rows, axis=0)
    ng = len(src_groups); gn = [g for g, _ in src_groups]
    sub = out / "attn_heads"; sub.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, ng * 0.4), max(8, len(head_labels) * 0.10)))
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(head_labels))); ax.set_yticklabels(head_labels, fontsize=4)
    ax.set_ylabel("Head (Layer.Head)")
    ax.set_xticks(range(ng)); ax.set_xticklabels(gn, rotation=60, ha='right', fontsize=8)
    for lbl, nm in zip(ax.get_xticklabels(), gn):
        if nm == dga.dst_label:
            lbl.set_fontweight("bold")
    ax.set_xlabel("Source format_pos"); plt.colorbar(im, ax=ax, label="Mean attention", shrink=0.8)
    ax.set_title(f"Head Attention — pair {pidx} — dst={dga.dst_label}", fontsize=10)
    plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


def _plot_source_bars(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    src_groups = _group_canonical_labels_by_format_pos(dga.canonical_labels)
    if not src_groups:
        return 0
    nr = len(src_groups)
    sub = out / "source_bars"; sub.mkdir(parents=True, exist_ok=True)
    rh = max(1.4, min(2.2, 60.0 / max(nr, 1)))
    fig, axes = plt.subplots(nr, 1, figsize=(14, rh * nr), squeeze=False)
    for ri, (gl, ci) in enumerate(src_groups):
        ax = axes[ri, 0]; entries: list[tuple[int, int, float, float]] = []
        for layer in layers:
            cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
            km = _layer_get(dga.corrupted, layer) or _zeros_like_other(dga, layer)
            if not cm or not km:
                continue
            ca, ka = np.array(cm, dtype=float), np.array(km, dtype=float)
            for h in range(min(ca.shape[0], ka.shape[0])):
                cv = float(ca[h, ci].mean()) if ci else 0.0
                kv = float(ka[h, ci].mean()) if ci else 0.0
                entries.append((layer, h, cv, kv))
        entries.sort(key=lambda r: max(r[2], r[3]), reverse=True)
        entries = entries[:20]
        if not entries:
            ax.axis('off'); continue
        lbs = [f"L{l}.H{h}" for l, h, _, _ in entries]
        x = np.arange(len(lbs)); w = 0.4
        ax.bar(x - w/2, [e[2] for e in entries], w, color=CLEAN_COLOR, alpha=BAR_ALPHA, label="Clean")
        ax.bar(x + w/2, [e[3] for e in entries], w, color=CORRUPTED_COLOR, alpha=BAR_ALPHA, label="Corrupted")
        ax.set_xticks(x); ax.set_xticklabels(lbs, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f"{gl}\n(n={len(ci)})", fontsize=8); ax.grid(axis='y', alpha=GRID_ALPHA)
        if ri == 0:
            ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f"Source Attention from {dga.dst_label} — pair {pidx}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); add_pair_label(fig, pidx)
    save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


# ─── Top attended ───────────────────────────────────────────────────────────


def _plot_top_attended(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels)
    hs: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        km = _layer_get(dga.corrupted, layer) or _zeros_like_other(dga, layer)
        if not cm or not km:
            continue
        ca, ka = np.array(cm, dtype=float), np.array(km, dtype=float)
        for h in range(min(ca.shape[0], ka.shape[0])):
            sc = float(max(ca[h].max(), ka[h].max()))
            hs.append((layer, h, sc, ca[h], ka[h]))
    if not hs:
        return 0
    hs.sort(key=lambda x: x[2], reverse=True); top = hs[:8]
    sub = out / "top_attended"; sub.mkdir(parents=True, exist_ok=True)
    nc = 2; nr = (len(top) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(11 * nc, max(7, nl * 0.05) * nr), squeeze=False)
    bh = 0.4
    for idx, (l, h, sc, cv, kv) in enumerate(top):
        ax = axes[idx // nc, idx % nc]; y = np.arange(nl)
        ax.barh(y + bh/2, cv, bh, color=CLEAN_COLOR, alpha=BAR_ALPHA, label='Clean')
        ax.barh(y - bh/2, kv, bh, color=CORRUPTED_COLOR, alpha=BAR_ALPHA, label='Corrupted')
        _set_group_yticks(ax, dga.canonical_labels, dga.dst_label)
        ax.set_xlabel("Attention", fontsize=9); ax.set_title(f"L{l}.H{h} (max={sc:.2f})", fontsize=10)
        ax.grid(axis='x', alpha=GRID_ALPHA)
    for idx in range(len(top), nr * nc):
        axes[idx // nc, idx % nc].axis('off')
    fig.legend(handles=[mpatches.Patch(color=CLEAN_COLOR, label='Clean'),
                        mpatches.Patch(color=CORRUPTED_COLOR, label='Corrupted')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle(f"Top Attended from {dga.dst_label} — pair {pidx}", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); add_pair_label(fig, pidx)
    save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


# ─── Attention flow ─────────────────────────────────────────────────────────


def _plot_attn_flow(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels); di = list(dga.dst_position_indices)
    if not di:
        return 0
    da = di[len(di) // 2]
    # Top-5 heads
    hs: list[tuple[int, int, float]] = []
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        if not cm:
            continue
        arr = np.array(cm, dtype=float)
        for h in range(arr.shape[0]):
            hs.append((layer, h, float(arr[h].max())))
    hs.sort(key=lambda x: x[2], reverse=True); top5 = hs[:5]
    tc = {(l, h): LAYER_COLORS[i % len(LAYER_COLORS)] for i, (l, h, _) in enumerate(top5)}
    sub = out / "attn_flow"; sub.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(14, nl * 0.10), max(6, len(layers) * 0.40)))
    ly = {l: i for i, l in enumerate(layers)}
    for layer in layers:
        y = ly[layer]
        ax.scatter(range(nl), [y] * nl, s=8, c='#DDDDDD', marker='o', zorder=1)
        ax.scatter(di, [y] * len(di), s=30, c='#E91E63', marker='s', zorder=4)
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        if not cm:
            continue
        arr = np.array(cm, dtype=float); y = ly[layer]
        ma = arr.mean(axis=0)
        for ci in range(nl):
            if ma[ci] > 0.02 and ci != da:
                ax.plot([ci, da], [y, y], color='#AAAAAA', lw=ma[ci]*18+0.3, alpha=0.4, zorder=2)
    lh = []
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        if not cm:
            continue
        arr = np.array(cm, dtype=float); y = ly[layer]
        for h in range(arr.shape[0]):
            if (layer, h) not in tc:
                continue
            c = tc[(layer, h)]
            for ci in range(nl):
                w = arr[h, ci]
                if w > 0.05 and ci != da:
                    ax.plot([ci, da], [y+0.18, y+0.18], color=c, lw=w*12+0.3, alpha=0.85, zorder=3)
    for (l, h), c in tc.items():
        lh.append(mpatches.Patch(color=c, label=f"L{l}.H{h}"))
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_ylabel("Layer"); ax.set_xlabel("format_pos")
    ax.set_xlim(-1, nl); ax.set_ylim(-0.5, len(layers)-0.3)
    _set_group_xticks(ax, dga.canonical_labels, dga.dst_label)
    if lh:
        ax.legend(handles=lh, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, title='Top heads')
    ax.set_title(f"Attention Flow — pair {pidx} — dst={dga.dst_label}", fontsize=10)
    ax.grid(axis='x', alpha=GRID_ALPHA, linestyle='--')
    plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"{dga.dst_label}.png", dpi=DPI)
    return 1


# ─── Per-(dst, layer) head plots ────────────────────────────────────────────


def _plot_head_heatmaps_per_layer(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels)
    sub = out / "head_heatmaps" / dga.dst_label; sub.mkdir(parents=True, exist_ok=True)
    n = 0
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        if not cm:
            continue
        arr = np.array(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(max(16, nl * 0.16), max(6, arr.shape[0] * 0.20)))
        im = ax.imshow(arr, aspect='auto', cmap='viridis'); ax.invert_yaxis(); ax.set_ylabel("Head")
        _set_group_xticks(ax, dga.canonical_labels, dga.dst_label, fontsize=8)
        plt.colorbar(im, ax=ax, label="Attention", shrink=0.8)
        ax.set_title(f"Per-Head Heatmap — pair {pidx} — dst={dga.dst_label} — L{layer}", fontsize=10)
        plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"L{layer}.png", dpi=DPI); n += 1
    return n


def _plot_head_diff_per_layer(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels)
    sub = out / "head_diff" / dga.dst_label; sub.mkdir(parents=True, exist_ok=True)
    n = 0
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        km = _layer_get(dga.corrupted, layer) or _zeros_like_other(dga, layer)
        if not cm or not km:
            continue
        ca, ka = np.array(cm, dtype=float), np.array(km, dtype=float)
        diff = ca - ka; vabs = max(abs(diff.min()), abs(diff.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        fig, ax = plt.subplots(figsize=(max(16, nl * 0.16), max(6, ca.shape[0] * 0.20)))
        im = ax.imshow(diff, aspect='auto', cmap=DIFF_CMAP, norm=norm); ax.invert_yaxis()
        ax.set_ylabel("Head"); _set_group_xticks(ax, dga.canonical_labels, dga.dst_label, fontsize=8)
        plt.colorbar(im, ax=ax, label="Clean - Corrupted", shrink=0.8)
        ax.set_title(f"Per-Head Diff — pair {pidx} — dst={dga.dst_label} — L{layer}", fontsize=10)
        plt.tight_layout(); add_pair_label(fig, pidx); save_figure(None, sub / f"L{layer}.png", dpi=DPI); n += 1
    return n


def _plot_heads_sidebyside_per_layer(dga: DstGroupAttention, out: Path, pidx: int) -> int:
    layers = _union_layer_keys(dga)
    if not layers:
        return 0
    nl = len(dga.canonical_labels)
    sub = out / "heads_sidebyside" / dga.dst_label; sub.mkdir(parents=True, exist_ok=True)
    n = 0
    for layer in layers:
        cm = _layer_get(dga.clean, layer) or _zeros_like_other(dga, layer)
        km = _layer_get(dga.corrupted, layer) or _zeros_like_other(dga, layer)
        if not cm or not km:
            continue
        ca, ka = np.array(cm, dtype=float), np.array(km, dtype=float)
        vm = max(ca.max(), ka.max(), 0.01)
        fig, axes = plt.subplots(1, 2, figsize=(max(20, nl * 0.22), max(6, ca.shape[0] * 0.20)), squeeze=False)
        for col, (m, nm) in enumerate([(ca, "Clean"), (ka, "Corrupted")]):
            ax = axes[0, col]; im = ax.imshow(m, aspect='auto', cmap='viridis', vmin=0, vmax=vm)
            ax.invert_yaxis(); ax.set_ylabel("Head"); ax.set_title(nm, fontsize=9)
            _set_group_xticks(ax, dga.canonical_labels, dga.dst_label, fontsize=7)
            plt.colorbar(im, ax=ax, label="Attention", shrink=0.8)
        fig.suptitle(f"Per-Head Side-by-Side — pair {pidx} — dst={dga.dst_label} — L{layer}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95]); add_pair_label(fig, pidx)
        save_figure(None, sub / f"L{layer}.png", dpi=DPI); n += 1
    return n
