"""Plan-vs-execute: can base finish the chain when *given* the correct working?

E1's emitted-token lens was confounded by autoregressive self-consistency (base's wrong tokens
crystallize like the LoRA's right tokens). This removes the confound by **teacher-forcing base on a
correct chain** and lensing the *gold* next token with the context held fixed to that chain. The
discriminating metric is base's teacher-forced agreement with the correct token, split by the token's
**role** in the procedure:

- high agreement on the *execution* role ⇒ base CAN do the per-step work given the working; its
  failure is in *generating/maintaining* the chain (a trajectory/planning deficit);
- low agreement there ⇒ a genuine per-step deficit even with the correct context laid out.

The roles are task-specific (``chain_token_roles``): GSM8K splits computed results (digits after
``=``) from copied digits (E1b); MuSiQue splits the sub-question (plan) from the hop answer
(execute), the final restatement, and format scaffold (P4).

Chain source follows the task. GSM8K has no in-format gold CoT, so the LoRA's own greedy CoT is
generated and verified first. MuSiQue's donor was trained on ``format_multihop_solution``, so the
**gold** chain is teacher-forced directly — exact role offsets, no anchoring, no drop rate. Either
way the donor teacher-forced on the same chain is the wiring sanity (≈ rank 0 everywhere).

    uv run python -m scripts.attribution.gold_token_lens_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
    uv run python -m scripts.attribution.gold_token_lens_gsm8k \
        --config configs/attribution/multihop_llama2.yaml --task multihop --layer 20 --n-contrast 317
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.attribution.attribution_common import (
    generate_cot_ids,
    get_task,
    gold_chain_ids,
    load_base_and_lora,
    load_contrast,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.common.bootstrap_stats import clustered_rate_gap
from src.probes.attribution.logit_lens import LogitLens
from src.probes.extraction import PerTokenResidualCapture


@torch.no_grad()
def gold_ranks(model, adapter_off, lora, full_ids, prompt_len, readout, readout_layers, lens):
    """Teacher-force ``full_ids`` and return per-CoT-position records of the gold next token's ranks.

    Position t (the residual/logits that predict token t+1) is scored for t in
    ``[prompt_len-1, seq-2]``; gold is ``full_ids[t+1]``. Records the final-logit rank, the per-layer
    lens rank, base's top-1 match, and token-type flags (is_digit, prev token == '=').
    """
    ctx = lora.disable_adapter() if adapter_off else nullcontext()
    readout.clear()
    with ctx, readout.capturing():
        logits = model(full_ids, use_cache=False).logits[0]            # (seq, vocab)

    seq = full_ids.shape[1]
    pos = torch.arange(prompt_len - 1, seq - 1, device=full_ids.device)
    gold = full_ids[0, pos + 1]                                         # (|pos|,)

    final_logit = logits[pos]
    final_gold = final_logit.gather(1, gold.view(-1, 1)).squeeze(1)
    final_rank = (final_logit > final_gold.view(-1, 1)).sum(1)          # (|pos|,)
    top1 = final_logit.argmax(1) == gold

    lens_rank = {}
    for li in readout_layers:
        ll = lens.project(readout.captured[li])[pos]
        g = ll.gather(1, gold.view(-1, 1)).squeeze(1)
        lens_rank[li] = (ll > g.view(-1, 1)).sum(1).cpu()

    return pos, gold, final_rank.cpu(), top1.cpu(), lens_rank


def classify(task, tok, full_ids, prompt_len, pos, gold):
    """Role of the *gold* token at each scored position (position ``t`` predicts token ``t+1``)."""
    roles = task.lens.token_roles(tok, full_ids[0].tolist(), prompt_len, gold)
    return [roles[t + 1] for t in pos.tolist()]


def summarize(records, readout_layers, role_classes):
    """Aggregate teacher-forced agreement and median gold rank over the task's token classes."""
    def med(xs):
        s = sorted(xs); n = len(s)
        return float("nan") if n == 0 else float(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))

    out = {}
    for name, sel in role_classes.items():
        rs = [r for r in records if sel(r)]
        if not rs:
            out[name] = {"n": 0}
            continue
        out[name] = {"n": len(rs),
                     "tf_acc": sum(r["top1"] for r in rs) / len(rs),
                     "final_rank_median": med([r["final_rank"] for r in rs]),
                     "lens_rank_median": {li: med([r["lens"][li] for r in rs]) for li in readout_layers}}
    return out


def contrast_intervals(records, role_classes, contrasts):
    """Problem-clustered 95% intervals on the role-class differences the verdict rests on.

    Tokens within one teacher-forced chain are not independent, so each problem contributes one
    resampling unit: a ``(n, hits)`` row per role class. Contrasts naming an empty class are
    reported as ``None`` rather than a spurious interval.
    """
    problems = sorted({r["problem"] for r in records})
    row_of = {p: i for i, p in enumerate(problems)}
    needed = {name for _, a, b in contrasts for name in (a, b)}
    counts = {}
    for name in needed:
        sel = role_classes[name]
        arr = np.zeros((len(problems), 2))
        for r in records:
            if sel(r):
                arr[row_of[r["problem"]]] += (1.0, float(r["top1"]))
        counts[name] = arr

    out = {}
    for label, a, b in contrasts:
        if counts[a][:, 0].sum() == 0 or counts[b][:, 0].sum() == 0:
            out[label] = None
            continue
        out[label] = clustered_rate_gap(counts[a], counts[b]).to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--readout-layers", default=None)
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    if task.lens is None:
        raise SystemExit(f"task {task.name!r} defines no gold-token lens seam")
    gold_chain = task.lens.gold_chain is not None

    print(f"Loading {cfg['base_model']} + adapter (task={task.name}) ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    n_layers = base.config.num_hidden_layers
    readout_layers = ([int(x) for x in args.readout_layers.split(",")]
                      if args.readout_layers else sorted({*range(L, n_layers, 2), n_layers - 1}))
    readout = PerTokenResidualCapture(base, readout_layers)
    lens = LogitLens(base)
    contrast = load_contrast(cfg, task)[:args.n_contrast]
    print(f"Using {len(contrast)} contrast problems; chain = "
          f"{'gold (teacher-forced)' if gold_chain else 'donor greedy CoT (verified)'}", flush=True)

    base_records, lora_tf_acc, kept = [], [], 0
    for n_seen, (q, gold) in enumerate(contrast, 1):
        if gold_chain:
            full_ids, plen = gold_chain_ids(tok, q, gold, device, task)
        else:
            full_ids, plen = generate_cot_ids(lora, tok, q, device, args.max_new, task)
            text = tok.decode(full_ids[0][plen:], skip_special_tokens=True)
            if not task.score(text, gold):
                continue                                               # only correct donor CoTs
        kept += 1
        pos, _gtok, frank, top1, lrank = gold_ranks(base, True, lora, full_ids, plen,
                                                    readout, readout_layers, lens)
        roles = classify(task, tok, full_ids, plen, pos, gold)
        for i in range(len(pos)):
            base_records.append({"final_rank": int(frank[i]), "top1": bool(top1[i]),
                                 "lens": {li: int(lrank[li][i]) for li in readout_layers},
                                 "problem": kept, **roles[i]})
        _, _, _, ltop1, _ = gold_ranks(lora, False, lora, full_ids, plen, readout, readout_layers, lens)
        lora_tf_acc.append(float(ltop1.float().mean()))
        print(f"  [{n_seen}/{len(contrast)}] kept={kept} tokens={len(base_records)}", flush=True)

    readout.remove()
    summary = summarize(base_records, readout_layers, task.lens.role_classes)
    gaps = contrast_intervals(base_records, task.lens.role_classes, task.lens.contrasts)
    results = {"task": task.name, "layer": L, "n_kept": kept, "readout_layers": readout_layers,
               "chain": "gold" if gold_chain else "donor_greedy",
               "n_tokens": len(base_records),
               "lora_tf_acc_mean": sum(lora_tf_acc) / max(len(lora_tf_acc), 1),
               "base": summary, "contrasts": gaps}

    print(f"\nbase teacher-forced on the correct CoT  (n_problems={kept}; "
          f"LoRA-TF sanity acc={results['lora_tf_acc_mean']:.3f})", flush=True)
    width = max(len(n) for n in task.lens.role_classes)
    print(f"{'class':{width}s} {'n':>6s} {'TF-acc':>7s} {'final-rank':>11s}  "
          f"lens-rank L{readout_layers[0]}..L{readout_layers[-1]}", flush=True)
    for name, s in summary.items():
        if not s.get("n"):
            print(f"{name:{width}s} {0:>6d}  (empty)", flush=True)
            continue
        lr = " ".join(f"{s['lens_rank_median'][li]:.0f}" for li in readout_layers)
        print(f"{name:{width}s} {s['n']:>6d} {s['tf_acc']:>7.3f} {s['final_rank_median']:>11.0f}  {lr}",
              flush=True)

    if gaps:
        print(f"\nTF-acc gaps, 95% bootstrap over problems (n={kept} clusters)", flush=True)
        for label, iv in gaps.items():
            body = ("(empty class)" if iv is None else
                    f"{iv['estimate']:+.3f} [{iv['lo']:+.3f}, {iv['hi']:+.3f}]"
                    f"{'' if (iv['lo'] > 0) or (iv['hi'] < 0) else '   (spans 0)'}")
            print(f"  {label:24s} {body}", flush=True)

    stem = "gold_token_lens" if task.name == "gsm8k" else f"gold_token_lens_{task.name}"
    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["steer_json"]).parent / f"{stem}_L{L}.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
