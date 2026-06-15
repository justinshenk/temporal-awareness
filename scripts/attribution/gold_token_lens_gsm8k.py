"""Format-vs-compute: can base finish the answer when *given* the correct working?

E1's emitted-token lens was confounded by autoregressive self-consistency (base's wrong tokens
crystallize like the LoRA's right tokens). This removes the confound by **teacher-forcing base on the
LoRA's correct CoT** and lensing the *gold* next token with the context held fixed to the correct
chain. The discriminating metric is base's teacher-forced agreement with the correct token on
**computed-result positions** (a digit immediately following ``=``):

- high agreement / gold token decodable deep in base ⇒ base CAN do the per-step arithmetic given the
  working; its failure is in *generating/maintaining* the chain (a trajectory/format deficit);
- low agreement / gold token high-rank throughout ⇒ a genuine per-step compute deficit even with the
  correct context laid out.

For each base-fails/LoRA-solves contrast problem we greedily generate the LoRA's (correct) CoT, verify
it, teacher-force it through base (adapter off) capturing readout-layer residuals, and for every CoT
position record base's rank of the gold next token (final logits + per-layer logit lens), classified
by token type. LoRA teacher-forced on its own chain is the wiring sanity (≈ rank 0 everywhere).

    uv run python -m scripts.attribution.gold_token_lens_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import generate_cot_ids, load_base_and_lora
from scripts.attribution.nonlinear_delta_gsm8k import load_contrast
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
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


def computed_flags(tok, full_ids):
    """Per-token flag: is this a *computed-result* digit — a digit in the space/digit run after '='?

    The Llama tokenizer emits the inter-token space as its own piece and splits multi-digit numbers
    per digit, so '= 48' tokenizes as '=', '', '4', '8'. A state machine opens a result span on any
    token containing '=', keeps it open across space/digit tokens (flagging the digits), and closes it
    on the first other token. This isolates genuine arithmetic results from copied digits (problem
    numbers, the 'The answer is: N' restatement).
    """
    decoded = [tok.decode([int(i)]) for i in full_ids[0].tolist()]
    flags = [False] * len(decoded)
    in_result = False
    for k, d in enumerate(decoded):
        if "=" in d:
            in_result = True
        elif in_result:
            if d.strip() == "":
                continue                                               # space token: stay in span
            if any(c.isdigit() for c in d):
                flags[k] = True                                        # computed-result digit
            else:
                in_result = False                                      # any other token closes the span
    return decoded, flags


def classify(tok, full_ids, pos, gold):
    """Per-scored-position flags: is_digit(gold) and whether gold is a computed-result digit."""
    decoded, comp = computed_flags(tok, full_ids)
    out = []
    for i, t in enumerate(pos.tolist()):
        gi = t + 1                                                     # gold token index in the sequence
        out.append({"is_digit": any(c.isdigit() for c in decoded[gi]), "computed": comp[gi]})
    return out


def summarize(records, readout_layers):
    """Aggregate teacher-forced agreement and median gold rank over token classes."""
    def med(xs):
        s = sorted(xs); n = len(s)
        return float("nan") if n == 0 else float(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))

    classes = {"all": lambda r: True,
               "digit": lambda r: r["is_digit"],
               "computed (result of =)": lambda r: r["computed"],
               "copied digit (not computed)": lambda r: r["is_digit"] and not r["computed"]}
    out = {}
    for name, sel in classes.items():
        rs = [r for r in records if sel(r)]
        if not rs:
            out[name] = {"n": 0}
            continue
        out[name] = {"n": len(rs),
                     "tf_acc": sum(r["top1"] for r in rs) / len(rs),
                     "final_rank_median": med([r["final_rank"] for r in rs]),
                     "lens_rank_median": {li: med([r["lens"][li] for r in rs]) for li in readout_layers}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--readout-layers", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    n_layers = base.config.num_hidden_layers
    readout_layers = ([int(x) for x in args.readout_layers.split(",")]
                      if args.readout_layers else sorted({*range(L, n_layers, 2), n_layers - 1}))
    readout = PerTokenResidualCapture(base, readout_layers)
    lens = LogitLens(base)
    contrast = load_contrast(cfg)[:args.n_contrast]

    base_records, lora_tf_acc, kept = [], [], 0
    for q, gold in contrast:
        full_ids, plen = generate_cot_ids(lora, tok, q, device, args.max_new)
        text = tok.decode(full_ids[0][plen:], skip_special_tokens=True)
        if not numeric_match(extract_pred_number(text), gold):
            continue                                                   # only correct LoRA CoTs
        kept += 1
        pos, gtok, frank, top1, lrank = gold_ranks(base, True, lora, full_ids, plen,
                                                   readout, readout_layers, lens)
        flags = classify(tok, full_ids, pos, gtok)
        for i in range(len(pos)):
            base_records.append({"final_rank": int(frank[i]), "top1": bool(top1[i]),
                                 "lens": {li: int(lrank[li][i]) for li in readout_layers},
                                 **flags[i]})
        _, _, _, ltop1, _ = gold_ranks(lora, False, lora, full_ids, plen, readout, readout_layers, lens)
        lora_tf_acc.append(float(ltop1.float().mean()))

    readout.remove()
    summary = summarize(base_records, readout_layers)
    results = {"layer": L, "n_kept": kept, "readout_layers": readout_layers,
               "lora_tf_acc_mean": sum(lora_tf_acc) / max(len(lora_tf_acc), 1),
               "base": summary}

    print(f"\nbase teacher-forced on the correct CoT  (n_problems={kept}; "
          f"LoRA-TF sanity acc={results['lora_tf_acc_mean']:.3f})", flush=True)
    print(f"{'class':24s} {'n':>5s} {'TF-acc':>7s} {'final-rank':>11s}  lens-rank L20..L31", flush=True)
    for name, s in summary.items():
        if not s.get("n"):
            continue
        lr = " ".join(f"{s['lens_rank_median'][li]:.0f}" for li in readout_layers)
        print(f"{name:24s} {s['n']:>5d} {s['tf_acc']:>7.3f} {s['final_rank_median']:>11.0f}  {lr}",
              flush=True)

    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["steer_json"]).parent / f"gold_token_lens_L{L}.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
