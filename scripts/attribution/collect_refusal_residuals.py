"""Phase 1 (refusal): stream per-token (a_t, δ_t) over the refusal-decision window.

Mirror of ``collect_cot_residuals.py`` for the safety axis. For each prompt in a MIXED
harmful+benign set: greedy-generate ``k_decode`` tokens with the EXPERT (chat), then
teacher-force that exact ``prompt+gen`` sequence through base and expert, capturing
all-token residuals at every layer in two ``use_cache=False`` passes. Over the
refusal-decision window (last prompt token + early decode): a_t = base residual,
δ_t = expert − base. Accumulate into a train and a held-out set of GramAccumulators.

The mixed set is the whole point: δ is large where the expert diverges (harmful handling)
and ~0 where it does not (benign), so the ridge learns the harm-conditionality for free —
no hand-built gating. Two separate full models (base, chat) share a vocab, so the chat
tokenizer's [INST] template builds ids fed identically to both.

    uv run python -m scripts.attribution.collect_refusal_residuals \
        --config configs/attribution/refusal_transfer_llama2.yaml \
        [--n-harmful 8 --n-benign 8 --k-decode 8 --out-suffix _smoke]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks
from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.refusal_collection import refusal_token_slice
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.safety_data import load_harmful, load_harmless


def chat_prompt_ids(tokenizer, text: str, device) -> torch.Tensor:
    """Llama-2 ``[INST] … [/INST]`` chat-formatted prompt ids ((1, L) on ``device``).

    The NousResearch Llama-2 tokenizers ship no ``chat_template``, so build the canonical
    single-user-turn format directly; the tokenizer prepends ``<s>`` (BOS). The same ids feed
    base and expert (shared vocab).
    """
    return tokenizer(f"[INST] {text.strip()} [/INST]", return_tensors="pt").input_ids.to(device)


def load_base_and_expert(cfg) -> tuple:
    """Tokenizer (chat, with [INST] template) + base + expert as two separate full models."""
    tok = AutoTokenizer.from_pretrained(cfg["expert_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=cfg["device"]).eval()
    expert = AutoModelForCausalLM.from_pretrained(
        cfg["expert_model"], torch_dtype=torch.bfloat16, device_map=cfg["device"]).eval()
    return tok, base, expert


def build_fit_prompts(n_harmful: int, n_benign: int, seed: int) -> list[tuple[str, str]]:
    """Mixed (text, label) prompts: first ``n_harmful`` AdvBench + first ``n_benign`` Alpaca, shuffled."""
    harmful = load_harmful()[:n_harmful]
    benign = load_harmless()[:n_benign]
    prompts = [(t, "harmful") for t in harmful] + [(t, "benign") for t in benign]
    random.Random(seed).shuffle(prompts)
    return prompts


@torch.no_grad()
def collect_split(prompts, base, expert, cap_base, cap_expert, tokenizer, layers, dim,
                  device, accum_dev, k_decode, tag) -> tuple[dict, dict, dict]:
    """Stream (a, δ) into per-layer Gram accumulators; also collect the baseline ingredients.

    ``accs`` feeds the ridge map (method d). ``baseline`` carries what the steering baselines
    need, fit on the SAME prompts: the per-layer CAA vector ``E[δ]`` over the harmful window
    (method a), and base last-token (refusal-decision position) residuals split by label, for
    the Arditi direction (method b) and the logistic gate (method c).
    """
    accs = {l: GramAccumulator(dim, device=accum_dev, layer=l) for l in layers}
    caa_sum = {l: torch.zeros(dim, dtype=torch.float64) for l in layers}
    base_harmful = {l: [] for l in layers}
    base_benign = {l: [] for l in layers}
    counts = {"tokens": 0, "harmful": 0, "benign": 0, "harmful_tokens": 0}
    for i, (text, label) in enumerate(prompts):
        ids_t = chat_prompt_ids(tokenizer, text, device)
        full_ids = expert.generate(ids_t, max_new_tokens=k_decode, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        sl = refusal_token_slice(ids_t.shape[1], full_ids.shape[1])
        base_cap = teacher_force_capture(base, cap_base, full_ids)
        exp_cap = teacher_force_capture(expert, cap_expert, full_ids)
        for l in layers:
            a, d = assemble_blocks(base_cap, exp_cap, l, sl)  # δ = expert − base
            accs[l].update(a, d)
            (base_harmful if label == "harmful" else base_benign)[l].append(a[0].cpu())
            if label == "harmful":
                caa_sum[l] += d.sum(dim=0).cpu()
        counts["tokens"] += sl.stop - sl.start
        counts[label] += 1
        if label == "harmful":
            counts["harmful_tokens"] += sl.stop - sl.start
        if (i + 1) % 20 == 0 or i == len(prompts) - 1:
            print(f"  [{tag}] {i+1}/{len(prompts)} prompts, {counts['tokens']} window tokens "
                  f"({counts['harmful']} harmful / {counts['benign']} benign)", flush=True)
    n_h = max(counts["harmful_tokens"], 1)
    baseline = {"caa": {l: (caa_sum[l] / n_h) for l in layers},
                "base_harmful": {l: torch.stack(base_harmful[l]) if base_harmful[l] else torch.empty(0, dim)
                                 for l in layers},
                "base_benign": {l: torch.stack(base_benign[l]) if base_benign[l] else torch.empty(0, dim)
                                for l in layers}}
    return accs, counts, baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-harmful", type=int, default=None)
    ap.add_argument("--n-benign", type=int, default=None)
    ap.add_argument("--k-decode", type=int, default=None)
    ap.add_argument("--out-suffix", default="", help="subdir suffix for smoke runs, e.g. _smoke")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    c = cfg["collect"]
    n_harmful = args.n_harmful or c["n_harmful_fit"]
    n_benign = args.n_benign or c["n_benign_fit"]
    k_decode = args.k_decode or c["k_decode"]
    device, accum_dev, dim = cfg["device"], cfg["accum_device"], cfg["hidden_dim"]
    layers = list(range(cfg["num_layers"]))

    print(f"Loading base={cfg['base_model']}  expert={cfg['expert_model']} ...", flush=True)
    tokenizer, base, expert = load_base_and_expert(cfg)
    cap_base = PerTokenResidualCapture(base, layers)
    cap_expert = PerTokenResidualCapture(expert, layers)

    prompts = build_fit_prompts(n_harmful, n_benign, cfg["seed"])
    n_heldout = max(1, int(len(prompts) * c["heldout_frac"]))
    heldout, train = prompts[:n_heldout], prompts[n_heldout:]
    print(f"fit={len(train)} held-out={len(heldout)} prompts "
          f"(harmful≈{n_harmful}, benign≈{n_benign}, k_decode={k_decode})", flush=True)

    acc_tr, ntr, baseline = collect_split(train, base, expert, cap_base, cap_expert, tokenizer,
                                          layers, dim, device, accum_dev, k_decode, "train")
    acc_te, nte, _ = collect_split(heldout, base, expert, cap_base, cap_expert, tokenizer, layers,
                                   dim, device, accum_dev, k_decode, "heldout")
    cap_base.remove()
    cap_expert.remove()

    out_dir = Path(cfg["output"]["acc_dir"] + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    for l in layers:
        torch.save(acc_tr[l].state_dict(), out_dir / f"train_L{l}.pt")
        torch.save(acc_te[l].state_dict(), out_dir / f"heldout_L{l}.pt")
    # baseline ingredients (fit on the train split, same prompts as the ridge map)
    torch.save(baseline, out_dir / "baseline.pt")
    meta = {"base_model": cfg["base_model"], "expert_model": cfg["expert_model"],
            "n_harmful": n_harmful, "n_benign": n_benign, "k_decode": k_decode,
            "layers": layers, "hidden_dim": dim, "train": ntr, "heldout": nte,
            "train_tokens_per_layer": acc_tr[layers[0]].n_tokens}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(layers)*2} accumulators + meta.json to {out_dir}", flush=True)
    print(f"train window tokens={ntr['tokens']}  held-out window tokens={nte['tokens']}", flush=True)


if __name__ == "__main__":
    main()
