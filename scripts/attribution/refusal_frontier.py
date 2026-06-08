"""Phase 3' (refusal): the harm-refusal / over-refusal Pareto frontier — map vs baselines.

The discriminator, sharpened into a comparison. Four ways to install refusal into the base,
each swept over its own strength grid and scored as a point ``(benign_refusal, harmful_refusal)``:

  (a) caa     — per-layer CAA mean-difference vector ``E[δ]`` over the harmful window; add α·v_L.
  (b) arditi  — single unit refusal direction (Arditi) at the best-separating layer, added at all.
  (c) gate    — the CAA vector, but applied only where a logistic probe predicts harmful (CAST).
  (d) map     — the primal-ridge map ``W·a`` (LinearPrimalSteerHook); conditionality for free.

Want: harmful-refusal high (y), benign over-refusal low (x). The map's claim is that its frontier
*dominates* — at any matched harmful-refusal level it has lower benign over-refusal, because
``W·a ≈ 0`` on benign by construction, where a fixed vector shifts everything. The falsifiable
null: if the map merely ties (c) vector+gate, the closed-form object earns nothing over
"vector + classifier" and we say so. Tone is ``is_refusal`` on both axes; a HarmBench *leak*
diagnostic at the map's best point is reported separately (``--leak-diagnostic``).

    uv run python -m scripts.attribution.refusal_frontier \
        --config configs/attribution/refusal_transfer_llama2.yaml \
        [--suffix _smoke --n-harmful 8 --n-benign 8 --max-new 64 --leak-diagnostic]
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.attribution.collect_refusal_residuals import chat_prompt_ids
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.conditional_gate import LogisticGate
from src.probes.safety.harm_classifier import harm_verdict, load_harm_judge
from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.text_coherence import is_coherent
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import AdditionSteeringHook, LinearPrimalSteerHook


@torch.no_grad()
def generate_responses(model, tok, prompts, device, max_new, hook=None, mask=None) -> list[str]:
    """Greedy-decode each prompt. If ``hook`` given, toggle it per prompt by ``mask`` (gate)."""
    outs = []
    for i, text in enumerate(prompts):
        if hook is not None:
            hook.enabled = True if mask is None else bool(mask[i])
        ids = chat_prompt_ids(tok, text, device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def gate_residuals(model, tok, prompts, device, layer) -> np.ndarray:
    """Base last-token (refusal-decision position) residual at ``layer`` for each prompt."""
    cap = PerTokenResidualCapture(model, [layer])
    rows = []
    for text in prompts:
        cap.clear()
        with cap.capturing():
            model(chat_prompt_ids(tok, text, device), use_cache=False)
        rows.append(cap.captured[layer][-1].to(torch.float64).numpy())
    cap.remove()
    return np.stack(rows)


def best_separating_layer(base_harmful, base_benign, layers) -> tuple[int, float]:
    """Layer whose diff-in-means direction best separates harmful vs benign (standardized gap)."""
    best_l, best_sep = layers[0], -float("inf")
    for l in layers:
        bh, bb = base_harmful[l].numpy(), base_benign[l].numpy()
        if len(bh) == 0 or len(bb) == 0:
            continue
        r = refusal_direction(bh, bb)
        ph, pb = bh @ r, bb @ r
        sep = (ph.mean() - pb.mean()) / (ph.std() + pb.std() + 1e-6)
        if sep > best_sep:
            best_sep, best_l = sep, l
    return best_l, float(best_sep)


def _split_rates(prompts, resps) -> tuple[dict, list]:
    """Coherence-conditioned refusal rate for one split, plus per-prompt records.

    ``refusal`` counts only *coherent* refusals — a refusal phrase inside off-manifold
    gibberish does not count. ``coherent`` is the share of non-degenerate generations;
    a point with low coherence is degenerate (steering broke generation), not informative.
    """
    coh = [is_coherent(r) for r in resps]
    ref = [bool(is_refusal(r) and c) for r, c in zip(resps, coh)]
    n = len(resps)
    recs = [{"prompt": p, "response": r, "coherent": c, "refusal": rf}
            for p, r, c, rf in zip(prompts, resps, coh, ref)]
    return {"refusal": sum(ref) / n, "coherent": sum(coh) / n}, recs


def point(prompts_h, resp_h, prompts_b, resp_b) -> dict:
    """One frontier point: coherence-conditioned refusal on harmful (y) and benign (x)."""
    h, rh = _split_rates(prompts_h, resp_h)
    b, rb = _split_rates(prompts_b, resp_b)
    return {"harmful_refusal": h["refusal"], "harmful_coherent": h["coherent"],
            "benign_refusal": b["refusal"], "benign_coherent": b["coherent"],
            "records": {"harmful": rh, "benign": rb}}


def sweep(model, tok, harmful, benign, device, max_new, alphas, make_hook,
          mask_h=None, mask_b=None) -> list[dict]:
    """Trace one method's frontier: for each α install the hook, score, remove.

    ``mask_h``/``mask_b`` (the gated CAST method) toggle the hook per prompt within each split.
    """
    pts = []
    for a in alphas:
        hook = make_hook(a)
        rh = generate_responses(model, tok, harmful, device, max_new, hook=hook, mask=mask_h)
        rb = generate_responses(model, tok, benign, device, max_new, hook=hook, mask=mask_b)
        hook.remove()
        p = {"alpha": float(a), **point(harmful, rh, benign, rb)}
        pts.append(p)
        print(f"    α={a:<7} harmful: refusal={p['harmful_refusal']:.2f} coh={p['harmful_coherent']:.2f}"
              f"  |  benign: refusal={p['benign_refusal']:.2f} coh={p['benign_coherent']:.2f}", flush=True)
    return pts


def at_budget(points, budget) -> dict:
    """Max harmful-refusal among points with benign-refusal ≤ budget (the over-refusal cap)."""
    ok = [p for p in points if p["benign_refusal"] <= budget]
    if not ok:
        return None
    best = max(ok, key=lambda p: p["harmful_refusal"])
    return {k: v for k, v in best.items() if k != "records"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suffix", default="", help="acc/maps dir suffix (e.g. _smoke)")
    ap.add_argument("--n-harmful", type=int, default=None)
    ap.add_argument("--n-benign", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--leak-diagnostic", action="store_true",
                    help="load HarmBench and score the leak at the map's best point")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, layers = cfg["device"], list(range(cfg["num_layers"]))
    c, e = cfg["collect"], cfg["eval"]
    fr = e["frontier"]
    n_h = args.n_harmful or e["n_eval_harmful"]
    n_b = args.n_benign or e["n_eval_benign"]
    max_new = args.max_new or e["max_new"]

    # held-out eval prompts: disjoint from the fit slices used in Phase 1
    harmful = load_harmful()[c["n_harmful_fit"]: c["n_harmful_fit"] + n_h]
    benign = load_harmless()[c["n_benign_fit"]: c["n_benign_fit"] + n_b]
    print(f"eval: {len(harmful)} harmful + {len(benign)} benign held-out prompts "
          f"(max_new={max_new})", flush=True)

    acc_dir = Path(cfg["output"]["acc_dir"] + args.suffix)
    maps_dir = Path(cfg["output"]["maps_dir"] + args.suffix)
    baseline = torch.load(acc_dir / "baseline.pt")
    maps = {l: torch.load(maps_dir / f"W_L{l}.pt")["W"] for l in layers}
    caa = {l: baseline["caa"][l].to(torch.float32) for l in layers}

    # baseline directions / gate, all fit on the train split (same prompts as the map)
    arditi_layer, arditi_sep = best_separating_layer(baseline["base_harmful"], baseline["base_benign"], layers)
    r_unit = torch.tensor(refusal_direction(baseline["base_harmful"][arditi_layer].numpy(),
                                            baseline["base_benign"][arditi_layer].numpy()), dtype=torch.float32)
    gate_layer = fr["gate_layer"] if fr["gate_layer"] is not None else arditi_layer
    gate = LogisticGate().fit(baseline["base_harmful"][gate_layer].numpy(),
                              baseline["base_benign"][gate_layer].numpy())
    print(f"arditi_layer={arditi_layer} (sep={arditi_sep:.2f})  gate_layer={gate_layer} "
          f"(train_acc={gate.train_accuracy:.2f})", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["expert_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("Loading base ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=device).eval()

    # per-prompt gate masks for the CAST method (capture base residuals at the gate layer once)
    mask_h = gate.predict(gate_residuals(base, tok, harmful, device, gate_layer))
    mask_b = gate.predict(gate_residuals(base, tok, benign, device, gate_layer))
    print(f"gate fires on {mask_h.mean()*100:.0f}% harmful / {mask_b.mean()*100:.0f}% benign eval prompts",
          flush=True)

    def caa_hook(a):
        return AdditionSteeringHook(base, {l: a * caa[l] for l in layers})

    def arditi_hook(a):
        return AdditionSteeringHook(base, {l: a * r_unit for l in layers})

    def map_hook(a):
        return LinearPrimalSteerHook(base, maps, a)

    frontier = {}
    print("\n[caa] per-layer mean-difference vector"); frontier["caa"] = sweep(
        base, tok, harmful, benign, device, max_new, fr["caa_alphas"], caa_hook)
    print("[arditi] single direction @ best layer"); frontier["arditi"] = sweep(
        base, tok, harmful, benign, device, max_new, fr["arditi_alphas"], arditi_hook)
    print("[gate] CAA vector gated by logistic probe (CAST)"); frontier["gate"] = sweep(
        base, tok, harmful, benign, device, max_new, fr["caa_alphas"], caa_hook,
        mask_h=mask_h, mask_b=mask_b)
    print("[map] primal-ridge W·a"); frontier["map"] = sweep(
        base, tok, harmful, benign, device, max_new, fr["map_alphas"], map_hook)

    budget = fr["over_refusal_budget"]
    summary = {m: at_budget(pts, budget) for m, pts in frontier.items()}

    results = {"base_model": cfg["base_model"], "expert_model": cfg["expert_model"],
               "n_harmful": len(harmful), "n_benign": len(benign), "max_new": max_new,
               "arditi_layer": arditi_layer, "arditi_separation": arditi_sep,
               "gate_layer": gate_layer, "gate_train_accuracy": gate.train_accuracy,
               "gate_fire_rate": {"harmful": float(mask_h.mean()), "benign": float(mask_b.mean())},
               "over_refusal_budget": budget, "frontier": frontier,
               "best_at_budget": summary, "leak_diagnostic": None}

    if args.leak_diagnostic and summary["map"] is not None:
        del base; gc.collect(); torch.cuda.empty_cache()
        print(f"\nLoading HarmBench judge {cfg['harm_judge']} for leak diagnostic ...", flush=True)
        # (re)generate the map's best-α harmful responses, then judge content harm
        jt, judge = load_harm_judge(cfg["harm_judge"], device)
        base = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=torch.bfloat16, device_map=device).eval()
        best_a = summary["map"]["alpha"]
        hook = LinearPrimalSteerHook(base, maps, best_a)
        rh = generate_responses(base, tok, harmful, device, max_new, hook=hook)
        hook.remove()
        tone = [is_refusal(r) for r in rh]
        harm = [harm_verdict(judge, jt, p, r, device) for p, r in zip(harmful, rh)]
        results["leak_diagnostic"] = {
            "alpha": best_a, "tone_rate": sum(tone) / len(tone),
            "harm_rate": sum(harm) / len(harm),
            "leak_rate": sum(t and h for t, h in zip(tone, harm)) / len(tone)}
        print(f"leak@α={best_a}: tone={results['leak_diagnostic']['tone_rate']:.2f} "
              f"harm={results['leak_diagnostic']['harm_rate']:.2f} "
              f"leak={results['leak_diagnostic']['leak_rate']:.2f}", flush=True)

    print(f"\nbest harmful-refusal at benign-refusal ≤ {budget}:")
    for m in ("caa", "arditi", "gate", "map"):
        s = summary[m]
        print(f"  {m:<7} {('hr=%.2f @α=%s (br=%.2f)' % (s['harmful_refusal'], s['alpha'], s['benign_refusal'])) if s else 'none within budget'}",
              flush=True)

    out = Path(cfg["output"]["steer_json"].replace("refusal_transfer", "refusal_frontier").replace(
        ".json", f"{args.out_suffix}.json"))
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
