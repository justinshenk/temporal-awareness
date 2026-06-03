"""Weight-space fix: project the refusal-erosion direction out of the LoRA update.

Instead of ablating activations, edit the LoRA weights so they never WRITE along the
ICL-guided direction w (the LoRA-minus-ICL residual = the erosion axis). For the
residual-writing modules (o_proj, down_proj) at each layer:
    B' = (I - w w^T) B          (B = lora_B; columns are the rank-r output directions)
so the LoRA's contribution carries no w-component, while its task write (along u ⊥ w) is
kept. Controls: project r (label-based) and a random direction.

    uv run python -m scripts.safety.run_weight_projection \
        --config configs/safety/route_safety_qwen.yaml --icl-k 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import (
    capture_resid,
    generate_reply,
    prompt_ids,
    set_seed,
    user_turn,
)
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, icl_messages, select_valid_indices
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless

ADAPTERS = {"75": "results/safety/qwen_sweep/adapter_d75",
            "600": "results/safety/qwen_sweep/adapter_d600"}


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def project_out_of_lora(model, name, dirs):
    """Remove direction dirs[L] from the residual-writing LoRA-B of every layer."""
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            if li not in dirs:
                continue
            for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
                B = proj.lora_B[name].weight              # (d_model, r)
                w = torch.tensor(dirs[li], dtype=B.dtype, device=B.device)
                w = w / w.norm()
                B.data = B.data - torch.outer(w, w @ B.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--icl-k", type=int, default=16)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]

    harmful, harmless = load_harmful(), load_harmless()
    nh = cfg["direction"]["n_harmful"]
    h_rfit, h_sfit = harmful[:nh], harmful[nh:nh + args.n_fit]
    es = nh + args.n_fit
    h_eval = harmful[es:es + cfg["eval"]["n_harmful"]]
    s_rfit = harmless[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf, nt = cfg["ddxplus"]["n_filler"], cfg["ddxplus"]["n_task_eval"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    task_eval = build_cases(ds, valid[nf:nf + nt], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    all_layers = list(range(28))
    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, all_layers)

    def resid(model, p, k=0):
        msgs = icl_messages(tokenizer, fillers[:k], user_turn(p), mc, ft) if k else user_turn(p)
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    hr = [resid(base, p) for p in h_rfit]
    sr = [resid(base, p) for p in s_rfit]
    r_all = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in all_layers}
    base_fit = [resid(base, p) for p in h_sfit]
    icl_fit = [resid(base, p, args.icl_k) for p in h_sfit]
    u_all = {L: unit(np.mean([i[L] - b[L] for i, b in zip(icl_fit, base_fit)], axis=0)) for L in all_layers}

    def measure(model):
        return {"refusal": refusal_rate([generate_reply(model, tokenizer, prompt_ids(tokenizer, user_turn(p)),
                                                         args.device, max_new) for p in h_eval]),
                "acc": ddxplus_accuracy(model, tokenizer, task_eval, args.device, mc)[0]}

    lora_model = None
    results = {}
    for tag, adapter in ADAPTERS.items():
        # w for this adapter (needs unedited lora_fit)
        probe = f"{tag}_probe"
        if lora_model is None:
            lora_model = PeftModel.from_pretrained(base, adapter, adapter_name=probe).eval()
        else:
            lora_model.load_adapter(adapter, adapter_name=probe)
        lora_model.set_adapter(probe)
        lora_fit = [resid(lora_model, p) for p in h_sfit]
        w_all = {}
        for L in all_layers:
            lm = np.mean([lo[L] - b[L] for lo, b in zip(lora_fit, base_fit)], axis=0)
            w_all[L] = unit(lm - np.dot(lm, u_all[L]) * u_all[L])
        rng = np.random.default_rng(cfg["seed"])
        rand = {L: unit(rng.standard_normal(w_all[L].shape[0])) for L in all_layers}

        res = {}
        for cond, dirs in [("none", None), ("project_w", w_all), ("project_r", r_all), ("project_random", rand)]:
            name = f"{tag}_{cond}"
            lora_model.load_adapter(adapter, adapter_name=name)
            lora_model.set_adapter(name)
            if dirs is not None:
                project_out_of_lora(base, name, dirs)
            res[cond] = measure(lora_model)
            print(f"  LoRA-{tag} [{cond}]: refusal={res[cond]['refusal']:.3f} acc={res[cond]['acc']:.3f}")
            lora_model.delete_adapter(name)
        results[tag] = res
        lora_model.delete_adapter(probe)

    capture.remove()
    out = {"base_model": cfg["base_model"], "icl_k": args.icl_k, "n_eval": len(h_eval), "results": results}
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "weight_projection.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/weight_projection.json")


if __name__ == "__main__":
    main()
