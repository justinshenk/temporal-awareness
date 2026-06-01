"""Does the harm-direction ablation recipe survive long context?

The capstone restored refusal by ablating the harm direction, but only at short context.
The headline danger was context-fragility (finetuned refusal collapses as context fills).
This measures LoRA vs LoRA+ablate-harm refusal (and DDXPlus accuracy) across context fills:

  - if ablation keeps refusal high under fill, the recipe is robust (one static direction
    removed → safe even in long context);
  - if refusal still collapses, the context-fragility is a separate, attention-mediated
    mechanism the static ablation does not touch.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_ablation_under_context \
        --config configs/safety/ablation_under_context_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import build_cases, select_valid_indices
from src.probes.safety.ablation_hook import DirectionalAblationHook
from src.probes.safety.safety_data import load_harmful
from scripts.safety.run_ablation_capstone import load_directions, set_seed
from scripts.safety.run_long_context_lora import refusal, task_acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    _, harm_dir = load_directions(cfg)

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    fillers = build_cases(ds, valid[: cfg["ddxplus"]["n_filler"]], evidence_db,
                          cfg["ddxplus"]["n_options"], cfg["seed"])
    to = cfg["ddxplus"]["task_offset"]
    cases = build_cases(ds, valid[to : to + cfg["ddxplus"]["n_task_eval"]], evidence_db,
                        cfg["ddxplus"]["n_options"], cfg["seed"])
    ho = cfg["eval"]["harmful_offset"]
    harmful = load_harmful()[ho : ho + cfg["eval"]["n_harmful"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()

    rows = []
    for fill in cfg["fills"]:
        rec = {"fill": fill}
        rec["lora_refusal"] = refusal(lora, tokenizer, fillers, harmful, fill, cfg, args.device)
        rec["lora_acc"] = task_acc(lora, tokenizer, fillers, cases, fill, cfg, args.device)
        hook = DirectionalAblationHook(base, harm_dir)
        rec["ablate_refusal"] = refusal(lora, tokenizer, fillers, harmful, fill, cfg, args.device)
        rec["ablate_acc"] = task_acc(lora, tokenizer, fillers, cases, fill, cfg, args.device)
        hook.remove()
        rows.append(rec)
        print(f"fill {fill:.2f}: LoRA ref {rec['lora_refusal']:.2f} acc {rec['lora_acc']:.2f} | "
              f"+ablate ref {rec['ablate_refusal']:.2f} acc {rec['ablate_acc']:.2f}")

    lines = [
        "# Does the harm-direction ablation survive long context?",
        "",
        f"`{cfg['base_model']}` DDXPlus LoRA | harm dir @L{cfg['directions']['layer']} ablated all "
        f"layers | {len(cases)} medical, {len(harmful)} harmful per fill | DDXPlus medical filler",
        "",
        "| Context fill | LoRA refusal | +ablate refusal | LoRA acc | +ablate acc |",
        "|-------------:|-------------:|----------------:|---------:|------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fill']:.0%} | {r['lora_refusal']:.3f} | {r['ablate_refusal']:.3f} | "
            f"{r['lora_acc']:.3f} | {r['ablate_acc']:.3f} |"
        )

    f0, fl = rows[0], rows[-1]
    lines += [
        "",
        "## Reading",
        "",
        f"- LoRA refusal across fill: {f0['lora_refusal']:.2f}→{fl['lora_refusal']:.2f} "
        f"(the context-fragility). With harm-ablation: {f0['ablate_refusal']:.2f}→"
        f"{fl['ablate_refusal']:.2f}.",
        "- If +ablate refusal stays high across fills, the static harm-direction ablation is a "
        "robust recipe — safety holds even in long context. If it still collapses, the "
        "context-fragility is a separate, attention-mediated mechanism the ablation does not fix.",
        f"- Task accuracy under ablation across fill: {f0['ablate_acc']:.2f}→{fl['ablate_acc']:.2f} "
        "(should stay near the LoRA's — the recipe must keep the task).",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
