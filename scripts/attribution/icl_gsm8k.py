"""Control: does few-shot ICL rescue GSM8K where static activation steering can't?

Static injection either washes out (prefill-only) or compounds into loops/salad (all-step),
and recovers 0.00 of the LoRA budget. ICL is the dynamic alternative: the demonstrations sit
in context and are re-attended at every decode step, regenerating a coherent reasoning state
rather than re-adding a fixed shift. If base + K-shot ICL recovers accuracy, the wall is the
static-injection mechanism, not the base lacking the capability.

    uv run python -m scripts.attribution.icl_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--shots 2,4,8 --n-eval 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import gsm8k_demos, gsm8k_problems, load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    metamath_fewshot_prompt,
    numeric_match,
)

# the model continues by hallucinating the next example; cut the completion at that boundary
_STOP_MARKERS = ("\n\n### Instruction", "\nBelow is an instruction", "\n\nBelow is an instruction")


@torch.no_grad()
def icl_accuracy(model, tokenizer, demos, problems, device, max_new) -> float:
    correct = 0
    for question, gold in problems:
        prompt = metamath_fewshot_prompt(question, demos)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        for marker in _STOP_MARKERS:
            text = text.split(marker)[0]
        if numeric_match(extract_pred_number(text), gold):
            correct += 1
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shots", default="2,4,8", help="comma list of K (number of ICL demos)")
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--base-acc", type=float, default=None, help="known base zero-shot acc (for budget)")
    ap.add_argument("--lora-acc", type=float, default=None, help="known LoRA zero-shot acc (for budget)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = cfg["device"]
    n_eval = args.n_eval or cfg["eval"]["n_eval"]
    max_new = cfg["eval"]["max_new"]
    shot_counts = [int(x) for x in args.shots.split(",")]

    tokenizer, base, lora = load_base_and_lora(cfg)
    problems = gsm8k_problems(cfg["eval"]["split"], n_eval, skip=0)
    demos_all = gsm8k_demos(max(shot_counts), split="train", skip=0)

    base_acc = args.base_acc
    lora_acc = args.lora_acc
    if base_acc is None:
        from scripts.attribution.attribution_common import gsm8k_accuracy
        with lora.disable_adapter():
            base_acc = gsm8k_accuracy(base, tokenizer, problems, device, max_new)
    print(f"REFERENCE base 0-shot={base_acc:.3f}" + (f"  LoRA 0-shot={lora_acc:.3f}" if lora_acc else ""),
          flush=True)

    results = {"base_acc": base_acc, "lora_acc": lora_acc, "n_eval": len(problems), "shots": {}}
    for k in shot_counts:
        with lora.disable_adapter():
            acc = icl_accuracy(base, tokenizer, demos_all[:k], problems, device, max_new)
        recov = (acc - base_acc) / (lora_acc - base_acc) if lora_acc and lora_acc > base_acc else None
        results["shots"][k] = {"icl_acc": acc, "recovery": recov}
        msg = f"  base + {k}-shot ICL: acc={acc:.3f}"
        if recov is not None:
            msg += f"  recovery={recov:+.2f}"
        print(msg, flush=True)

    out = Path(cfg["output"]["sweep_json"].replace("sweep.json", "icl_gsm8k.json"))
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
