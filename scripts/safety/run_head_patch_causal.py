"""Causal test: do the LoRA-rerouted attention heads CAUSE the refusal erosion?

For each harmful prompt, capture the BASE model's per-head attention output (the o_proj
input slice) at the last token, then run the LoRA with those head outputs patched back in
at just the top-divergence heads (from attention_base_vs_lora.json). If refusal recovers,
those heads route the erosion; if not, the attention change is a correlate and the action
is elsewhere (values / MLP). Controls: patch random heads (specificity) and all heads at
the involved layers (upper bound).

    uv run python -m scripts.safety.run_head_patch_causal \
        --config configs/safety/route_safety_qwen.yaml --adapter results/safety/qwen_sweep/adapter_d600 --topk 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import prompt_ids, set_seed, user_turn
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.safety_data import load_harmful


class HeadPatch:
    """Capture base last-token per-head attention output, then patch it into a forward."""

    def __init__(self, model, layers, head_dim):
        self.head_dim, self.mode, self.store, self.heads = head_dim, "off", {}, {}
        self._hooks = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(self._mk(L))
                       for L in layers]

    def _mk(self, L):
        def hook(module, args):
            x = args[0]
            if x.shape[1] <= 1:  # prefill only
                return None
            if self.mode == "capture":
                self.store[L] = x[0, -1, :].detach().clone()
            elif self.mode == "patch" and L in self.heads:
                x = x.clone()
                for hd in self.heads[L]:
                    sl = slice(hd * self.head_dim, (hd + 1) * self.head_dim)
                    x[:, -1, sl] = self.store[L][sl].to(x.device, x.dtype)
                return (x,)
            return None
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def to_map(pairs):
    m = {}
    for L, h in pairs:
        m.setdefault(L, []).append(h)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-harmful", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    max_new = cfg["eval"]["max_new"]
    out_dir = Path(cfg["output"]["dir"])

    attn = json.loads((out_dir / "attention_base_vs_lora.json").read_text())
    top = [(t["layer"], t["head"]) for t in attn["harmful"]["top_heads"][:args.topk]]
    rng = np.random.default_rng(cfg["seed"])
    layers_involved = sorted({L for L, _ in top})
    n_heads = 28
    rand = [(int(rng.integers(0, 28)), int(rng.integers(0, n_heads))) for _ in range(args.topk)]
    all_at_layers = [(L, h) for L in layers_involved for h in range(n_heads)]
    print(f"top heads: {top}")

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()
    head_dim = base.config.hidden_size // base.config.num_attention_heads
    patch = HeadPatch(base, sorted({L for L, _ in top + rand + all_at_layers}), head_dim)

    def gen(p):
        ids = prompt_ids(tokenizer, user_turn(p))
        out = lora_model.generate(torch.tensor([ids], device=args.device), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)

    def run_condition(head_map):
        outs = []
        for p in h_eval:
            ids = prompt_ids(tokenizer, user_turn(p))
            patch.mode = "capture"
            with torch.no_grad(), lora_model.disable_adapter():
                base(torch.tensor([ids], device=args.device), use_cache=False)
            patch.heads, patch.mode = head_map, "patch"
            outs.append(gen(p))
            patch.mode = "off"
        return refusal_rate(outs)

    results = {}
    patch.mode = "off"
    with lora_model.disable_adapter():
        results["base"] = refusal_rate([gen(p) for p in h_eval])
    results["lora_none"] = refusal_rate([gen(p) for p in h_eval])
    results[f"lora_patch_top{args.topk}"] = run_condition(to_map(top))
    results["lora_patch_random"] = run_condition(to_map(rand))
    results["lora_patch_all_at_layers"] = run_condition(to_map(all_at_layers))
    patch.remove()

    for k, v in results.items():
        print(f"  {k:28s} refusal={v:.3f}")
    (out_dir / "head_patch_causal.json").write_text(json.dumps(
        {"top_heads": top, "random_heads": rand, "layers_involved": layers_involved,
         "n_harmful": len(h_eval), "refusal": results}, indent=2))
    print(f"\nSaved {out_dir}/head_patch_causal.json")


if __name__ == "__main__":
    main()
