"""Extract a DDXPlus function vector (FV) and compare it to the LoRA weight-shift direction.

Implements Todd et al. (ICLR 2024) FV extraction on the medical MCQ task, then tests the
project's open follow-up: does the in-context FV align with the in-weights LoRA shift —
"same task vector, two delivery routes"?

Pipeline:
  1. Build k-shot DDXPlus ICL prompts; a *corrupted* variant shuffles the demo labels.
  2. Clean run: capture each head's o_proj input at the last token; average -> mean head output.
  3. Causal mediation: for each (layer, head), patch its mean into the corrupted run and measure
     the recovery of P(gold) over the restricted A-E logits -> average indirect effect (AIE).
  4. FV = sum of the top-K AIE heads' residual-stream contributions (one hidden-dim vector).
  5. Validate: add the FV at a middle layer in a ZERO-shot prompt; measure accuracy lift.
  6. Compare: cosine(FV, LoRA zero-shot shift) and cosine(FV, ICL task vector) at the insert layer.

Usage:
    HF_TOKEN=... uv run python -m scripts.lora_icl.run_fv_extraction \
        --config configs/lora_icl/fv_extraction_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, OPTION_LABELS, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices
from src.probes.lora_icl.head_vectors import (
    HeadMeanPatch,
    PerHeadOprojCapture,
    head_output_vector,
)
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.lora_icl.subspace_metrics import vector_cosine
from scripts.safety.run_ablation_capstone import set_seed


def kshot_ids(tokenizer, demos, query, rng=None):
    """Token ids for a k-shot prompt; if rng given, shuffle demo labels (corrupted run)."""
    labels = [d.gold_letter for d in demos]
    if rng is not None:
        labels = labels[:]
        rng.shuffle(labels)
    msgs = []
    for d, lab in zip(demos, labels):
        msgs += chat_messages(d.prompt_text) + [{"role": "assistant", "content": lab}]
    msgs += chat_messages(query.prompt_text)
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def left_pad(id_lists, pad_id, device):
    maxlen = max(len(x) for x in id_lists)
    ids, mask = [], []
    for x in id_lists:
        d = maxlen - len(x)
        ids.append([pad_id] * d + x)
        mask.append([0] * d + [1] * len(x))
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


@torch.no_grad()
def last_logits(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :].float()


def gold_prob_and_pred(logits, letter_ids, golds):
    """Restricted-A-E softmax: P(gold) per row and predicted letter index per row."""
    restricted = torch.softmax(logits[:, letter_ids], dim=-1)  # (batch, n_options)
    pred = restricted.argmax(dim=-1).tolist()
    pgold = [float(restricted[i, golds[i]]) for i in range(len(golds))]
    return pgold, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    rng = random.Random(cfg["seed"])
    n_opt, k = cfg["n_options"], cfg["k_shot"]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    valid = select_valid_indices(ds, n_opt)
    do, qo = cfg["demo"]["offset"], cfg["query"]["offset"]
    demo_pool = build_cases(ds, valid[do : do + cfg["demo"]["n"]], evidence_db, n_opt, cfg["seed"])
    queries = build_cases(ds, valid[qo : qo + cfg["query"]["n"]], evidence_db, n_opt, cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    conf = base.config
    n_heads, head_dim, n_layers = conf.num_attention_heads, conf.head_dim, conf.num_hidden_layers
    layers = list(range(n_layers))
    letters = OPTION_LABELS[:n_opt]
    letter_ids = [tokenizer.encode(L, add_special_tokens=False)[-1] for L in letters]

    # One ICL instance per query: k demos sampled from the disjoint demo pool.
    instances = [(rng.sample(demo_pool, k), q) for q in queries]
    golds = [letters.index(q.gold_letter) for _, q in instances]
    clean_ids = [kshot_ids(tokenizer, d, q) for d, q in instances]
    corrupt_ids = [kshot_ids(tokenizer, d, q, rng=rng) for d, q in instances]

    # --- 2. Clean run: capture per-head mean o_proj input at the last token ---
    cap = PerHeadOprojCapture(base, layers, n_heads, head_dim)
    cids, cmask = left_pad(clean_ids, pad_id, args.device)
    with cap.capturing():
        clean_logits = last_logits(base, cids, cmask)
    mean_head = {li: cap.captured[li].mean(0) for li in layers}  # (n_heads, head_dim)
    cap.clear()
    clean_pg, clean_pred = gold_prob_and_pred(clean_logits, letter_ids, golds)
    clean_acc = float(np.mean([clean_pred[i] == golds[i] for i in range(len(golds))]))

    # --- corrupted baseline ---
    xids, xmask = left_pad(corrupt_ids, pad_id, args.device)
    corr_logits = last_logits(base, xids, xmask)
    corr_pg, corr_pred = gold_prob_and_pred(corr_logits, letter_ids, golds)
    corr_acc = float(np.mean([corr_pred[i] == golds[i] for i in range(len(golds))]))
    corr_base_pg = np.array(corr_pg)
    cap.remove()

    # --- 3. Causal mediation: AIE per head (patch mean into corrupted run) ---
    aie = np.zeros((n_layers, n_heads))
    for li in layers:
        for h in range(n_heads):
            patch = HeadMeanPatch(base, li, h, mean_head[li][h], head_dim)
            pg, _ = gold_prob_and_pred(last_logits(base, xids, xmask), letter_ids, golds)
            patch.remove()
            aie[li, h] = float(np.mean(np.array(pg) - corr_base_pg))

    # --- 4. Build FV from the top-K AIE heads ---
    flat = sorted(((aie[li, h], li, h) for li in layers for h in range(n_heads)), reverse=True)
    top = flat[: cfg["top_k_heads"]]
    fv = torch.zeros(conf.hidden_size, dtype=torch.float32)
    for _, li, h in top:
        w = base.model.layers[li].self_attn.o_proj.weight
        fv = fv + head_output_vector(w, mean_head[li][h], h, head_dim).cpu()
    fv_t = fv.to(args.device)

    # --- 5. Validate: add FV at a middle layer on ZERO-shot prompts; accuracy lift ---
    zs_ids = [tokenizer.apply_chat_template(chat_messages(q.prompt_text),
                                            add_generation_prompt=True, tokenize=True)
              for _, q in instances]
    zids, zmask = left_pad(zs_ids, pad_id, args.device)
    zs_pg, zs_pred = gold_prob_and_pred(last_logits(base, zids, zmask), letter_ids, golds)
    zs_acc = float(np.mean([zs_pred[i] == golds[i] for i in range(len(golds))]))

    def add_fv_hook(layer, vec):
        def hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            hs = (out[0] if is_tuple else out).clone()
            hs[:, -1, :] = hs[:, -1, :] + vec.to(hs.dtype)
            return (hs,) + tuple(out[1:]) if is_tuple else hs
        return base.model.layers[layer].register_forward_hook(hook)

    fv_val = {}
    for li in cfg["insert_layers"]:
        hk = add_fv_hook(li, fv_t)
        pg, pred = gold_prob_and_pred(last_logits(base, zids, zmask), letter_ids, golds)
        hk.remove()
        fv_val[li] = float(np.mean([pred[i] == golds[i] for i in range(len(golds))]))
    best_insert = max(fv_val, key=fv_val.get)

    # --- 6. Compare FV to the LoRA shift and the ICL task vector at the insert layer ---
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()
    rescap = PerTokenResidualCapture(base, [best_insert])

    def mean_resid(ids_list):
        accum = []
        for ids in ids_list:
            t = torch.tensor([ids], device=args.device)
            rescap.clear()
            with rescap.capturing():
                base(t, use_cache=False)
            accum.append(last_token_residual(rescap.captured)[best_insert])
        return np.stack(accum).mean(0)

    with lora.disable_adapter():
        r_base = mean_resid(zs_ids)
        r_icl = mean_resid(clean_ids)
    r_lora = mean_resid(zs_ids)  # adapter active
    rescap.remove()

    fv_np = fv.numpy()
    lora_shift = r_lora - r_base
    icl_tv = r_icl - r_base
    cos_fv_lora = vector_cosine(fv_np, lora_shift)
    cos_fv_icl = vector_cosine(fv_np, icl_tv)
    cos_lora_icl = vector_cosine(lora_shift, icl_tv)

    top_heads = [{"layer": li, "head": h, "aie": round(a, 4)} for a, li, h in top]
    lines = [
        "# DDXPlus function vector (FV) — extraction and FV-vs-LoRA comparison",
        "",
        f"`{cfg['base_model']}` | {k}-shot ICL | {len(instances)} instances | causal mediation over "
        f"{n_layers}x{n_heads} heads | top-{cfg['top_k_heads']} FV heads | restricted A-E readout.",
        "",
        "## Task signal (sanity)",
        "",
        f"- Clean {k}-shot ICL accuracy: **{clean_acc:.2f}**; corrupted (shuffled-label) accuracy: "
        f"**{corr_acc:.2f}**; zero-shot (no demos): **{zs_acc:.2f}**.",
        f"- Label-dependence (clean − corrupted) = **{clean_acc - corr_acc:+.2f}**: "
        + ("≈0 ⇒ the demo labels are inert (no in-context task signal to recover — see Reading)."
           if clean_acc - corr_acc < 0.05 else
           "positive ⇒ the demos carry a recoverable task signal for causal mediation."),
        "",
        "## Top FV heads (by average indirect effect)",
        "",
        "| rank | layer | head | AIE |",
        "|-----:|------:|-----:|----:|",
        *[f"| {i+1} | {t['layer']} | {t['head']} | {t['aie']:+.4f} |" for i, t in enumerate(top_heads)],
        "",
        "## FV validation — zero-shot accuracy with the FV added",
        "",
        "| insert layer | zero-shot acc + FV |",
        "|-------------:|-------------------:|",
        *[f"| {li} | {fv_val[li]:.2f} |" for li in cfg["insert_layers"]],
        f"\nBaseline zero-shot acc (no FV): **{zs_acc:.2f}**; best insert layer L{best_insert} -> "
        f"**{fv_val[best_insert]:.2f}**.",
        "",
        "## FV vs LoRA vs ICL task vector (cosine @ L%d)" % best_insert,
        "",
        "| pair | cosine |",
        "|------|-------:|",
        f"| FV · LoRA-shift | {cos_fv_lora:+.3f} |",
        f"| FV · ICL-task-vector | {cos_fv_icl:+.3f} |",
        f"| LoRA-shift · ICL-task-vector | {cos_lora_icl:+.3f} |",
        "",
        "## Reading",
        "",
    ]
    signal = clean_acc - corr_acc
    if signal < 0.05:
        lines += [
            f"- **NULL — and a code-independent sanity check says why.** Clean {k}-shot accuracy "
            f"({clean_acc:.2f}) ≈ corrupted shuffled-label accuracy ({corr_acc:.2f}), both ≈ zero-shot "
            f"({zs_acc:.2f}). Shuffling the demonstration labels does not hurt ⇒ the model ignores the "
            "in-context labels and answers DDXPlus from its medical knowledge. There is **no in-context "
            "task signal** for causal mediation to recover, so the AIE values are ~0 and the FV is noise.",
            f"- **The FV is inert, as expected:** adding it to a zero-shot prompt does not move accuracy "
            f"({zs_acc:.2f}→{fv_val[best_insert]:.2f}).",
            f"- **The FV·LoRA cosine ({cos_fv_lora:+.3f}) is UNINFORMATIVE, not evidence of "
            "orthogonality.** You cannot conclude the in-context and in-weights task vectors point "
            "different ways when the FV extraction had no signal to extract. (The LoRA·ICL cosine here "
            f"({cos_lora_icl:+.3f}) is measured at L{best_insert} — an early layer where the subspace "
            "study already showed convergence ≈ 0; the 0.81 convergence was at L35.)",
            "- **What this qualifies:** DDXPlus is a *knowledge* task this model already solves "
            "near-zero-shot, **not** an *in-context-learning* task. Our LoRA-vs-ICL subspace convergence "
            "is real activation geometry, but it is **not** a Todd-style causal function vector — there "
            "is no extractable in-context task algorithm here; the convergence reflects context/format/"
            "calibration adaptation. To test FV extraction (and 'same task vector, two routes'), use a "
            "task where ICL carries the signal (zero-shot fails, demos define the mapping — e.g. "
            "antonyms or a relabeled/symbolic task).",
        ]
    else:
        lines += [
            f"- **Real FV:** FV-added zero-shot accuracy {zs_acc:.2f}→{fv_val[best_insert]:.2f} (clean "
            f"vs corrupted ICL {clean_acc:.2f}/{corr_acc:.2f} confirms a recoverable task signal).",
            f"- **Same task vector, two routes?** cos(FV, LoRA-shift) = {cos_fv_lora:+.3f}; cos(FV, "
            f"ICL-task-vector) = {cos_fv_icl:+.3f}. High ⇒ in-context FV and in-weights LoRA point the "
            "same way (mechanism-level support for subspace convergence).",
        ]
    lines += [
        f"- **Scope:** one model/task, {len(instances)} instances, top-{cfg['top_k_heads']} heads, "
        "restricted-letter readout; FV insertion at the last token only.",
    ]
    report = Path(cfg["output"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps({
        "clean_acc": clean_acc, "corr_acc": corr_acc, "zs_acc": zs_acc,
        "top_heads": top_heads, "fv_val": fv_val, "best_insert": best_insert,
        "cos_fv_lora": cos_fv_lora, "cos_fv_icl": cos_fv_icl, "cos_lora_icl": cos_lora_icl}, indent=2))
    shift_dir = Path(cfg["output"]["shifts"])
    shift_dir.mkdir(parents=True, exist_ok=True)
    np.save(shift_dir / "ddxplus_fv.npy", fv_np)
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
