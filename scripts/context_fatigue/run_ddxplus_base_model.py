"""
Full pipeline on Gemma 2 9B BASE (non-IT) with PT SAEs.

Runs DDXPlus MCQ accumulation with:
  - Entropy/accuracy tracking
  - Residual stream probes (correct/incorrect, context fill, clean/fatigued)
  - MLP vs attention sublayer decomposition
  - SAE feature tracking at layers 9, 20, 31 using pretrained PT SAEs
  - Cosine similarity drift

No chat template — uses raw text completion since base model has no
instruction-following training.
"""

import json, re, ast, random, gc, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


OPTION_LABELS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b")
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--fill-target", type=float, default=0.88)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--out-dir", default="results/ddxplus_base_model")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ── evidence helpers ────────────────────────────────────────────────────

def load_evidence_db(path):
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        vm = {vk: (vv.get("en", str(vv)) if isinstance(vv, dict) else str(vv))
              for vk, vv in info.get("value_meaning", {}).items()}
        db[code] = {"question": info.get("question_en", ""),
                    "is_antecedent": info.get("is_antecedent", False),
                    "data_type": info.get("data_type", "B"), "value_meanings": vm}
    return db


def decode_evidence(ev_str, evidence_db):
    evs = ast.literal_eval(ev_str)
    symptoms, antecedents = [], []
    grouped = {}
    for ev in evs:
        if "@" in ev:
            b, v = ev.split("@", 1)
            grouped.setdefault(b.strip().rstrip("_"), []).append(v.strip())
        else:
            grouped[ev] = []
    for code, vals in grouped.items():
        if code not in evidence_db: continue
        info = evidence_db[code]
        stmt = info["question"].replace("Do you have ", "Has ").replace("Are you ", "Is ").rstrip("?.")
        if info["data_type"] == "B": t = f"Yes — {stmt}"
        elif info["data_type"] == "M" and vals:
            dec = [info["value_meanings"].get(v, v) for v in vals if info["value_meanings"].get(v, v) != "NA"]
            t = f"{stmt}: {', '.join(dec)}" if dec else f"Yes — {stmt}"
        else: t = f"Yes — {stmt}"
        (antecedents if info["is_antecedent"] else symptoms).append(t)
    return symptoms, antecedents


def format_case_mcq(age, sex, initial_ev, evidence_str, evidence_db, options):
    sex_full = "Male" if sex == "M" else "Female"
    chief = evidence_db.get(initial_ev, {}).get("question", initial_ev).replace("Do you have ", "").replace("?", "").strip()
    symptoms, antecedents = decode_evidence(evidence_str, evidence_db)
    lines = [f"Patient: {age}-year-old {sex_full}", f"Chief complaint: {chief}"]
    if symptoms:
        lines.append("Symptoms:")
        lines.extend(f"  - {s}" for s in symptoms)
    if antecedents:
        lines.append("History:")
        lines.extend(f"  - {a}" for a in antecedents)
    lines.append("\nMost likely diagnosis:")
    lines.extend(f"{OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(options[:5]))
    lines.append("\nAnswer:")
    return "\n".join(lines)


def extract_mcq_answer(text):
    text = text.strip().upper()
    if text and text[0] in "ABCDE": return text[0]
    m = re.search(r'\b([ABCDE])\b', text)
    return m.group(1) if m else None


# ── Sublayer capture ────────────────────────────────────────────────────

class SublayerCapture:
    def __init__(self, model, target_layers):
        self.attn_out = {li: None for li in target_layers}
        self.mlp_out = {li: None for li in target_layers}
        self.res_out = {li: None for li in target_layers}
        self.hooks = []
        self.enabled = False

        for li in target_layers:
            layer = model.model.layers[li]
            self.hooks.append(layer.self_attn.register_forward_hook(self._attn_hook(li)))
            self.hooks.append(layer.mlp.register_forward_hook(self._mlp_hook(li)))
            self.hooks.append(layer.register_forward_hook(self._res_hook(li)))

    def _attn_hook(self, li):
        def hook(mod, inp, out):
            if not self.enabled: return
            o = out[0] if isinstance(out, tuple) else out
            self.attn_out[li] = o[0, -1, :].detach().float().cpu()
        return hook

    def _mlp_hook(self, li):
        def hook(mod, inp, out):
            if not self.enabled: return
            o = out[0] if isinstance(out, tuple) else out
            self.mlp_out[li] = o[0, -1, :].detach().float().cpu()
        return hook

    def _res_hook(self, li):
        def hook(mod, inp, out):
            if not self.enabled: return
            o = out[0] if isinstance(out, tuple) else out
            self.res_out[li] = o[0, -1, :].detach().float().cpu()
        return hook

    def clear(self):
        for li in self.attn_out:
            self.attn_out[li] = None
            self.mlp_out[li] = None
            self.res_out[li] = None

    def remove(self):
        for h in self.hooks: h.remove()


# ── Linear probe ────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, x):
        return self.w(x).squeeze(-1)


def train_probe(X, y, n_folds=5, n_epochs=500, lr=1e-3):
    n = X.shape[0]; fs = n // n_folds; accs = []
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    X, y = X[perm], y[perm]
    for fold in range(n_folds):
        vs, ve = fold*fs, ((fold+1)*fs if fold < n_folds-1 else n)
        ti = list(range(0,vs)) + list(range(ve,n)); vi = list(range(vs,ve))
        if len(ti)<5 or len(vi)<2: continue
        Xt,Xv,yt,yv = X[ti],X[vi],y[ti],y[vi]
        if yt.unique().numel()<2: continue
        p = LinearProbe(X.shape[1])
        opt = optim.Adam(p.parameters(), lr=lr, weight_decay=1e-4)
        p.train()
        for _ in range(n_epochs):
            opt.zero_grad(); nn.BCEWithLogitsLoss()(p(Xt), yt).backward(); opt.step()
        p.eval()
        with torch.no_grad():
            lo = p(Xv); accs.append(((lo>0).float() == yv).float().mean().item())
    return np.mean(accs) if accs else 0


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_layers = [9, 20, 31]

    evidence_db = load_evidence_db("release_evidences.json")

    print(f"Loading {args.model} (BASE, non-IT)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load PT SAEs (no mismatch with base model)
    print("Loading PT SAEs...")
    sae_ids = {9: "layer_9/width_16k/average_l0_51",
               20: "layer_20/width_16k/average_l0_68",
               31: "layer_31/width_16k/average_l0_63"}
    saes = {}
    for li in probe_layers:
        sae = SAE.from_pretrained(release="gemma-scope-9b-pt-res", sae_id=sae_ids[li])
        saes[li] = sae.to(args.device).eval()
        print(f"  Layer {li}: d_sae={sae.cfg.d_sae}")

    capture = SublayerCapture(model, probe_layers)

    # Base model: no chat template, use raw text with few-shot prompt
    FEW_SHOT = (
        "You are a doctor diagnosing patients. Read the patient profile and "
        "pick the most likely diagnosis.\n\n"
    )

    def build_prompt(conversation_turns):
        """Build a raw text prompt from accumulated turns."""
        text = FEW_SHOT
        for turn in conversation_turns:
            if turn["role"] == "user":
                text += turn["content"] + "\n"
            else:
                text += turn["content"] + "\n\n"
        return text

    # Load cases
    print("Loading DDXPlus...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    rng = random.Random(args.seed)
    valid = [i for i in range(len(ds))
             if ds[i]["PATHOLOGY"] in [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:5]]]
    rng.shuffle(valid)
    if args.max_cases: valid = valid[:args.max_cases]

    print(f"\n{'='*70}")
    print(f"DDXPlus Base Model Pipeline — {args.model}")
    print(f"{'='*70}\n")

    conversation = []
    all_turns = []
    fat_res = {li: [] for li in probe_layers}
    fat_attn = {li: [] for li in probe_layers}
    fat_mlp = {li: [] for li in probe_layers}
    clean_res = {li: [] for li in probe_layers}
    all_sae_feats = {li: [] for li in probe_layers}
    clean_sae_feats = {li: [] for li in probe_layers}
    crng = random.Random(args.seed)

    for case_num, idx in enumerate(valid):
        prompt = build_prompt(conversation)
        ctx_now = len(tokenizer.encode(prompt))
        if ctx_now / args.max_ctx > args.fill_target:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        row = ds[idx]
        pathology = row["PATHOLOGY"]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        opts = [d[0] for d in ddx[:5]]
        sh = list(enumerate(opts)); crng.shuffle(sh)
        sn = [n for _, n in sh]
        gp = next(i for i, n in enumerate(sn) if n == pathology)
        gl = OPTION_LABELS[gp]

        case_text = format_case_mcq(
            row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
            row["EVIDENCES"], evidence_db, sn)

        context_fill = ctx_now / args.max_ctx

        # ── Fatigued forward pass ───────────────────────────────────
        conversation.append({"role": "user", "content": case_text})
        full_prompt = build_prompt(conversation)
        fat_ids = tokenizer(full_prompt, return_tensors="pt",
                           truncation=True, max_length=args.max_ctx).input_ids.to(args.device)

        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            _ = model(fat_ids, use_cache=False)
        capture.enabled = False
        del _

        for li in probe_layers:
            fat_res[li].append(capture.res_out[li].clone())
            fat_attn[li].append(capture.attn_out[li].clone())
            fat_mlp[li].append(capture.mlp_out[li].clone())

            # SAE features
            act = capture.res_out[li].to(args.device)
            with torch.no_grad():
                feat = saes[li].encode(act.unsqueeze(0))[0].cpu()
            all_sae_feats[li].append(feat)

        # Generate
        with torch.no_grad():
            gen = model.generate(fat_ids, max_new_tokens=args.max_new, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(gen[0, fat_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_mcq_answer(resp)
        correct = pred == gl if pred else False

        # ── Clean forward pass ──────────────────────────────────────
        clean_prompt = FEW_SHOT + case_text + "\n"
        clean_ids = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(args.device)

        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            _ = model(clean_ids, use_cache=False)
        capture.enabled = False
        del _

        for li in probe_layers:
            clean_res[li].append(capture.res_out[li].clone())
            act = capture.res_out[li].to(args.device)
            with torch.no_grad():
                feat = saes[li].encode(act.unsqueeze(0))[0].cpu()
            clean_sae_feats[li].append(feat)

        all_turns.append({
            "case": case_num, "context_fill": round(context_fill, 4),
            "pathology": pathology, "gold": gl, "pred": pred, "correct": correct,
            "response": resp[:200],
            **{f"sae_{li}_active": int((all_sae_feats[li][-1] > 0).sum().item()) for li in probe_layers},
        })

        conversation.append({"role": "assistant", "content": resp})

        if (case_num + 1) % 5 == 0 or case_num < 3:
            acc = sum(t["correct"] for t in all_turns) / len(all_turns) * 100
            print(f"  C{case_num+1:3d} ctx={context_fill:5.1%} pred={pred} gold={gl} "
                  f"{'✓' if correct else '✗'} acc={acc:.0f}%")

        torch.cuda.empty_cache(); gc.collect()

    capture.remove()
    n = len(all_turns)

    # ── Probes ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training probes ({n} cases)...")
    print(f"{'='*70}")

    y_correct = torch.tensor([t["correct"] for t in all_turns], dtype=torch.float32)
    fills = torch.tensor([t["context_fill"] for t in all_turns])
    y_fill = (fills >= fills.median()).float()

    print(f"\n{'Layer':>6s} {'Res→Corr':>10s} {'Attn→Corr':>10s} {'MLP→Corr':>10s} "
          f"{'Res→Fill':>10s} {'Cos mean':>10s} {'Cos↔fill':>10s}")
    print("-" * 68)

    for li in probe_layers:
        fr = torch.stack(fat_res[li])
        fa = torch.stack(fat_attn[li])
        fm = torch.stack(fat_mlp[li])
        cr = torch.stack(clean_res[li])

        # Normalize
        mu, std = fr.mean(0), fr.std(0).clamp(min=1e-6)
        frn = (fr - mu) / std
        fan = (fa - fa.mean(0)) / fa.std(0).clamp(min=1e-6)
        fmn = (fm - fm.mean(0)) / fm.std(0).clamp(min=1e-6)

        # Probes
        res_corr = train_probe(frn, y_correct) if y_correct.unique().numel() > 1 else 0
        attn_corr = train_probe(fan, y_correct) if y_correct.unique().numel() > 1 else 0
        mlp_corr = train_probe(fmn, y_correct) if y_correct.unique().numel() > 1 else 0
        res_fill = train_probe(frn, y_fill) if y_fill.unique().numel() > 1 else 0

        # Cosine sim
        coss = torch.tensor([torch.nn.functional.cosine_similarity(fr[i], cr[i], dim=0).item()
                            for i in range(n)])
        cos_corr = np.corrcoef(coss.numpy(), fills.numpy())[0, 1] if n > 2 else 0

        print(f"{li:6d} {res_corr:10.3f} {attn_corr:10.3f} {mlp_corr:10.3f} "
              f"{res_fill:10.3f} {coss.mean():10.4f} {cos_corr:10.3f}")

    # ── SAE drift ───────────────────────────────────────────────────
    print(f"\nSAE feature drift (layer 20):")
    if n >= 4:
        fat_sae = torch.stack(all_sae_feats[20])
        cln_sae = torch.stack(clean_sae_feats[20])
        print(f"  Active: clean={int((cln_sae>0).sum(1).float().mean())} "
              f"fatigued={int((fat_sae>0).sum(1).float().mean())}")
        diff = (fat_sae - cln_sae).abs().mean(0)
        topk = diff.topk(10)
        for val, idx in zip(topk.values, topk.indices):
            cm = cln_sae[:, idx].mean().item()
            fm = fat_sae[:, idx].mean().item()
            print(f"  F{idx.item():6d}: Δ={val.item():.4f} clean={cm:.4f} fat={fm:.4f} {'↑' if fm>cm else '↓'}")

    # ── Save ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_turns)
    df.to_csv(out_dir / "turns.csv", index=False)
    torch.save({
        "fat_res": {li: torch.stack(fat_res[li]) for li in probe_layers},
        "clean_res": {li: torch.stack(clean_res[li]) for li in probe_layers},
        "fat_sae": {li: torch.stack(all_sae_feats[li]) for li in probe_layers},
        "clean_sae": {li: torch.stack(clean_sae_feats[li]) for li in probe_layers},
        "turns": all_turns,
    }, out_dir / "all_data.pt")

    acc = sum(t["correct"] for t in all_turns) / n * 100
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n} cases, {acc:.1f}% accuracy")
    print(f"{'='*70}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
