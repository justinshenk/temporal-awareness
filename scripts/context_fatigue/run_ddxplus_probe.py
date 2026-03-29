"""
DDXPlus Fatigue Probe — Context Fatigue

Collects residual stream activations from clean and fatigued runs,
then trains linear probes to predict:
  1. Correct vs incorrect (from fatigued activations)
  2. Context fill level (from fatigued activations)
  3. Clean vs fatigued (from paired activations)

If a simple linear probe can predict these, the fatigue signal is
linearly encoded in the residual stream.
"""

import json, re, ast, argparse, random, gc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.optim as optim


SYSTEM_PROMPT = "You are a doctor."
OPTION_LABELS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--fill-target", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--probe-layers", default="0,7,14,21,27",
                   help="Layers to probe")
    p.add_argument("--out-dir", default="results/ddxplus_probe")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ── evidence helpers ────────────────────────────────────────────────────

def load_evidence_db(path):
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        vm = {}
        for vk, vv in info.get("value_meaning", {}).items():
            vm[vk] = vv.get("en", str(vv)) if isinstance(vv, dict) else str(vv)
        db[code] = {
            "question": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "data_type": info.get("data_type", "B"),
            "value_meanings": vm,
        }
    return db


def decode_evidence(ev_str, evidence_db):
    evs = ast.literal_eval(ev_str)
    symptoms, antecedents = [], []
    grouped = {}
    for ev in evs:
        if "@" in ev:
            base, val = ev.split("@", 1)
            grouped.setdefault(base.strip().rstrip("_"), []).append(val.strip())
        else:
            grouped[ev] = []
    for code, values in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        stmt = info["question"].replace("Do you have ", "Has ").replace("Are you ", "Is ").rstrip("?.")
        if info["data_type"] == "B":
            text = f"Yes — {stmt}"
        elif info["data_type"] == "M" and values:
            dec = [info["value_meanings"].get(v, v) for v in values if info["value_meanings"].get(v, v) != "NA"]
            text = f"{stmt}: {', '.join(dec)}" if dec else f"Yes — {stmt}"
        elif info["data_type"] == "C" and values:
            text = f"{stmt}: {', '.join(values)}"
        else:
            text = f"Yes — {stmt}"
        (antecedents if info["is_antecedent"] else symptoms).append(text)
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
    if text and text[0] in "ABCDE":
        return text[0]
    m = re.search(r'\b([ABCDE])\b', text)
    return m.group(1) if m else None


# ── activation capture ──────────────────────────────────────────────────

class ResidualCapture:
    """Capture last-token residual stream at target layers."""

    def __init__(self, model, target_layers):
        self.captured = {}
        self.hooks = []
        self.enabled = False

        for li in target_layers:
            hook = model.model.layers[li].register_forward_hook(self._make_hook(li))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if not self.enabled:
                return
            hs = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = hs[0, -1, :].detach().float().cpu()
        return hook_fn

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_layers = [int(x) for x in args.probe_layers.split(",")]

    evidence_db = load_evidence_db("release_evidences.json")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    capturer = ResidualCapture(model, probe_layers)

    def count_tokens(conv):
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))

    # Load cases
    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    rng = random.Random(args.seed)
    valid_indices = []
    for i in range(len(ds)):
        ddx = ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])
        if ds[i]["PATHOLOGY"] in [d[0] for d in ddx[:5]]:
            valid_indices.append(i)
    rng.shuffle(valid_indices)
    if args.max_cases:
        valid_indices = valid_indices[:args.max_cases]

    print(f"Probe layers: {probe_layers}")
    print(f"\n{'='*70}")
    print("Phase 1: Collecting activations...")
    print(f"{'='*70}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    case_rng = random.Random(args.seed)

    # Storage: per-layer lists of activations and labels
    fatigued_acts = {li: [] for li in probe_layers}
    clean_acts = {li: [] for li in probe_layers}
    labels = []  # metadata per case

    for case_num, idx in enumerate(valid_indices):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > args.fill_target:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        row = ds[idx]
        pathology = row["PATHOLOGY"]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        option_names = [d[0] for d in ddx[:5]]
        shuffled = list(enumerate(option_names))
        case_rng.shuffle(shuffled)
        shuffled_names = [n for _, n in shuffled]
        gold_pos = next(i for i, n in enumerate(shuffled_names) if n == pathology)
        gold_letter = OPTION_LABELS[gold_pos]

        case_text = format_case_mcq(
            row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
            row["EVIDENCES"], evidence_db, shuffled_names)

        context_fill = ctx_now / args.max_ctx

        # ── Fatigued forward pass ───────────────────────────────────
        conversation.append({"role": "user", "content": case_text})
        fat_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        fat_ids = tokenizer(fat_text, return_tensors="pt").input_ids.to(args.device)

        capturer.clear()
        capturer.enabled = True
        with torch.no_grad():
            _ = model(fat_ids, use_cache=False)
        capturer.enabled = False
        del _

        for li in probe_layers:
            fatigued_acts[li].append(capturer.captured[li].clone())

        # Generate fatigued answer
        with torch.no_grad():
            gen = model.generate(fat_ids, max_new_tokens=args.max_new,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        fat_resp = tokenizer.decode(gen[0, fat_ids.shape[1]:], skip_special_tokens=True).strip()
        fat_pred = extract_mcq_answer(fat_resp)
        fat_correct = fat_pred == gold_letter if fat_pred else False

        # ── Clean forward pass ──────────────────────────────────────
        clean_conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_text},
        ]
        clean_text = tokenizer.apply_chat_template(
            clean_conv, tokenize=False, add_generation_prompt=True)
        clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(args.device)

        capturer.clear()
        capturer.enabled = True
        with torch.no_grad():
            _ = model(clean_ids, use_cache=False)
        capturer.enabled = False
        del _

        for li in probe_layers:
            clean_acts[li].append(capturer.captured[li].clone())

        # Generate clean answer
        with torch.no_grad():
            gen = model.generate(clean_ids, max_new_tokens=args.max_new,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        clean_resp = tokenizer.decode(gen[0, clean_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_pred = extract_mcq_answer(clean_resp)
        clean_correct = clean_pred == gold_letter if clean_pred else False

        labels.append({
            "case": case_num,
            "context_fill": context_fill,
            "fat_correct": fat_correct,
            "clean_correct": clean_correct,
            "fat_pred": fat_pred,
            "clean_pred": clean_pred,
            "gold": gold_letter,
            "pathology": pathology,
        })

        # Advance conversation
        conversation.append({"role": "assistant", "content": fat_resp})

        if (case_num + 1) % 10 == 0 or case_num < 3:
            print(f"  C{case_num+1:3d} ctx={context_fill:5.1%} | "
                  f"fat={fat_pred}{'✓' if fat_correct else '✗'} "
                  f"clean={clean_pred}{'✓' if clean_correct else '✗'}")

        torch.cuda.empty_cache()
        gc.collect()

    capturer.remove()
    n = len(labels)
    print(f"\nCollected {n} cases")

    # ── Phase 2: Train probes ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("Phase 2: Training probes...")
    print(f"{'='*70}\n")

    label_df = pd.DataFrame(labels)
    probe_results = []

    def train_linear_probe(X_tensor, y_tensor, n_epochs=500, lr=1e-3, n_folds=5):
        """Train a linear probe with k-fold CV using torch."""
        n_samples = X_tensor.shape[0]
        hidden_dim = X_tensor.shape[1]
        fold_size = n_samples // n_folds
        accs, aucs = [], []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

            if len(train_idx) < 2 or len(val_idx) < 1:
                continue

            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            if y_train.unique().numel() < 2:
                continue

            probe = nn.Linear(hidden_dim, 1)
            optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()

            probe.train()
            for _ in range(n_epochs):
                optimizer.zero_grad()
                logits = probe(X_train).squeeze(-1)
                loss = loss_fn(logits, y_train)
                loss.backward()
                optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val).squeeze(-1)
                val_preds = (val_logits > 0).float()
                acc = (val_preds == y_val).float().mean().item()
                accs.append(acc)

                # AUC (manual: sort by predicted prob, compute trapezoid)
                if y_val.unique().numel() >= 2:
                    probs = torch.sigmoid(val_logits)
                    sorted_idx = probs.argsort(descending=True)
                    sorted_y = y_val[sorted_idx]
                    tp = sorted_y.cumsum(0)
                    fp = (1 - sorted_y).cumsum(0)
                    tpr = tp / sorted_y.sum()
                    fpr = fp / (1 - sorted_y).sum()
                    # Trapezoid rule
                    auc = torch.trapezoid(tpr, fpr).abs().item()
                    aucs.append(auc)

        mean_acc = np.mean(accs) if accs else 0.0
        std_acc = np.std(accs) if accs else 0.0
        mean_auc = np.mean(aucs) if aucs else 0.0
        return mean_acc, std_acc, mean_auc

    for li in probe_layers:
        fat_X = torch.stack(fatigued_acts[li])  # (n, hidden_dim)
        clean_X = torch.stack(clean_acts[li])

        # Normalize for stable training
        fat_mean = fat_X.mean(0)
        fat_std = fat_X.std(0).clamp(min=1e-6)
        fat_Xn = (fat_X - fat_mean) / fat_std
        clean_Xn = (clean_X - fat_mean) / fat_std  # normalize with fatigued stats

        # ── Probe 1: Predict correct/incorrect from fatigued activations ──
        y_correct = torch.tensor(label_df["fat_correct"].astype(float).values)
        if y_correct.unique().numel() > 1:
            # Shuffle for CV
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            probe1_acc, probe1_std, probe1_auc = train_linear_probe(
                fat_Xn[perm], y_correct[perm])
        else:
            probe1_acc = probe1_auc = probe1_std = 0.0

        # ── Probe 2: Predict high vs low context fill ──────────────────
        median_fill = label_df["context_fill"].median()
        y_fill = torch.tensor((label_df["context_fill"] >= median_fill).astype(float).values)
        if y_fill.unique().numel() > 1:
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            probe2_acc, _, probe2_auc = train_linear_probe(
                fat_Xn[perm], y_fill[perm])
        else:
            probe2_acc = probe2_auc = 0.0

        # ── Probe 3: Classify clean vs fatigued (paired) ──────────────
        combined_X = torch.cat([fat_Xn, clean_Xn], dim=0)
        y_source = torch.cat([torch.ones(n), torch.zeros(n)])
        perm = torch.randperm(2 * n, generator=torch.Generator().manual_seed(42))
        probe3_acc, _, probe3_auc = train_linear_probe(
            combined_X[perm], y_source[perm])

        # ── Cosine similarity: clean vs fatigued per case ─────────────
        cosines = []
        for i in range(n):
            f = fat_X[i]
            c = clean_X[i]
            cos = np.dot(f, c) / (np.linalg.norm(f) * np.linalg.norm(c) + 1e-10)
            cosines.append(cos)
        cosines = np.array(cosines)

        # Correlation between cosine similarity and context fill
        fill_vals = label_df["context_fill"].values
        corr = np.corrcoef(cosines, fill_vals)[0, 1]

        # Cosine for correct vs incorrect
        correct_mask = label_df["fat_correct"].values.astype(bool)
        cos_correct = cosines[correct_mask].mean() if correct_mask.sum() > 0 else 0
        cos_incorrect = cosines[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

        result = {
            "layer": li,
            "probe1_correct_acc": probe1_acc,
            "probe1_correct_std": probe1_std,
            "probe1_correct_auc": probe1_auc,
            "probe2_fill_acc": probe2_acc,
            "probe2_fill_auc": probe2_auc,
            "probe3_source_acc": probe3_acc,
            "probe3_source_auc": probe3_auc,
            "mean_cosine_sim": cosines.mean(),
            "cosine_fill_corr": corr,
            "cosine_correct": cos_correct,
            "cosine_incorrect": cos_incorrect,
        }
        probe_results.append(result)

        print(f"Layer {li:2d}:")
        print(f"  Probe 1 (correct/incorrect): acc={probe1_acc:.3f}±{probe1_std:.3f} auc={probe1_auc:.3f}")
        print(f"  Probe 2 (high/low fill):     acc={probe2_acc:.3f} auc={probe2_auc:.3f}")
        print(f"  Probe 3 (clean/fatigued):    acc={probe3_acc:.3f} auc={probe3_auc:.3f}")
        print(f"  Cosine sim (clean↔fatigued): mean={cosines.mean():.4f} corr_w_fill={corr:.3f}")
        print(f"    correct cases: {cos_correct:.4f} | incorrect: {cos_incorrect:.4f}")
        print()

    # ── Save ────────────────────────────────────────────────────────
    probe_df = pd.DataFrame(probe_results)
    probe_df.to_csv(out_dir / "probe_results.csv", index=False)
    label_df.to_csv(out_dir / "case_labels.csv", index=False)

    # Save activations for later use
    torch.save({
        "fatigued": {li: torch.stack(fatigued_acts[li]) for li in probe_layers},
        "clean": {li: torch.stack(clean_acts[li]) for li in probe_layers},
        "labels": label_df.to_dict("records"),
    }, out_dir / "activations.pt")

    print(f"\n{'='*70}")
    print("PROBE SUMMARY")
    print(f"{'='*70}")
    print(f"Cases: {n} | Fatigued acc: {label_df['fat_correct'].mean()*100:.1f}% | Clean acc: {label_df['clean_correct'].mean()*100:.1f}%")
    print(f"\n{'Layer':>6s} {'Correct↕':>10s} {'Fill↕':>10s} {'Source↕':>10s} {'Cos μ':>8s} {'Cos↔fill':>10s}")
    print("-" * 58)
    for r in probe_results:
        print(f"{r['layer']:6d} {r['probe1_correct_auc']:10.3f} {r['probe2_fill_auc']:10.3f} "
              f"{r['probe3_source_auc']:10.3f} {r['mean_cosine_sim']:8.4f} {r['cosine_fill_corr']:10.3f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
