"""v6: Clean → fatigued activation patching experiments.

v6-1: Direction-projected patching — adjusts residual at prediction sites
      along the within_task_fraction probe direction to match the clean-context
      probe readout. Tested at L0 and L10.

v6-2: L0 full-residual patching — replaces the entire L0 residual at each
      prediction site with the residual captured from the clean (single-case)
      forward pass. Sanity-checks whether post-patch tokens become degenerate.
"""

from __future__ import annotations

import ast
import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.probes.task_position.extract_activations import (
    _apply_template_with_generation_prompt,
    _prepare_messages,
    _tokenizer_supports_system_role,
)
from src.probes.ddxplus import (
    OPTION_LABELS,
    SYSTEM_PROMPT,
    format_case_mcq,
    load_evidence_db,
)
from src.probes.task_position.patching import DirectionPatchHook, FullResidualPatchHook
from src.probes.task_position.probes import RidgeProbe

MODEL_ID = "google/gemma-2-9b-it"
DEVICE = "cuda"
ACTIVATIONS_PATH = "results/probes/task_position/gemma-9b-it-v5/activations.pt"
SPLIT_PATH = "results/probes/task_position/gemma-9b-it/probes/split.json"
CORRECTNESS_PATH = "results/probes/task_position/gemma-9b-it/correctness.json"
PROBE_DIR = Path("results/probes/task_position/gemma-9b-it-v5/probes")
EVIDENCE_DB_PATH = "data/context_fatigue/release_evidences.json"
OUT_MD = Path("results/probes/task_position/2026-04-14-v6-patch-experiment.md")
OUT_CSV = Path("results/probes/task_position/2026-04-14-v6-patch-records.csv")


def get_option_token_ids(tokenizer) -> dict[str, int]:
    out = {}
    for letter in OPTION_LABELS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) == 1:
            out[letter] = ids[0]
        else:
            out[letter] = tokenizer.encode(letter, add_special_tokens=False)[0]
    return out


def replay_case_texts(
    ds,
    valid_indices: list[int],
    evidence_db: dict,
    n_traces: int,
    case_counts: dict[int, int],
) -> dict[int, list[dict]]:
    """Replay the extraction RNG to reconstruct per-case case_texts.

    Returns dict mapping trace_id -> list of {"case_text": str, "gold": str}
    truncated to case_counts[trace_id] entries.
    """
    trace_rng = random.Random(42)
    result = {}
    for trace_i in range(n_traces):
        case_rng_seed = trace_rng.randint(0, 2**31 - 1)
        case_rng = random.Random(case_rng_seed)
        indices = list(valid_indices)
        case_rng.shuffle(indices)

        n_cases = case_counts.get(trace_i, 0)
        case_texts = []
        for idx in indices[:n_cases + 50]:  # small buffer in case of boundary mismatch
            if len(case_texts) >= n_cases:
                break
            row = ds[idx]
            pathology = row["PATHOLOGY"]
            ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            option_names = [d[0] for d in ddx[:5]]
            shuffled = list(option_names)
            case_rng.shuffle(shuffled)
            gold_letter = OPTION_LABELS[shuffled.index(pathology)]
            case_text = format_case_mcq(
                row["AGE"],
                row["SEX"],
                row["INITIAL_EVIDENCE"],
                row["EVIDENCES"],
                evidence_db,
                shuffled,
            )
            case_texts.append({"case_text": case_text, "gold": gold_letter})

        result[trace_i] = case_texts
    return result


def build_clean_conversation(tokenizer, case_text: str) -> list[dict]:
    """Build a single-case conversation following the Gemma system-role workaround."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case_text},
    ]


def capture_residual_at_last_token(
    model,
    tokenizer,
    conversation: list[dict],
    layers: list[int],
    device: str,
) -> tuple[dict[int, torch.Tensor], int]:
    """Run a forward pass and return residual vectors at the last token.

    Returns:
        residuals: dict mapping layer_index -> 1D tensor of shape (hidden,)
        seq_len: length of the prompt
    """
    prompt_text = _apply_template_with_generation_prompt(tokenizer, conversation)
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    seq_len = input_ids.shape[1]
    last_pos = seq_len - 1

    residuals: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            residuals[layer_idx] = hs[0, last_pos].detach().cpu()
        return hook_fn

    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    with torch.no_grad():
        out = model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # Capture option logits at last position
    last_logits = out.logits[0, last_pos, :]
    return residuals, seq_len, last_logits, input_ids


def compute_option_probs(last_logits: torch.Tensor, option_token_ids: dict[str, int]) -> dict[str, float]:
    option_ids = [option_token_ids[l] for l in OPTION_LABELS]
    option_logits = last_logits[option_ids].float()
    probs = torch.softmax(option_logits, dim=0).cpu()
    return {l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)}


def probe_readout(resid: torch.Tensor, probe: RidgeProbe) -> float:
    """Compute probe prediction on a single residual vector."""
    x = resid.float().numpy().reshape(1, -1)
    return float(probe.predict(x)[0])


def main():
    print("Loading activations...")
    blob = torch.load(ACTIVATIONS_PATH, weights_only=False)
    traces = blob["traces"]

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    test_ids = {int(x) for x in split["test_ids"]}

    with open(CORRECTNESS_PATH) as f:
        correctness_by_trace = json.load(f)

    print(f"Test traces: {sorted(test_ids)}")

    # Load probes from v5
    probe_L0 = RidgeProbe.load(PROBE_DIR / "within_task_fraction_L0.pkl")
    probe_L10 = RidgeProbe.load(PROBE_DIR / "within_task_fraction_L10.pkl")
    probes = {0: probe_L0, 10: probe_L10}

    # Extract probe directions and biases as tensors for patch computation
    probe_data = {}
    for layer, probe in probes.items():
        w = torch.tensor(probe.direction(), dtype=torch.float32)
        b = float(probe._model.intercept_)
        w_norm_sq = float((w * w).sum())
        probe_data[layer] = {"w": w, "b": b, "w_norm_sq": w_norm_sq}

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    option_token_ids = get_option_token_ids(tokenizer)
    option_id_tensor = torch.tensor(
        [option_token_ids[l] for l in OPTION_LABELS], device=DEVICE
    )

    # Load evidence db and dataset for RNG replay
    print("Loading DDXPlus dataset for RNG replay...")
    evidence_db = load_evidence_db(EVIDENCE_DB_PATH)
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid_indices = [
        i for i in range(len(ds))
        if ds[i]["PATHOLOGY"] in [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:5]]
    ]

    # Build case count per trace from v5 activations
    case_counts = {t["trace_id"]: len(t["case_boundaries"]) for t in traces}

    print("Replaying extraction RNG to reconstruct case texts...")
    case_texts_by_trace = replay_case_texts(
        ds, valid_indices, evidence_db, 20, case_counts
    )

    # Cross-check: decode first token slice of trace 0 and compare to replayed case_text
    t0 = traces[0]
    cr0 = correctness_by_trace.get("0", [])
    if cr0:
        # Prediction site of case 0 in trace 0
        ps0 = cr0[0]["prediction_site"]
        # Tokenize clean case 0
        case_text_0 = case_texts_by_trace[0][0]["case_text"]
        conv0 = build_clean_conversation(tokenizer, case_text_0)
        prompt_text_0 = _apply_template_with_generation_prompt(tokenizer, conv0)
        clean_ids_0 = tokenizer(prompt_text_0, return_tensors="pt", add_special_tokens=False).input_ids[0]
        print(f"\nCross-check: clean prompt len={len(clean_ids_0)}, fatigued prediction_site={ps0}")
        if abs(len(clean_ids_0) - ps0) > 200:
            raise SystemExit(
                f"RNG replay cross-check failed: clean len {len(clean_ids_0)} vs "
                f"fatigued prediction_site {ps0} — off by more than 200 tokens."
            )
        # Decode first 50 tokens of trace 0 and compare to clean case text
        trace0_first_tokens = t0["tokens"][:min(50, ps0)]
        decoded_trace0 = tokenizer.decode(trace0_first_tokens)
        print(f"Trace0 first tokens decoded (first 200 chars): {decoded_trace0[:200]!r}")
        print(f"Replayed case_text_0 (first 200 chars): {case_text_0[:200]!r}")

    # -------------------------------------------------------------------------
    # Phase A: capture clean activations
    # -------------------------------------------------------------------------
    print("\n=== Phase A: Capture clean activations ===")
    clean_data: dict[tuple[int, int], dict] = {}

    test_traces = [t for t in traces if t["trace_id"] in test_ids]
    total_cases = sum(
        len(correctness_by_trace.get(str(t["trace_id"]), []))
        for t in test_traces
    )
    print(f"Total test cases: {total_cases}")

    case_count = 0
    for t in test_traces:
        tid = t["trace_id"]
        cr = correctness_by_trace.get(str(tid), [])
        case_texts = case_texts_by_trace[tid]

        for case_rec in cr:
            ci = case_rec["case_index"]
            if ci >= len(case_texts):
                continue
            case_text = case_texts[ci]["case_text"]
            conv = build_clean_conversation(tokenizer, case_text)

            resids, clean_seq_len, clean_last_logits, clean_input_ids = (
                capture_residual_at_last_token(model, tokenizer, conv, [0, 10], DEVICE)
            )
            clean_option_probs = compute_option_probs(
                clean_last_logits.to("cpu"), option_token_ids
            )
            clean_pred = max(clean_option_probs, key=clean_option_probs.get)
            clean_correct = clean_pred == case_rec["gold"]

            clean_data[(tid, ci)] = {
                "L0_resid": resids[0],
                "L10_resid": resids[10],
                "L0_wf": probe_readout(resids[0], probe_L0),
                "L10_wf": probe_readout(resids[10], probe_L10),
                "clean_option_probs": clean_option_probs,
                "clean_pred": clean_pred,
                "clean_correct": clean_correct,
                "clean_seq_len": clean_seq_len,
            }

            case_count += 1
            if case_count % 10 == 0 or case_count == total_cases:
                print(f"  Phase A: {case_count}/{total_cases} clean passes done")

            del resids, clean_last_logits, clean_input_ids
            torch.cuda.empty_cache()

    clean_acc = sum(1 for v in clean_data.values() if v["clean_correct"]) / max(len(clean_data), 1)
    print(f"\nPhase A clean accuracy: {clean_acc:.3f} ({sum(1 for v in clean_data.values() if v['clean_correct'])}/{len(clean_data)})")
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase B: v6-1 direction-projected patching
    # -------------------------------------------------------------------------
    print("\n=== Phase B: v6-1 direction-projected patching ===")

    # Pre-compute fat_wf per case from stored v5 activations
    fat_wf: dict[tuple[int, int], dict[int, float]] = {}
    for t in test_traces:
        tid = t["trace_id"]
        cr = correctness_by_trace.get(str(tid), [])
        acts_L0 = t["activations"][0]   # (seq, hidden)
        acts_L10 = t["activations"][10]

        for case_rec in cr:
            ci = case_rec["case_index"]
            ps = case_rec["prediction_site"]
            if ps >= acts_L0.shape[0]:
                continue
            resid_L0 = acts_L0[ps].float()
            resid_L10 = acts_L10[ps].float()
            fat_wf[(tid, ci)] = {
                0: probe_readout(resid_L0, probe_L0),
                10: probe_readout(resid_L10, probe_L10),
            }

    records_b: list[dict] = []

    for patch_layer in [0, 10]:
        pd_entry = probe_data[patch_layer]
        w = pd_entry["w"]
        w_norm_sq = pd_entry["w_norm_sq"]
        b = pd_entry["b"]

        direction_hook = DirectionPatchHook(model, patch_layer)
        cond_name = f"patch_wf_L{patch_layer}"

        for t in test_traces:
            tid = t["trace_id"]
            cr = correctness_by_trace.get(str(tid), [])
            if not cr:
                continue
            tokens = t["tokens"]
            input_ids = torch.tensor([tokens], device=DEVICE)

            # Build patch_map: {prediction_site: delta_vector}
            patch_map = {}
            for case_rec in cr:
                ci = case_rec["case_index"]
                ps = case_rec["prediction_site"]
                if (tid, ci) not in clean_data or (tid, ci) not in fat_wf:
                    continue
                clean_wf = clean_data[(tid, ci)][f"L{patch_layer}_wf"]
                current_fat_wf = fat_wf[(tid, ci)][patch_layer]
                # delta = (clean_wf - current_fat_wf) * w / (w @ w)
                delta = ((clean_wf - current_fat_wf) / w_norm_sq) * w
                patch_map[ps] = delta

            direction_hook.set_patches(patch_map)

            with direction_hook.patching(), torch.no_grad():
                out = model(input_ids, use_cache=False)
            logits = out.logits[0]

            for case_rec in cr:
                ci = case_rec["case_index"]
                ps = case_rec["prediction_site"]
                if ps >= logits.shape[0]:
                    continue
                option_logits = logits[ps][option_id_tensor].float()
                probs = torch.softmax(option_logits, dim=0).cpu()
                option_probs = {l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)}
                pred = max(option_probs, key=option_probs.get)
                wf_readout = fat_wf.get((tid, ci), {}).get(10, 0.5)
                records_b.append({
                    "trace_id": tid,
                    "case_index": ci,
                    "gold": case_rec["gold"],
                    "condition": cond_name,
                    "pred": pred,
                    "correct": int(pred == case_rec["gold"]),
                    "confidence": option_probs[pred],
                    "wf_readout": wf_readout,
                })

            del out, logits
            torch.cuda.empty_cache()
            print(f"  Phase B: trace {tid} condition {cond_name} done")

        direction_hook.remove()

    # Also run no_patch baseline
    print("\nRunning no_patch baseline...")
    for t in test_traces:
        tid = t["trace_id"]
        cr = correctness_by_trace.get(str(tid), [])
        if not cr:
            continue
        tokens = t["tokens"]
        input_ids = torch.tensor([tokens], device=DEVICE)

        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        logits = out.logits[0]

        for case_rec in cr:
            ci = case_rec["case_index"]
            ps = case_rec["prediction_site"]
            if ps >= logits.shape[0]:
                continue
            option_logits = logits[ps][option_id_tensor].float()
            probs = torch.softmax(option_logits, dim=0).cpu()
            option_probs = {l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)}
            pred = max(option_probs, key=option_probs.get)
            wf_readout = fat_wf.get((tid, ci), {}).get(10, 0.5)
            records_b.append({
                "trace_id": tid,
                "case_index": ci,
                "gold": case_rec["gold"],
                "condition": "no_patch",
                "pred": pred,
                "correct": int(pred == case_rec["gold"]),
                "confidence": option_probs[pred],
                "wf_readout": wf_readout,
            })

        del out, logits
        torch.cuda.empty_cache()
        print(f"  no_patch trace {tid} done")

    gc.collect()

    # -------------------------------------------------------------------------
    # Phase C: v6-2 L0 full-residual patching
    # -------------------------------------------------------------------------
    print("\n=== Phase C: v6-2 L0 full-residual patching ===")
    records_c: list[dict] = []
    sanity_slices: list[dict] = []
    first_test_trace_done = False

    full_hook = FullResidualPatchHook(model, layer=0)

    for t in test_traces:
        tid = t["trace_id"]
        cr = correctness_by_trace.get(str(tid), [])
        if not cr:
            continue
        tokens = t["tokens"]
        input_ids = torch.tensor([tokens], device=DEVICE)

        # Build patch_map: {prediction_site: clean_L0_resid}
        patch_map = {}
        for case_rec in cr:
            ci = case_rec["case_index"]
            ps = case_rec["prediction_site"]
            if (tid, ci) not in clean_data:
                continue
            patch_map[ps] = clean_data[(tid, ci)]["L0_resid"]

        full_hook.set_patches(patch_map)

        # Also capture logits at prediction_site+1,2,3 for first test trace
        capture_sanity = not first_test_trace_done
        sanity_logits: dict[int, torch.Tensor] = {}

        if capture_sanity:
            sanity_positions = set()
            for case_rec in cr[:3]:  # first 3 cases
                ps = case_rec["prediction_site"]
                for offset in [1, 2, 3]:
                    if ps + offset < len(tokens):
                        sanity_positions.add(ps + offset)

            sanity_hooks = []
            def make_sanity_hook(positions):
                def hook_fn(module, inputs, output):
                    logits_out = output
                    # This is called after the whole model, not a layer hook
                    pass
                return hook_fn

        with full_hook.patching(), torch.no_grad():
            out = model(input_ids, use_cache=False)
        logits = out.logits[0]

        if capture_sanity:
            for case_rec in cr[:3]:
                ps = case_rec["prediction_site"]
                for offset in [1, 2, 3]:
                    pos = ps + offset
                    if pos < logits.shape[0]:
                        top5 = torch.topk(logits[pos].float(), 5)
                        top5_tokens = [tokenizer.decode([tid_]) for tid_ in top5.indices.tolist()]
                        sanity_slices.append({
                            "trace_id": tid,
                            "case_index": case_rec["case_index"],
                            "prediction_site": ps,
                            "offset": offset,
                            "top5_tokens": top5_tokens,
                            "top5_ids": top5.indices.tolist(),
                        })
            first_test_trace_done = True

        for case_rec in cr:
            ci = case_rec["case_index"]
            ps = case_rec["prediction_site"]
            if ps >= logits.shape[0]:
                continue
            option_logits = logits[ps][option_id_tensor].float()
            probs = torch.softmax(option_logits, dim=0).cpu()
            option_probs = {l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)}
            pred = max(option_probs, key=option_probs.get)
            wf_readout = fat_wf.get((tid, ci), {}).get(10, 0.5)
            records_c.append({
                "trace_id": tid,
                "case_index": ci,
                "gold": case_rec["gold"],
                "condition": "patch_L0_full",
                "pred": pred,
                "correct": int(pred == case_rec["gold"]),
                "confidence": option_probs[pred],
                "wf_readout": wf_readout,
            })

        del out, logits
        torch.cuda.empty_cache()
        print(f"  Phase C: trace {tid} done")

    full_hook.remove()

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n=== Analysis ===")
    all_records = records_b + records_c
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved records to {OUT_CSV}")

    conditions_order = ["no_patch", "patch_wf_L0", "patch_wf_L10", "patch_L0_full"]
    summary_rows = []
    for cond in conditions_order:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        acc = sub["correct"].mean()
        mean_conf = sub["confidence"].mean()
        cal_gap = mean_conf - acc
        summary_rows.append({
            "condition": cond,
            "n": len(sub),
            "accuracy": round(acc, 4),
            "mean_confidence": round(float(mean_conf), 4),
            "calibration_gap": round(float(cal_gap), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\nGlobal summary:")
    print(summary_df.to_string(index=False))

    # Flip stats vs no_patch
    pivot = df.pivot_table(
        index=["trace_id", "case_index"],
        columns="condition",
        values="correct",
    ).reset_index()

    flip_rows = []
    for cond in conditions_order:
        if cond == "no_patch" or cond not in pivot.columns or "no_patch" not in pivot.columns:
            continue
        w2r = int(((pivot["no_patch"] == 0) & (pivot[cond] == 1)).sum())
        r2w = int(((pivot["no_patch"] == 1) & (pivot[cond] == 0)).sum())
        flip_rows.append({
            "condition": cond,
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "net_flips": w2r - r2w,
        })
    flips_df = pd.DataFrame(flip_rows)
    print("\nFlip stats vs no_patch:")
    print(flips_df.to_string(index=False))

    # A4-style binned table for each condition
    # Bins by wf_readout (quantile-based, 5 bins)
    df["wf_bin"] = pd.qcut(df["wf_readout"], q=5, labels=False, duplicates="drop")
    a4_rows = []
    for cond in conditions_order:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        for bin_id, bin_df in sub.groupby("wf_bin"):
            acc = bin_df["correct"].mean()
            mean_conf = bin_df["confidence"].mean()
            cal_gap = mean_conf - acc
            a4_rows.append({
                "condition": cond,
                "wf_bin": int(bin_id),
                "n": len(bin_df),
                "accuracy": round(acc, 4),
                "mean_confidence": round(float(mean_conf), 4),
                "calibration_gap": round(float(cal_gap), 4),
            })
    a4_df = pd.DataFrame(a4_rows)

    # -------------------------------------------------------------------------
    # Write markdown
    # -------------------------------------------------------------------------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# v6: Clean→Fatigued Activation Patching Experiments\n\n")
    lines.append(
        "Tests whether substituting clean-context activations into fatigued-context "
        "prediction sites can move accuracy, confidence, or calibration gap. "
        "v6-1 patches only the within_task_fraction direction; v6-2 patches the "
        "full L0 residual at each prediction site.\n\n"
    )

    lines.append("## Phase A: Clean baseline (single-case forward passes)\n\n")
    clean_total = len(clean_data)
    clean_correct = sum(1 for v in clean_data.values() if v["clean_correct"])
    lines.append(
        f"- Total test cases run (clean context, no ICL): **{clean_total}**\n"
        f"- Correct predictions: **{clean_correct}** ({clean_correct/max(clean_total,1):.1%})\n\n"
    )
    lines.append(
        "Note: clean accuracy is expected to be lower than fatigued accuracy (no ICL context). "
        f"Fatigued no_patch accuracy is shown in the global table below.\n\n"
    )

    lines.append("## Global summary table\n\n")
    lines.append(summary_df.to_markdown(index=False) + "\n\n")

    lines.append("## Flip stats vs no_patch\n\n")
    lines.append(flips_df.to_markdown(index=False) + "\n\n")

    lines.append("## A4-style binned table (by within_task_fraction readout, no_patch wf)\n\n")
    for cond in conditions_order:
        sub_a4 = a4_df[a4_df["condition"] == cond]
        if sub_a4.empty:
            continue
        lines.append(f"### {cond}\n\n")
        lines.append(sub_a4.to_markdown(index=False) + "\n\n")

    lines.append("## v6-2 sanity slice (token logits after patched position)\n\n")
    lines.append(
        "Top-5 predicted tokens at positions prediction_site+1, +2, +3 "
        "(first test trace, first 3 cases). "
        "If output is degenerate (e.g., all mass on 'Based'), L0 full patching "
        "corrupts downstream computation.\n\n"
    )
    if sanity_slices:
        sanity_df = pd.DataFrame([
            {
                "case_index": s["case_index"],
                "prediction_site": s["prediction_site"],
                "offset": s["offset"],
                "top5_tokens": str(s["top5_tokens"]),
            }
            for s in sanity_slices
        ])
        lines.append(sanity_df.to_markdown(index=False) + "\n\n")
    else:
        lines.append("No sanity slices captured.\n\n")

    lines.append("## Interpretation\n\n")
    # Compute delta from no_patch for interpretation
    no_patch_row = summary_df[summary_df["condition"] == "no_patch"]
    if not no_patch_row.empty:
        np_acc = no_patch_row.iloc[0]["accuracy"]
        np_cal = no_patch_row.iloc[0]["calibration_gap"]
        interp_lines = []
        for _, row in summary_df.iterrows():
            if row["condition"] == "no_patch":
                continue
            acc_delta = row["accuracy"] - np_acc
            cal_delta = row["calibration_gap"] - np_cal
            interp_lines.append(
                f"- **{row['condition']}**: accuracy Δ={acc_delta:+.4f}, "
                f"calibration_gap Δ={cal_delta:+.4f}"
            )
        lines.append("\n".join(interp_lines) + "\n\n")

    lines.append(
        "If neither v6-1 nor v6-2 moves accuracy or calibration gap beyond noise "
        "(|Δ| < 0.02), the patching result is consistent with the steering and "
        "ablation nulls from v2–v4: the within_task_fraction direction is a "
        "readout of context length, not a causal lever. The model's calibration "
        "gap is driven by mechanisms not localizable to a single layer's residual "
        "stream direction.\n\n"
        "If v6-2 (full L0 patch) also produces near-zero effects, the result "
        "strengthens the conclusion that even the *full* L0 state — which encodes "
        "absolute position through RoPE at the attention level — cannot transplant "
        "the 'freshness' of a clean context into a fatigued one.\n"
    )

    OUT_MD.write_text("".join(lines))
    print(f"\nWrote {OUT_MD}")

    # Print summary to console
    print("\n=== Final Summary ===")
    print(summary_df.to_string(index=False))
    print("\nFlips:")
    print(flips_df.to_string(index=False))
    print(f"\nClean accuracy (Phase A): {clean_correct}/{clean_total} = {clean_correct/max(clean_total,1):.3f}")
    print("\nSanity slices (first 5):")
    for s in sanity_slices[:5]:
        print(f"  case={s['case_index']} ps={s['prediction_site']} +{s['offset']}: {s['top5_tokens']}")


if __name__ == "__main__":
    main()
