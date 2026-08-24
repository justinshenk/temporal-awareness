"""E7 — counterfactual format-instruction patching (Stage 1: maximal patch).

§4.5's chain "read, represented, overruled" rests on decoding and behavior; nothing yet shows
the represented instruction is *causally recoverable* from context states. Here hidden states
are transplanted between counterfactual instruction conditions: System A (the clinical
template) vs System B (a JSON answer shape), drafted to identical token counts, over the SAME
accumulated filler tokens — so the system span is the only token difference and every position
aligns. The patch replaces the intervening-context states (system span excluded) of the
recipient with the donor's, and the readout asks which format the model is about to produce.

**Closure semantics.** With identical context tokens, closing the system span during the
*donor capture* would make donor and recipient context states bit-identical (the closure
removes the only route from the system tokens into the context), nulling the patch by
construction. So: donor states are always captured with attention OPEN; the closure
(``SpanAttentionClamp`` scale=0 on the system span) applies to the *scored* forwards — pure
and patched alike in the primary arms, so the direct system route is closed everywhere the
behavior is measured and any A-signal in a patched-B run must arrive through the transplanted
states. ``--no-close`` runs the secondary open arms.

**Metric.** Teacher-forced prefix log-probs of canonical prefixes (S_A, S_B) plus the
bare-letter mass S_P = log Σ_letters P(first token is a letter, bare or space-prefixed).
Prefix defaults are validated empirically in preflight against real generations of each pure
condition. Estimand: ΔΔ = (S_A − S_B)_patched − (S_A − S_B)_unpatched per item.

**Baselines.** Under closure, a pure run's context states form closed while donor states are
captured open, so "received open-captured states at all" would confound donor identity (the
unrelated-fact control exposed this in preflight: it moved as much as the counterfactual
patch). Every patched condition is therefore compared to a SELF-patch baseline — the
recipient's own open-captured states patched in through the identical procedure — and the
estimand is ΔΔ = (S_A − S_B)_patch_donor − (S_A − S_B)_patch_self.

**Controls.** A→A self-patch asserted bit-identical in preflight (no closure, so it is a true
no-op there); unrelated-fact donor pair (same instruction, one irrelevant clause differing,
token-matched) run per item against its own self-patch; delivery states (final-position
hidden states, every layer) saved for the Probe-1 re-read on patched runs.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_format_patch.py --preflight
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_format_patch.py \
        --filler mmlu --depth 42
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import (
    MEDICAL_SUBJECTS,
    OPTION_LABELS,
    extract_mcq_answer,
    generate_with_entropy,
    load_code_filler_pool,
    load_filler_pool,
    render_prompt,
)

from src.probes.context_fatigue.activation_patch import (
    SpanActivationPatch,
    capture_layer_states,
)
from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_token_span,
    locate_turn_spans,
)
from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    load_evidence_db,
    load_probe_pool,
)
from src.probes.context_fatigue.instruction_checks import (
    CLINICAL_FORMAT_SYSTEM,
    check_clinical_format,
)

FORMAT_A = CLINICAL_FORMAT_SYSTEM

# Trimmed to FORMAT_A's exact token count at runtime (the matched-twin method); the tail
# clause exists to absorb the trim, so a mid-clause cut still leaves a complete instruction.
FORMAT_B_SEED = (
    "You are a doctor. For each patient, reply with one JSON object in exactly this shape:\n"
    '{"answer": "<letter>"}\n'
    "Output only that JSON object on a single line, with no other text before it and no "
    "other text after it at all."
)

# Unrelated-fact donor pair: the SAME instruction, differing only in an irrelevant clause.
# The clause words are chosen and asserted token-count-equal at runtime.
UNRELATED_X = FORMAT_A + "\nThe clinic is located in Geneva."
UNRELATED_Y = FORMAT_A + "\nThe clinic is located in Toronto."

PREFIX_A_DEFAULT = "ANSWER:"
PREFIX_B_DEFAULT = '{"answer":'

DEPTH_DEFAULT = {"mmlu": 42, "code": 15}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--filler", choices=["mmlu", "code"], default="mmlu")
    p.add_argument("--depth", type=int, default=None,
                   help="filler turns before the probe (default: mmlu 42, code 15 — the "
                        "committed E6 deep ends; code is the fill-matched deep arm)")
    p.add_argument("--filler-max-new", type=int, default=200)
    p.add_argument("--n-probes", type=int, default=40)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix-a", default=PREFIX_A_DEFAULT)
    p.add_argument("--prefix-b", default=PREFIX_B_DEFAULT)
    p.add_argument("--patch-layers", type=int, nargs="+", default=None,
                   help="decoder layers to patch (default: all — the Stage-1 maximal patch)")
    p.add_argument("--patch-positions", default="all",
                   choices=["all", "assistant_turns", "user_turns", "last_1", "last_2",
                            "last_4"],
                   help="Stage-2 position subsets: the context's assistant turns, its user "
                        "turns, or its last k turns (any role); 'all' is the Stage-1 patch")
    p.add_argument("--random-control", action="store_true",
                   help="replace the position subset with a size-matched uniform sample of "
                        "intervening positions (Stage-2 control; meaningless with 'all')")
    p.add_argument("--filler-letter-only", action="store_true",
                   help="store each mmlu filler reply as its extracted bare letter. On OLMo "
                        "the model does this on its own, making the accumulated context a "
                        "counter-template; Qwen complies with the clinical format even on "
                        "filler (replies 'B\\nSUPPORTING: ...'), which would make the "
                        "'precedent' arm demonstrate the instructed format instead. This flag "
                        "pins the demonstrated content to the bare letter so the cell means "
                        "the same thing across families.")
    p.add_argument("--no-close", action="store_true",
                   help="secondary arms: score with the system span's attention open")
    p.add_argument("--generate-n", type=int, default=0,
                   help="also greedy-generate and grade replies for the first N probes")
    p.add_argument("--gen-max-new", type=int, default=96)
    p.add_argument("--out-dir", default="results/context_fatigue/e7_format_patch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def _rendered_length(tokenizer, system: str, is_chat: bool) -> int:
    """Token count of a minimal rendered transcript carrying ``system``.

    The alignment that matters is of the *rendered* transcript: two system prompts equal in
    isolation can render to different lengths because their final word merges differently
    with the chat template's following tokens. The context after the system turn is
    byte-identical across counterfactuals, so matching on this minimal render matches every
    full transcript too.
    """
    conv = [{"role": "system", "content": system}, {"role": "user", "content": "x"}]
    return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))


def trim_to_rendered_match(tokenizer, seed_text: str, reference: str, is_chat: bool) -> str:
    """The matched-twin trim, against the reference prompt's rendered token count."""
    target = _rendered_length(tokenizer, reference, is_chat)
    ids = tokenizer.encode(seed_text)
    for cut in range(len(ids), 10, -1):
        text = tokenizer.decode(ids[:cut])
        if _rendered_length(tokenizer, text, is_chat) == target:
            return text
    raise ValueError("no trim of the seed renders to the reference length")


def letter_token_ids(tokenizer):
    """First-token ids for each option letter, bare and space-prefixed."""
    ids = set()
    for letter in OPTION_LABELS:
        for form in (letter, f" {letter}"):
            enc = tokenizer.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def verify_alignment(tokenizer, text_x, text_y, sys_x, sys_y):
    """§5: donor/recipient position alignment, asserted loudly before any patching.

    Equal length, and identical token ids everywhere outside the union of the two rendered
    system spans. Returns the patchable region ``(sys_end, len)`` boundary start.
    """
    ids_x = tokenizer.encode(text_x)
    ids_y = tokenizer.encode(text_y)
    if len(ids_x) != len(ids_y):
        raise RuntimeError(f"transcripts misaligned: {len(ids_x)} vs {len(ids_y)} tokens")
    span_x = locate_token_span(tokenizer, text_x, sys_x)
    span_y = locate_token_span(tokenizer, text_y, sys_y)
    end = max(span_x[1], span_y[1])
    if ids_x[end:] != ids_y[end:]:
        first = next(i for i, (a, b) in enumerate(zip(ids_x[end:], ids_y[end:])) if a != b)
        raise RuntimeError(f"token ids differ outside the system span at position {end + first}")
    return end


def accumulate_filler(model, tokenizer, args, depth, is_chat):
    """The committed E6 accumulation, reproduced on its exact rng stream, under System A."""
    if args.filler == "mmlu":
        filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens, MEDICAL_SUBJECTS)
    else:
        filler_pool = load_code_filler_pool((depth + 2) * 4, args.seed)
    probe_pool = load_probe_pool(load_evidence_db(), args.n_options, args.seed)
    rng = random.Random(args.seed)
    filler = rng.sample(filler_pool, min(depth + 2, len(filler_pool)))
    probes = rng.sample(probe_pool, args.n_probes)

    conv = [{"role": "system", "content": FORMAT_A}]
    for item in filler[:depth]:
        conv = conv + [{"role": "user", "content": item["text"]}]
        budget = 8 if args.filler == "mmlu" else args.filler_max_new
        resp, _, _, _ = generate_with_entropy(
            model, tokenizer, render_prompt(tokenizer, conv, is_chat),
            args.device, budget, args.max_ctx)
        answer = (resp or "A").strip()
        if args.filler == "mmlu":
            answer = ((extract_mcq_answer(answer) or "A") if args.filler_letter_only
                      else answer[:8])
        conv = conv + [{"role": "assistant", "content": answer}]
    return conv[1:], probes  # context turns without the system turn


def position_spans(tokenizer, text, context, mode, region):
    """Token spans for a Stage-2 position subset, clipped to the patchable region.

    ``context`` is the shared filler turn list; subsets follow the brief's bisection:
    the context's assistant turns, its user turns, or its last k turns of either role.
    """
    if mode == "all":
        return [region]
    turn_spans = locate_turn_spans(tokenizer, text, [t["content"] for t in context])
    roles = [t["role"] for t in context]
    if mode == "assistant_turns":
        chosen = [s for s, r in zip(turn_spans, roles) if r == "assistant"]
    elif mode == "user_turns":
        chosen = [s for s, r in zip(turn_spans, roles) if r == "user"]
    else:
        chosen = turn_spans[-int(mode.rsplit("_", 1)[1]):]
    clipped = [(max(a, region[0]), min(b, region[1])) for a, b in chosen]
    spans = [(a, b) for a, b in clipped if b > a]
    if not spans:
        raise ValueError(f"position subset {mode!r} is empty inside region {region}")
    return spans


def random_spans(region, n_tokens, rng):
    """A size-matched uniform sample of positions in ``region``, merged into spans."""
    positions = sorted(rng.sample(range(region[0], region[1]), n_tokens))
    spans = []
    for p in positions:
        if spans and p == spans[-1][1]:
            spans[-1] = (spans[-1][0], p + 1)
        else:
            spans.append((p, p + 1))
    return spans


def prefix_logprob(model, tokenizer, ids, prefix_ids, device):
    """Teacher-forced sum log-prob of ``prefix_ids`` appended after ``ids``."""
    full = torch.cat([ids, torch.tensor([prefix_ids], device=device)], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    logprobs = torch.log_softmax(logits[0, ids.shape[1] - 1:-1].float(), dim=-1)
    return float(sum(logprobs[i, t] for i, t in enumerate(prefix_ids)))


def letter_mass(model, ids, letter_ids):
    """log Σ_letters P(first generated token is a letter), and the final hidden states."""
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    states = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    return float(torch.log(probs[letter_ids].sum())), states


class Scorer:
    """All three readouts for one rendered transcript under optional closure and patch.

    The donor capture (when ``donor_ids`` is given) happens per scored forward on the same
    appended prefix, so patch lengths always align; donor context states do not depend on the
    appended tokens (causality), only their positions must.
    """

    def __init__(self, model, tokenizer, device, prefix_ids_a, prefix_ids_b, letters,
                 patch_layers=None, close=True):
        self.m, self.tok, self.dev = model, tokenizer, device
        self.pa, self.pb, self.letters = prefix_ids_a, prefix_ids_b, letters
        self.patch_layers = patch_layers
        self.close = close

    def _hooks(self, sys_span, donor_states, patch_span):
        ctx = []
        if self.close:
            ctx.append(SpanAttentionClamp(self.m, span=sys_span, scale=0.0))
        if donor_states is not None:
            ctx.append(SpanActivationPatch(self.m, donor_states, span=patch_span,
                                           layers=self.patch_layers))
        return ctx

    def score(self, ids, sys_span, donor_ids=None, patch_span=None):
        out = {}
        for name, prefix in (("s_a", self.pa), ("s_b", self.pb), ("s_p", None)):
            donor_states = None
            if donor_ids is not None:
                ext = (torch.cat([donor_ids, torch.tensor([prefix], device=self.dev)], dim=1)
                       if prefix else donor_ids)
                donor_states = capture_layer_states(self.m, ext, layers=self.patch_layers)
            hooks = self._hooks(sys_span, donor_states, patch_span)
            try:
                if prefix:
                    out[name] = prefix_logprob(self.m, self.tok, ids, prefix, self.dev)
                else:
                    out[name], out["states"] = letter_mass(self.m, ids, self.letters)
            finally:
                for h in reversed(hooks):
                    h.remove()
            del donor_states
        return out


def main():
    args = parse_args()
    depth = args.depth if args.depth is not None else DEPTH_DEFAULT[args.filler]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()  # closure arms need an explicit mask
    is_chat = tokenizer.chat_template is not None

    format_b = trim_to_rendered_match(tokenizer, FORMAT_B_SEED, FORMAT_A, is_chat)
    if (_rendered_length(tokenizer, UNRELATED_X, is_chat)
            != _rendered_length(tokenizer, UNRELATED_Y, is_chat)):
        raise RuntimeError("unrelated-fact pair does not render token-count matched")
    print(f"System B (trimmed to {len(tokenizer.encode(format_b))} tokens):\n{format_b}\n",
          flush=True)

    prefix_ids_a = tokenizer.encode(args.prefix_a, add_special_tokens=False)
    prefix_ids_b = tokenizer.encode(args.prefix_b, add_special_tokens=False)
    letters = letter_token_ids(tokenizer)
    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.gen_max_new,
                          headroom=args.headroom)

    print(f"Accumulating {depth} {args.filler} filler turns under System A ...", flush=True)
    context, probes = accumulate_filler(model, tokenizer, args, depth, is_chat)
    n_probes = 1 if args.preflight else args.n_probes
    probes = probes[:n_probes]

    systems = {"a": FORMAT_A, "b": format_b, "x": UNRELATED_X, "y": UNRELATED_Y}
    scorer = Scorer(model, tokenizer, device, prefix_ids_a, prefix_ids_b, letters,
                    patch_layers=args.patch_layers, close=not args.no_close)

    records, state_rows, state_stack, skipped = [], [], [], 0
    turns_path = out_dir / "turns.csv"
    for pi, probe in enumerate(probes):
        question = format_case_question(probe["options"], args.n_options, answer_cue=False)
        user = {"role": "user", "content": probe["vignette"] + question}
        texts, ids, spans = {}, {}, {}
        for key, system in systems.items():
            turns = [{"role": "system", "content": system}] + context + [user]
            texts[key] = render_prompt(tokenizer, turns, is_chat)
            ids[key] = tokenizer(texts[key], return_tensors="pt").input_ids.to(device)
            spans[key] = locate_token_span(tokenizer, texts[key], system)
        if not guard.fits(texts["a"], used=0, index=pi):
            skipped += 1
            continue

        sys_end = verify_alignment(tokenizer, texts["a"], texts["b"], FORMAT_A, format_b)
        verify_alignment(tokenizer, texts["x"], texts["y"], UNRELATED_X, UNRELATED_Y)
        probe_start = locate_token_span(tokenizer, texts["a"], user["content"])[0]
        patch_span = (sys_end, probe_start)
        unrel_end = max(locate_token_span(tokenizer, texts["x"], UNRELATED_X)[1],
                        locate_token_span(tokenizer, texts["y"], UNRELATED_Y)[1])
        unrel_span = (unrel_end,
                      locate_token_span(tokenizer, texts["x"], user["content"])[0])

        spans_ab = position_spans(tokenizer, texts["a"], context,
                                  args.patch_positions, patch_span)
        spans_xy = position_spans(tokenizer, texts["x"], context,
                                  args.patch_positions, unrel_span)
        if args.random_control:
            rng_pos = random.Random(args.seed + 7919 * pi)
            spans_ab = random_spans(patch_span,
                                    sum(b - a for a, b in spans_ab), rng_pos)
            spans_xy = random_spans(unrel_span,
                                    sum(b - a for a, b in spans_xy), rng_pos)

        if args.preflight:  # §6: the A→A no-op is exact, asserted before anything is scored
            with torch.no_grad():
                base = model(ids["a"]).logits
            donor = capture_layer_states(model, ids["a"])
            with SpanActivationPatch(model, donor, span=patch_span):
                with torch.no_grad():
                    patched = model(ids["a"]).logits
            delta = float((patched - base).abs().max())
            assert delta == 0.0, f"A->A patch is not a no-op: max|dlogits| = {delta}"
            print(f"  A->A no-op asserted: max|dlogits| = {delta}", flush=True)
            del donor, base, patched

        conditions = [
            ("pure_a", "a", None, None),
            ("pure_b", "b", None, None),
            ("patch_ab", "b", "a", spans_ab),
            ("patch_bb", "b", "b", spans_ab),   # self-patch baseline for patch_ab
            ("patch_ba", "a", "b", spans_ab),
            ("patch_aa", "a", "a", spans_ab),   # self-patch baseline for patch_ba
            ("unrel_pure_y", "y", None, None),
            ("unrel_patch_xy", "y", "x", spans_xy),
            ("unrel_patch_yy", "y", "y", spans_xy),
        ]
        for name, rkey, dkey, pspan in conditions:
            rid, sys_span = ids[rkey], spans[rkey]
            donor_ids = ids[dkey] if dkey else None
            got = scorer.score(rid, sys_span, donor_ids=donor_ids, patch_span=pspan)
            states = got.pop("states")
            if name in ("pure_a", "pure_b", "patch_ab", "patch_bb"):
                state_rows.append({"probe": pi, "condition": name})
                state_stack.append(states)
            records.append({
                "probe": pi, "condition": name, "filler": args.filler, "depth": depth,
                "closed": not args.no_close, "ctx_tokens": int(rid.shape[1]),
                "fill": round(int(rid.shape[1]) / args.max_ctx, 4),
                "patch_tokens": sum(b - a for a, b in pspan) if pspan else 0,
                "gold": probe["gold"], **got,
            })

        if pi < args.generate_n or args.preflight:
            for name, rkey, dkey, pspan in conditions:
                sys_span = spans[rkey]
                donor_states = (capture_layer_states(model, ids[dkey],
                                                     layers=args.patch_layers)
                                if dkey else None)
                hooks = scorer._hooks(sys_span, donor_states, pspan)
                try:
                    resp, _, _, _ = generate_with_entropy(
                        model, tokenizer, texts[rkey], device, args.gen_max_new,
                        args.max_ctx)
                finally:
                    for h in reversed(hooks):
                        h.remove()
                graded = check_clinical_format(resp or "", probe["vignette"],
                                               options=probe["options"][:args.n_options])
                is_json = (resp or "").strip().startswith('{"answer"')
                records.append({
                    "probe": pi, "condition": f"{name}_gen", "filler": args.filler,
                    "depth": depth, "closed": not args.no_close,
                    "gold": probe["gold"], "pred": graded["answer"],
                    "compliant_a": graded["fully_compliant"], "compliant_b": is_json,
                    "response": (resp or "")[:400],
                })
                if args.preflight:
                    print(f"  [{name}] reply: {(resp or '')[:160]!r}", flush=True)
                del donor_states

        pd.DataFrame(records).to_csv(turns_path, index=False)
        if (pi + 1) % 5 == 0 or args.preflight:
            print(f"  {pi + 1}/{len(probes)} probes scored", flush=True)
        torch.cuda.empty_cache()

    np.savez_compressed(out_dir / "delivery_states.npz",
                        states=np.stack(state_stack) if state_stack else np.empty(0),
                        rows=json.dumps(state_rows))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    scored = df[~df["condition"].str.endswith("_gen")]
    wide = scored.pivot(index="probe", columns="condition", values=["s_a", "s_b"])
    contrast = {c: (wide[("s_a", c)] - wide[("s_b", c)]) for c in
                ("pure_a", "pure_b", "patch_ab", "patch_bb", "patch_ba", "patch_aa",
                 "unrel_pure_y", "unrel_patch_xy", "unrel_patch_yy")}
    dd_ab = float((contrast["patch_ab"] - contrast["patch_bb"]).mean())
    dd_ba = float((contrast["patch_ba"] - contrast["patch_aa"]).mean())
    dd_unrel = float((contrast["unrel_patch_xy"] - contrast["unrel_patch_yy"]).mean())
    summary = {
        "model": args.model, "filler": args.filler, "depth": depth,
        "closed": not args.no_close, "n_probes": len(probes), "skipped": skipped,
        "format_b": format_b, "prefix_a": args.prefix_a, "prefix_b": args.prefix_b,
        "patch_layers": args.patch_layers,
        "patch_positions": args.patch_positions, "random_control": args.random_control,
        "mean_contrast_by_condition": {k: float(v.mean()) for k, v in contrast.items()},
        "dd_ab": dd_ab, "dd_ba": dd_ba, "dd_unrelated": dd_unrel,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*66}\nE7 FORMAT PATCH — {args.filler} depth {depth} "
          f"({'closed' if not args.no_close else 'open'})\n{'='*66}")
    for k, v in contrast.items():
        print(f"  {k:16s} mean (S_A - S_B) = {float(v.mean()):+8.3f}")
    print(f"  dd_ab={dd_ab:+.3f}  dd_ba={dd_ba:+.3f}  dd_unrelated={dd_unrel:+.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
