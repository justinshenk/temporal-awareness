"""DDXPlus case decoding and MCQ formatting.

Lifted out of ``scripts/context_fatigue/_cf_common.py`` so it is importable, testable, and — the
reason it moved — **splittable**. E1 relocates the case vignette to an earlier turn while leaving
the 5-option question byte-identical, which needs the two halves as separate values rather than
one concatenated string.

``format_case_mcq`` composes the halves and its output is byte-for-byte what the committed drivers
already produce; a golden test pins that, because those bytes are the interface to results already
in the paper.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_PATH = _REPO_ROOT / "data" / "context_fatigue" / "release_evidences.json"


def load_evidence_db(path=DEFAULT_EVIDENCE_PATH) -> dict:
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        value_meanings = {}
        for vk, vv in info.get("value_meaning", {}).items():
            value_meanings[vk] = vv.get("en", str(vv)) if isinstance(vv, dict) else str(vv)
        db[code] = {
            "question": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "data_type": info.get("data_type", "B"),
            "value_meanings": value_meanings,
        }
    return db


def decode_evidence(ev_str: str, evidence_db: dict):
    """Turn DDXPlus evidence codes into (symptoms, antecedents) English statements."""
    evs = ast.literal_eval(ev_str)
    if not isinstance(evs, (list, tuple)):
        raise ValueError(f"evidence string must encode a list, got {type(evs).__name__}")
    symptoms, antecedents = [], []
    grouped: dict[str, list[str]] = {}
    for ev in evs:
        if "@" in ev:
            base, val = ev.split("@", 1)
            # DDXPlus encodes an evidence-value pair as ``E_54_@_V_112``: the underscores flanking
            # the ``@`` belong to the separator, not to the code or the value. Keeping the value's
            # leading underscore makes every lookup miss ``value_meaning`` (whose keys are
            # ``V_112``, ``6``, ...) and silently renders the raw code into the vignette.
            grouped.setdefault(base.strip().rstrip("_"), []).append(
                val.strip().lstrip("_"))
        else:
            grouped[ev] = []

    for code, values in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        statement = info["question"].replace("Do you have ", "Has ").replace("Are you ", "Is ")
        # Several DDXPlus questions already end in a colon ("Characterize your pain:"), which the
        # value-joining format would double into "pain:: chest".
        statement = statement.rstrip("?").rstrip(".").rstrip(":").rstrip()
        if info["data_type"] == "B":
            text = f"Yes — {statement}"
        elif info["data_type"] in ("M", "C") and values:
            # Categorical evidences carry a value table too (4 of the 10 C-type codes do, e.g.
            # E_204's travel regions). Joining their raw values leaves codes like "V_10" in the
            # vignette; codes without a table — the numeric scales — fall through unchanged.
            decoded = [info["value_meanings"].get(v, v) for v in values
                       if info["value_meanings"].get(v, v) not in ("NA", None, "")]
            text = f"{statement}: {', '.join(decoded)}" if decoded else f"Yes — {statement}"
        else:
            text = f"Yes — {statement}"
        (antecedents if info["is_antecedent"] else symptoms).append(text)
    return symptoms, antecedents


def format_case_vignette(age, sex, initial_ev, evidence_str, evidence_db) -> str:
    """The patient presentation — everything the answer depends on, and no options.

    This is the block E1 moves. It deliberately carries no option text and no answer cue, so
    relocating it changes *where the evidence lives* and nothing else.
    """
    sex_full = "Male" if sex == "M" else "Female"
    chief = evidence_db.get(initial_ev, {}).get("question", initial_ev)
    chief = chief.replace("Do you have ", "").replace("?", "").strip()
    symptoms, antecedents = decode_evidence(evidence_str, evidence_db)

    lines = [f"Patient: {age}-year-old {sex_full}", f"Chief complaint: {chief}"]
    if symptoms:
        lines.append("Symptoms:")
        lines.extend(f"  - {s}" for s in symptoms)
    if antecedents:
        lines.append("History:")
        lines.extend(f"  - {a}" for a in antecedents)
    return "\n".join(lines)


def format_case_question(options, n_options: int = 5, referent: str | None = None) -> str:
    """The 5-option question. Byte-identical across every E1 arm.

    ``referent`` prepends an explicit pointer back to the case ("For the patient described
    earlier"). In the deep ``back_k`` arms the question arrives many turns after the vignette with
    unrelated filler in between, and without a referent the arm partly measures whether the model
    noticed a patient was mentioned at all rather than whether it can use evidence at distance.
    It is applied to **every** arm, ``local`` included, so the question's bytes stay equal across
    the ladder. Off by default, so the single-turn case format the committed drivers emit is
    unchanged.
    """
    opener = "Most likely diagnosis:" if referent is None else f"{referent}, most likely diagnosis:"
    lines = [f"\n{opener}"]
    lines.extend(f"{OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(options[:n_options]))
    lines.append("\nAnswer:")
    return "\n".join(lines)


def format_case_mcq(age, sex, initial_ev, evidence_str, evidence_db, options, n_options=5) -> str:
    """Vignette and question as one block — the original single-turn case format."""
    return (format_case_vignette(age, sex, initial_ev, evidence_str, evidence_db)
            + "\n" + format_case_question(options, n_options))
