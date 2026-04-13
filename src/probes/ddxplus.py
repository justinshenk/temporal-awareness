"""DDXPlus dataset helpers: evidence decoding and MCQ case formatting.

These are pure functions copied from
`scripts/context_fatigue/run_ddxplus_probe.py` so the probe infrastructure can
use them without depending on a script path. The original script is left
untouched to avoid disturbing existing Qwen results.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

OPTION_LABELS = ["A", "B", "C", "D", "E"]
SYSTEM_PROMPT = "You are a doctor."


def load_evidence_db(path: str | Path) -> dict:
    """Load the DDXPlus evidence database JSON."""
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


def decode_evidence(ev_str: str, evidence_db: dict) -> tuple[list[str], list[str]]:
    """Decode an evidence list string into (symptoms, antecedents)."""
    evs = ast.literal_eval(ev_str)
    grouped: dict[str, list[str]] = {}
    for ev in evs:
        if "@" in ev:
            base, val = ev.split("@", 1)
            grouped.setdefault(base.strip().rstrip("_"), []).append(val.strip())
        else:
            grouped[ev] = []

    symptoms: list[str] = []
    antecedents: list[str] = []
    for code, values in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        stmt = (
            info["question"]
            .replace("Do you have ", "Has ")
            .replace("Are you ", "Is ")
            .rstrip("?.")
        )
        if info["data_type"] == "B":
            text = f"Yes — {stmt}"
        elif info["data_type"] == "M" and values:
            dec = [
                info["value_meanings"].get(v, v)
                for v in values
                if info["value_meanings"].get(v, v) != "NA"
            ]
            text = f"{stmt}: {', '.join(dec)}" if dec else f"Yes — {stmt}"
        elif info["data_type"] == "C" and values:
            text = f"{stmt}: {', '.join(values)}"
        else:
            text = f"Yes — {stmt}"
        (antecedents if info["is_antecedent"] else symptoms).append(text)
    return symptoms, antecedents


def format_case_mcq(
    age: int,
    sex: str,
    initial_ev: str,
    evidence_str: str,
    evidence_db: dict,
    options: list[str],
) -> str:
    """Format a single DDXPlus case as an MCQ prompt string."""
    sex_full = "Male" if sex == "M" else "Female"
    chief = (
        evidence_db.get(initial_ev, {})
        .get("question", initial_ev)
        .replace("Do you have ", "")
        .replace("?", "")
        .strip()
    )
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


def extract_mcq_answer(text: str) -> str | None:
    """Extract A-E answer letter from model output.

    Parses common Gemma/Qwen MCQ continuations:
      - "B"           -> "B"
      - " B"          -> "B"
      - "B."          -> "B"
      - "B) ..."      -> "B"
      - "Answer: B"   -> "B"
      - "The answer is B" -> "B"
      - "B because..." -> "B"
      - "wrong format" -> None
    """
    if not text:
        return None
    text = text.strip().upper()
    if not text:
        return None
    m = re.match(r"^[\s\W]*([ABCDE])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCDE])\b", text)
    return m.group(1) if m else None
