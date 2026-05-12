"""Loaders for Maar et al. (ICLR 2026) rhyme and QA datasets.

Source: supplementary material of
"What's the plan? Metrics for implicit planning in LLMs and their
application to rhyme generation and question answering"
(Maar, Paperno, McDougall, Nanda — ICLR 2026)

Prompt formats are reproduced byte-for-byte from Maar's
`paper_experiments/shared_utils.py` so probe activations are taken on
the EXACT same prompts where they applied causal steering. This is
what makes our staircase a discriminator against their ground truth
rather than a parallel-but-different measurement.

Datasets exposed:
    load_maar_rhyme(split)              → 10-class rhyme-family task
    load_maar_qa_suggestive(split)      → article (a/an) + noun task,
                                          surface content carries info
    load_maar_qa_neutral()              → article (a/an) + noun task,
                                          surface content cannot carry info
                                          (test-only; this is the
                                          within-task discriminator)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from ..utils.types import PlanningExample, TaskType


# ──────────────────────────────────────────────────────────────────────
# Exact prompt prefixes from Maar's shared_utils.py
# ──────────────────────────────────────────────────────────────────────

_RHYME_PREFIX = "A rhyming couplet:\n"

# Two-shot few-shot demonstration that anchors the (a / an) article pattern.
# Lifted verbatim from Maar's load_prompts() in shared_utils.py:803-832.
_QA_FEW_SHOT = (
    "Question: What two-wheeled vehicle do you pedal?\n"
    "Answer: a bicycle\n"
    "\n"
    "Question: What flying vehicle carries passengers in the sky?\n"
    "Answer: an airplane\n"
    "\n"
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _is_vowel_starting(noun: str) -> bool:
    """Article rule: 'an' for vowel-initial, 'a' for consonant-initial."""
    return noun[0].lower() in {"a", "e", "i", "o", "u"}


def _article_for(noun: str) -> str:
    return "an" if _is_vowel_starting(noun) else "a"


def _make_id(*parts) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _maar_data_root(data_root: str | Path | None) -> Path:
    """Resolve the supplementary_material root.

    If not provided, look for the canonical location: ./data/maar_supplementary_material/
    Override via env var MAAR_DATA_ROOT or explicit argument.
    """
    import os
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get("MAAR_DATA_ROOT")
    if env:
        return Path(env)
    return Path("data/maar_supplementary_material")


# ──────────────────────────────────────────────────────────────────────
# Rhyme loader
# ──────────────────────────────────────────────────────────────────────

# Canonical 10-family ordering (matches Maar's data files)
RHYME_FAMILIES = ("ing", "air", "ip", "oat", "ird", "ee", "ight", "ake", "ow", "it")


def load_maar_rhyme(
    split: Literal["train", "test"] = "test",
    data_root: str | Path | None = None,
) -> list[PlanningExample]:
    """Load Maar's rhyme-family-lines dataset as PlanningExamples.

    File layout expected:
        {data_root}/{split}/rhyme_family_lines.json

    The JSON is a dict: rhyme_family → list of first lines, each line
    already terminated with '\\n'. The full prompt is:

        "A rhyming couplet:\\n{first_line}"

    where {first_line} already includes its trailing newline.

    Label: rhyme_family (one of RHYME_FAMILIES, 10-way).
    """
    root = _maar_data_root(data_root)
    path = root / split / "rhyme_family_lines.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Maar rhyme dataset not found at {path}.\n"
            f"Extract supplementary materials and ensure structure: "
            f"{root}/{{train,test}}/rhyme_family_lines.json"
        )

    with open(path) as f:
        by_family = json.load(f)

    examples: list[PlanningExample] = []
    for family in RHYME_FAMILIES:
        if family not in by_family:
            continue
        for line_idx, first_line in enumerate(by_family[family]):
            prompt = _RHYME_PREFIX + first_line
            ex = PlanningExample(
                task_type=TaskType.RHYME,
                prompt=prompt,
                target_value=family,
                target_token_positions=[],  # filled by activation extractor
                metadata={
                    "source": "maar2026",
                    "split": split,
                    "rhyme_family": family,
                    "first_line": first_line,
                    "is_control": False,
                },
                example_id=_make_id("maar_rhyme", split, family, line_idx),
            )
            examples.append(ex)

    if not examples:
        raise RuntimeError(
            f"Loaded zero rhyme examples from {path}. "
            f"File contents may be malformed."
        )
    return examples


# ──────────────────────────────────────────────────────────────────────
# QA — suggestive
# ──────────────────────────────────────────────────────────────────────

def load_maar_qa_suggestive(
    split: Literal["train", "test"] = "test",
    data_root: str | Path | None = None,
) -> list[PlanningExample]:
    """Load Maar's suggestive QA dataset.

    File layout: {data_root}/{split}/noun_qa.json
    Structure: noun → list of suggestive questions ending in '?'

    Prompt format (matches Maar exactly):
        Two-shot bicycle/airplane prefix + "Question: {q}\\nAnswer:"

    Two labels per example:
        target_value: noun (29-way classification, secondary task)
        metadata.article: "a" or "an" (binary primary task)
        metadata.noun: same as target_value (for convenience)
    """
    root = _maar_data_root(data_root)
    path = root / split / "noun_qa.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Maar suggestive QA dataset not found at {path}."
        )

    with open(path) as f:
        by_noun = json.load(f)

    return _build_qa_examples(by_noun, kind="suggestive", split=split)


# ──────────────────────────────────────────────────────────────────────
# QA — neutral (the within-task discriminator)
# ──────────────────────────────────────────────────────────────────────

def load_maar_qa_neutral(
    data_root: str | Path | None = None,
    use_filtered: bool = True,
) -> list[PlanningExample]:
    """Load Maar's neutral QA dataset.

    Only the TEST split exists for neutral. Filtered version is the
    paper's curated set where neutrality was verified empirically.

    Same prompt format as suggestive. KEY PROPERTY: the same question
    text appears under both nouns of a pair (e.g., "What organ is
    essential for life?" under both 'eye' and 'heart'). This means a
    surface-feature probe (BoW, content) cannot distinguish — only a
    representation that has done planning can.
    """
    root = _maar_data_root(data_root)
    fname = "noun_qa_neutral_filtered.json" if use_filtered else "noun_qa_neutral.json"
    path = root / "test" / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Maar neutral QA dataset not found at {path}."
        )

    with open(path) as f:
        by_noun = json.load(f)

    return _build_qa_examples(by_noun, kind="neutral", split="test")


def _build_qa_examples(
    by_noun: dict[str, list[str]],
    kind: Literal["suggestive", "neutral"],
    split: str,
) -> list[PlanningExample]:
    """Shared construction logic for suggestive and neutral QA."""
    examples: list[PlanningExample] = []
    for noun, questions in sorted(by_noun.items()):
        article = _article_for(noun)
        for q_idx, question in enumerate(questions):
            # Match Maar's generation-time prompt exactly
            prompt = _QA_FEW_SHOT + f"Question: {question.strip()}\nAnswer:"
            ex = PlanningExample(
                task_type=TaskType.CODE_RETURN,  # closest existing enum value; see note below
                prompt=prompt,
                target_value=noun,  # primary noun-class label
                target_token_positions=[],
                metadata={
                    "source": "maar2026",
                    "kind": kind,
                    "split": split,
                    "noun": noun,
                    "article": article,
                    "question": question.strip(),
                    "is_control": False,
                },
                example_id=_make_id("maar_qa", kind, split, noun, q_idx),
            )
            examples.append(ex)

    if not examples:
        raise RuntimeError(
            f"Loaded zero {kind} QA examples — file appears empty."
        )
    return examples
# NOTE on task_type: the existing TaskType enum doesn't have QA values.
# We use CODE_RETURN as a placeholder and rely on metadata['kind'] +
# metadata['source'] for actual task routing. A future cleanup adds
# TaskType.QA_SUGGESTIVE / TaskType.QA_NEUTRAL but we keep this loader
# backward-compatible with the existing enum to avoid touching the
# workshop's serialized result files.


# ──────────────────────────────────────────────────────────────────────
# Dataset summary helper (useful for sanity checks / logging)
# ──────────────────────────────────────────────────────────────────────

def summarize(examples: list[PlanningExample]) -> dict:
    """Cheap structural summary for stdout/logs."""
    if not examples:
        return {"n": 0}

    # Inspect first example to deduce kind
    md0 = examples[0].metadata
    kind = md0.get("kind", "rhyme" if md0.get("rhyme_family") else "unknown")

    out: dict = {
        "n": len(examples),
        "kind": kind,
        "split": md0.get("split"),
    }

    if kind == "rhyme":
        from collections import Counter
        fam_counts = Counter(e.metadata.get("rhyme_family") for e in examples)
        out["families"] = dict(fam_counts)
    else:  # qa
        from collections import Counter
        noun_counts = Counter(e.metadata.get("noun") for e in examples)
        art_counts = Counter(e.metadata.get("article") for e in examples)
        out["n_nouns"] = len(noun_counts)
        out["article_dist"] = dict(art_counts)
        out["q_per_noun_avg"] = round(len(examples) / max(1, len(noun_counts)), 2)

    return out


__all__ = [
    "RHYME_FAMILIES",
    "load_maar_rhyme",
    "load_maar_qa_suggestive",
    "load_maar_qa_neutral",
    "summarize",
]
