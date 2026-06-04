"""Graded-distance-to-refusal MCQ buckets (MMLU subjects + WMDP) for the route sweep.

Holds the task FUNCTION fixed (4-option factual MCQ; see ``mcq_icl.py``) while dialing
topical proximity to the refusal axis ``r`` across four content buckets:

    B0 neutral          MMLU: geography, prehistory, world history, marketing
    B1 sensitive-benign MMLU: clinical knowledge, law, medicine, jurisprudence
    B2 dual-use         MMLU: college biology, anatomy, computer security, college medicine
    B3 hazardous        WMDP: bio, chem, cyber (hazardous knowledge as benign exam MCQs)

Every bucket yields uniform :class:`MCQItem` cases split into disjoint filler / fit / eval /
train slices. The dataset reads are thin; row-normalization, ordering, and splitting are pure
functions unit-tested on fake datasets (no network).

WMDP-cyber items embed long code snippets that blow the ICL token budget, so cyber rows are
ordered *after* the rest: the filler slice (taken from the front) avoids them, while they
still populate the fit/eval/train slices.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset

from src.common.base_schema import BaseSchema
from src.probes.safety.mcq_icl import MCQItem, mcq_item

B3 = "B3"
BUCKETS = ["B0", "B1", "B2", "B3"]

BUCKET_SUBJECTS = {
    "B0": ["high_school_geography", "prehistory", "high_school_world_history", "marketing"],
    "B1": ["clinical_knowledge", "professional_law", "professional_medicine", "jurisprudence"],
    "B2": ["college_biology", "anatomy", "computer_security", "college_medicine"],
}
WMDP_CONFIGS = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]
DEPRIORITIZED_SOURCES = {"wmdp-cyber"}  # long code-snippet items: keep out of the filler slice


@dataclass
class BucketData(BaseSchema):
    """Disjoint MCQ slices for one content bucket."""

    bucket: str = ""
    filler: list[MCQItem] = field(default_factory=list)
    fit: list[MCQItem] = field(default_factory=list)
    eval: list[MCQItem] = field(default_factory=list)
    train: list[MCQItem] = field(default_factory=list)
    n_available: int = 0


def normalize_row(row, source: str) -> dict:
    """Uniform ``{question, choices, answer, source}`` from an MMLU/WMDP row."""
    return {
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer": int(row["answer"]),
        "source": source,
    }


def mmlu_rows_for_bucket(mmlu_ds, bucket: str) -> list[dict]:
    """Normalized MMLU rows whose ``subject`` is in this bucket's subject list."""
    keep = set(BUCKET_SUBJECTS[bucket])
    return [normalize_row(r, r["subject"]) for r in mmlu_ds if r["subject"] in keep]


def wmdp_rows(wmdp_by_config) -> list[dict]:
    """Normalized rows from every WMDP config, tagged with the config as source."""
    return [normalize_row(r, cfg) for cfg in WMDP_CONFIGS for r in wmdp_by_config[cfg]]


def order_rows(rows: list[dict], seed: int) -> list[dict]:
    """Seeded order with deprioritized sources pushed to the back (front/back shuffled)."""
    rng = random.Random(seed)
    front = [r for r in rows if r["source"] not in DEPRIORITIZED_SOURCES]
    back = [r for r in rows if r["source"] in DEPRIORITIZED_SOURCES]
    rng.shuffle(front)
    rng.shuffle(back)
    return front + back


def bucket_rows(bucket: str, mmlu_ds, wmdp_by_config, seed: int) -> list[dict]:
    """Ordered normalized rows for one bucket (MMLU subjects, or WMDP for B3)."""
    rows = wmdp_rows(wmdp_by_config) if bucket == B3 else mmlu_rows_for_bucket(mmlu_ds, bucket)
    return order_rows(rows, seed)


def split_bucket(rows: list[dict], bucket: str, n_filler: int, n_fit: int,
                 n_eval: int, n_train: int) -> BucketData:
    """Carve disjoint filler/fit/eval/train MCQItem slices off ``rows`` (front to back)."""
    need = n_filler + n_fit + n_eval + n_train
    if len(rows) < need:
        raise ValueError(f"bucket {bucket}: need {need} rows, have {len(rows)}")
    cuts, a = [], 0
    for n in (n_filler, n_fit, n_eval, n_train):
        cuts.append([mcq_item(r, bucket, r["source"]) for r in rows[a:a + n]])
        a += n
    filler, fit, ev, train = cuts
    return BucketData(bucket, filler, fit, ev, train, len(rows))


def load_mmlu():
    return load_dataset("cais/mmlu", "all", split="test", token=os.environ.get("HF_TOKEN"))


def load_wmdp():
    tok = os.environ.get("HF_TOKEN")
    return {c: load_dataset("cais/wmdp", c, split="test", token=tok) for c in WMDP_CONFIGS}


def load_buckets(buckets, seed: int, n_filler: int, n_fit: int, n_eval: int,
                 n_train: int) -> dict[str, BucketData]:
    """Load + split every requested bucket. Only fetches MMLU/WMDP if a bucket needs it."""
    mmlu = load_mmlu() if any(b != B3 for b in buckets) else None
    wmdp = load_wmdp() if B3 in buckets else None
    return {
        b: split_bucket(bucket_rows(b, mmlu, wmdp, seed), b, n_filler, n_fit, n_eval, n_train)
        for b in buckets
    }


def mean_cosine_to_dir(acts, direction) -> float:
    """Mean cosine of each row of ``acts`` (n, d) with ``direction`` (d,).

    The bucket-level cos(a_i, r) used to verify the graded-distance-to-refusal premise.
    """
    X = np.asarray(acts, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    r = np.asarray(direction, dtype=np.float64)
    rn = np.linalg.norm(r)
    if rn == 0.0:
        raise ValueError("cannot take cosine against a zero direction")
    xn = np.linalg.norm(X, axis=1)
    if np.any(xn == 0.0):
        raise ValueError("zero activation row")
    return float(np.mean((X @ r) / (xn * rn)))


def is_monotone_increasing(values) -> bool:
    """True iff ``values`` is strictly increasing — the gradient gate's pass condition."""
    return all(b > a for a, b in zip(values, values[1:]))
