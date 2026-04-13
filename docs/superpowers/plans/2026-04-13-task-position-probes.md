# Task-Position Probes (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train linear probes on Gemma-9B-IT residual streams that decode three task-position targets (`task_index`, `within_task_fraction`, `tokens_until_boundary`) from DDXPlus multi-task traces, then run analyses A1/A2/A4 to test whether fatigue tracks the model's subjective sense of task-lateness rather than raw context length.

**Architecture:** New module `src/probes/` with pure-function labelers, reusable per-token residual-stream extraction, ridge/MLP probe training, and analysis functions. Existing Qwen-only last-token extraction in `scripts/context_fatigue/run_ddxplus_probe.py` is left untouched; v1 is a parallel clean implementation because it needs per-token activations that the existing capture doesn't provide.

**Tech Stack:** PyTorch, HuggingFace Transformers (Gemma-2-9B-IT), scikit-learn (ridge), NumPy/Pandas, existing `BaseSchema` from `src/common/base_schema.py`.

**Branch:** `context-fatigue-datasets` (no separate worktree; work continues on this branch).

**Spec:** `docs/superpowers/specs/2026-04-13-task-position-probes-design.md`

**Open data question resolved:** Existing activations on disk are last-token-only (see `scripts/context_fatigue/run_ddxplus_probe.py:136`). Per-token re-extraction on Gemma-9B-IT is required regardless. Task 5 handles this.

**Compute budget:** Gemma-9B-IT in bf16 is ~18 GB; 8k context window fits on one A100/H100. Per-token activation storage for 5 layers at hidden_dim=3584 is ~140 KB/token → ~1 GB per trace at 8k tokens. Plan for ~20 traces → ~20 GB on disk.

---

## File structure

**New files:**

| Path | Responsibility |
|---|---|
| `src/probes/__init__.py` | Auto-export public symbols from submodules |
| `src/probes/ddxplus.py` | DDXPlus case loading, evidence decoding, MCQ formatting (extracted from the existing script, not modified in place) |
| `src/probes/extraction.py` | Per-token residual-stream capture hook; one reusable `PerTokenResidualCapture` class |
| `src/probes/task_position/__init__.py` | Auto-export |
| `src/probes/task_position/labels.py` | `TaskPositionLabels` BaseSchema dataclass + pure-function labeler `label_trace()` |
| `src/probes/task_position/probes.py` | `RidgeProbe` class (fit/predict/save/load); shared base for later `MLPProbe` |
| `src/probes/task_position/analysis.py` | Functions `analysis_a1_orthogonality`, `analysis_a2_failure_prediction`, `analysis_a4_calibration_gap` |
| `tests/probes/__init__.py` | Empty — marks as test package |
| `tests/probes/task_position/__init__.py` | Empty |
| `tests/probes/task_position/test_labels.py` | Unit tests for the labeler (pure functions; no model required) |
| `tests/probes/task_position/test_probes.py` | Unit tests for ridge probe on synthetic data (smoke test + reconstruction) |
| `scripts/probes/task_position/extract_activations.py` | Driver: runs Gemma-9B-IT on DDXPlus, saves per-token residuals + labels |
| `scripts/probes/task_position/train_probes.py` | Driver: fits all (target × layer) ridge probes, saves artifacts |
| `scripts/probes/task_position/run_analyses.py` | Driver: runs A1/A2/A4, writes results markdown |

**Modified files:** None in MVP. (The existing `scripts/context_fatigue/run_ddxplus_probe.py` is explicitly left alone — rewiring it risks breaking published Qwen results.)

**Output artifacts:**
- `results/probes/task_position/gemma-9b-it/activations.pt` — per-token residuals + labels
- `results/probes/task_position/gemma-9b-it/probes/<target>_<layer>.pkl` — trained ridge probes
- `results/probes/task_position/2026-04-13-v1-results.md` — analysis writeup

---

## Task 1: Module skeleton and labels dataclass

**Files:**
- Create: `src/probes/__init__.py`
- Create: `src/probes/task_position/__init__.py`
- Create: `src/probes/task_position/labels.py`

- [ ] **Step 1: Create the `src/probes/` package with auto-export**

Create `src/probes/__init__.py`:

```python
"""Probe infrastructure for temporal/task-position analyses."""

from src.common.auto_export import auto_export

auto_export(__name__, __path__)
```

- [ ] **Step 2: Create the `src/probes/task_position/` subpackage**

Create `src/probes/task_position/__init__.py`:

```python
"""Task-position probes: decode model's representation of task progress."""

from src.common.auto_export import auto_export

auto_export(__name__, __path__)
```

- [ ] **Step 3: Create the labels dataclass**

Create `src/probes/task_position/labels.py`:

```python
"""Pure-function labelers for task-position targets.

Produces per-token labels for:
  - task_index: 1..N ordinal position of the current task in the trace
  - within_task_fraction: [0, 1] fractional progress through the current task
  - tokens_until_boundary: positive int, distance to the next task boundary

Labelers are pure functions on (trace_length, case_boundaries). No model required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.common.base_schema import BaseSchema


@dataclass
class TaskPositionLabels(BaseSchema):
    """Per-token task-position labels for one trace.

    All arrays have length equal to the trace's token count.
    """

    task_index: list[int] = field(default_factory=list)
    within_task_fraction: list[float] = field(default_factory=list)
    tokens_until_boundary: list[int] = field(default_factory=list)
    raw_token_position: list[int] = field(default_factory=list)
    total_context_length: int = 0


def label_trace(
    trace_length: int,
    case_boundaries: Sequence[int],
) -> TaskPositionLabels:
    """Produce per-token task-position labels.

    Args:
        trace_length: total number of tokens in the trace
        case_boundaries: sorted list of token indices where each case begins.
            Must start at 0 (first case begins at token 0). The sentinel
            boundary at `trace_length` is appended internally — callers must
            NOT include it.

    Returns:
        TaskPositionLabels with arrays of length `trace_length`.

    Raises:
        ValueError: if boundaries are not sorted, not starting at 0, or empty.
    """
    if not case_boundaries:
        raise ValueError("case_boundaries must be non-empty")
    if case_boundaries[0] != 0:
        raise ValueError(f"case_boundaries must start at 0, got {case_boundaries[0]}")
    if list(case_boundaries) != sorted(case_boundaries):
        raise ValueError("case_boundaries must be sorted ascending")
    if case_boundaries[-1] >= trace_length:
        raise ValueError(
            f"last boundary {case_boundaries[-1]} must be < trace_length {trace_length}"
        )

    boundaries = list(case_boundaries) + [trace_length]
    n_cases = len(case_boundaries)

    task_index = np.zeros(trace_length, dtype=np.int64)
    within_task_fraction = np.zeros(trace_length, dtype=np.float64)
    tokens_until_boundary = np.zeros(trace_length, dtype=np.int64)

    for i in range(n_cases):
        start = boundaries[i]
        end = boundaries[i + 1]
        case_length = end - start
        for t in range(start, end):
            task_index[t] = i + 1  # 1-indexed
            within_task_fraction[t] = (t - start) / case_length
            tokens_until_boundary[t] = end - t

    return TaskPositionLabels(
        task_index=task_index.tolist(),
        within_task_fraction=within_task_fraction.tolist(),
        tokens_until_boundary=tokens_until_boundary.tolist(),
        raw_token_position=list(range(trace_length)),
        total_context_length=trace_length,
    )
```

- [ ] **Step 4: Commit**

```bash
git add src/probes/__init__.py src/probes/task_position/__init__.py src/probes/task_position/labels.py
git commit -m "Add task-position labels module skeleton"
```

---

## Task 2: Labeler unit tests

**Files:**
- Create: `tests/probes/__init__.py`
- Create: `tests/probes/task_position/__init__.py`
- Create: `tests/probes/task_position/test_labels.py`

- [ ] **Step 1: Create test package markers**

```bash
touch tests/probes/__init__.py tests/probes/task_position/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/probes/task_position/test_labels.py`:

```python
"""Tests for task-position labelers."""

import pytest

from src.probes.task_position.labels import TaskPositionLabels, label_trace


def test_single_task_full_trace():
    labels = label_trace(trace_length=10, case_boundaries=[0])
    assert labels.task_index == [1] * 10
    assert labels.within_task_fraction == pytest.approx(
        [i / 10 for i in range(10)]
    )
    assert labels.tokens_until_boundary == list(range(10, 0, -1))
    assert labels.raw_token_position == list(range(10))
    assert labels.total_context_length == 10


def test_two_equal_tasks():
    labels = label_trace(trace_length=10, case_boundaries=[0, 5])
    assert labels.task_index == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert labels.within_task_fraction == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8]
    )
    assert labels.tokens_until_boundary == [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]


def test_three_unequal_tasks():
    labels = label_trace(trace_length=12, case_boundaries=[0, 4, 10])
    assert labels.task_index == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
    assert labels.tokens_until_boundary == [4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 2, 1]


def test_boundaries_must_start_at_zero():
    with pytest.raises(ValueError, match="must start at 0"):
        label_trace(trace_length=10, case_boundaries=[2, 5])


def test_boundaries_must_be_sorted():
    with pytest.raises(ValueError, match="sorted ascending"):
        label_trace(trace_length=10, case_boundaries=[0, 5, 3])


def test_last_boundary_must_be_less_than_trace_length():
    with pytest.raises(ValueError, match="must be <"):
        label_trace(trace_length=10, case_boundaries=[0, 10])


def test_empty_boundaries_raises():
    with pytest.raises(ValueError, match="non-empty"):
        label_trace(trace_length=10, case_boundaries=[])


def test_serialization_roundtrip():
    labels = label_trace(trace_length=6, case_boundaries=[0, 3])
    as_dict = labels.to_dict()
    assert as_dict["task_index"] == [1, 1, 1, 2, 2, 2]
    assert as_dict["total_context_length"] == 6
```

- [ ] **Step 3: Run the tests to verify they pass**

```bash
cd /workspace/temporal-awareness
uv run pytest tests/probes/task_position/test_labels.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/probes/__init__.py tests/probes/task_position/__init__.py tests/probes/task_position/test_labels.py
git commit -m "Add labeler unit tests"
```

---

## Task 3: DDXPlus helpers factored into src/probes/ddxplus.py

**Files:**
- Create: `src/probes/ddxplus.py`

The existing helpers (`load_evidence_db`, `decode_evidence`, `format_case_mcq`, `extract_mcq_answer`) live at module scope in `scripts/context_fatigue/run_ddxplus_probe.py`. Importing from a script path is awkward; instead, copy these into the new module verbatim (they are pure functions). The existing script is left alone.

- [ ] **Step 1: Create `src/probes/ddxplus.py` with the four helper functions**

Create `src/probes/ddxplus.py`:

```python
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
    """Extract A-E answer letter from model output."""
    text = text.strip().upper()
    if text and text[0] in "ABCDE":
        return text[0]
    m = re.search(r"\b([ABCDE])\b", text)
    return m.group(1) if m else None
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
uv run python -c "from src.probes.ddxplus import load_evidence_db, format_case_mcq, extract_mcq_answer, decode_evidence, SYSTEM_PROMPT, OPTION_LABELS; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/probes/ddxplus.py
git commit -m "Factor DDXPlus helpers into src/probes/ddxplus.py"
```

---

## Task 4: Per-token residual capture

**Files:**
- Create: `src/probes/extraction.py`

The existing `ResidualCapture` class captures only the last token. We need every token's residual stream at the specified layers. This captures the full `(seq_len, hidden_dim)` tensor per layer.

- [ ] **Step 1: Create the per-token capture class**

Create `src/probes/extraction.py`:

```python
"""Per-token residual-stream extraction hooks.

Captures the residual stream output at specified layers for every token in a
forward pass. Produces `dict[int, Tensor]` keyed by layer index, each tensor
shape `(seq_len, hidden_dim)` on CPU in float32.
"""

from __future__ import annotations

from typing import Sequence

import torch


class PerTokenResidualCapture:
    """Captures per-token residual streams at a set of layers.

    Usage:
        capture = PerTokenResidualCapture(model, layers=[0, 10, 20, 30, 41])
        capture.enabled = True
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        capture.enabled = False
        acts = capture.captured  # dict[int, Tensor(seq_len, hidden_dim)]
        capture.clear()
        ...
        capture.remove()  # release hooks when done
    """

    def __init__(self, model, layers: Sequence[int]):
        self.layers = list(layers)
        self.captured: dict[int, torch.Tensor] = {}
        self.enabled: bool = False
        self._hooks: list = []

        for li in self.layers:
            hook = model.model.layers[li].register_forward_hook(self._make_hook(li))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return
            hs = output[0] if isinstance(output, tuple) else output
            # hs shape: (batch, seq_len, hidden_dim). Assume batch=1.
            self.captured[layer_idx] = hs[0].detach().float().cpu()

        return hook_fn

    def clear(self) -> None:
        self.captured = {}

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
uv run python -c "from src.probes.extraction import PerTokenResidualCapture; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/probes/extraction.py
git commit -m "Add per-token residual capture"
```

---

## Task 5: Extraction driver — run Gemma-9B-IT on DDXPlus, save per-token activations and labels

**Files:**
- Create: `scripts/probes/task_position/__init__.py` (empty)
- Create: `scripts/probes/task_position/extract_activations.py`

Each trace accumulates cases until context fills (target 90% of Gemma's 8k window). At each new case, we record the token index where the case prompt begins — that's a boundary. After the trace finishes, we run one more forward pass over the full concatenated conversation with the per-token capture enabled, then label every token using `label_trace`.

**Key design choice:** we do the labels pass *once at the end* with the full accumulated conversation, not incrementally. This avoids re-capturing activations per case and ensures boundaries align with the tokenizer output exactly.

- [ ] **Step 1: Create the driver directory marker**

```bash
mkdir -p scripts/probes/task_position && touch scripts/probes/task_position/__init__.py
```

- [ ] **Step 2: Write the extraction driver**

Create `scripts/probes/task_position/extract_activations.py`:

```python
"""Extract per-token residual streams + task-position labels on DDXPlus.

Runs Gemma-9B-IT (default) through a multi-case DDXPlus trace, capturing
residual streams at specified layers for every token. Saves activations and
per-token labels to disk.
"""

from __future__ import annotations

import argparse
import ast
import gc
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import (
    OPTION_LABELS,
    SYSTEM_PROMPT,
    format_case_mcq,
    load_evidence_db,
)
from src.probes.extraction import PerTokenResidualCapture
from src.probes.task_position.labels import label_trace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--fill-target", type=float, default=0.90)
    p.add_argument("--n-traces", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default="0,10,20,30,41")
    p.add_argument("--evidence-db", default="release_evidences.json")
    p.add_argument("--out-dir", default="results/probes/task_position/gemma-9b-it")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_trace(tokenizer, ds, valid_indices, evidence_db, max_ctx, fill_target, rng):
    """Build one trace: accumulate DDXPlus cases until context fills.

    Returns:
        tokens: List[int], the tokenized concatenated conversation
        case_start_tokens: List[int], token index where each case begins (starts at 0)
    """
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    case_start_tokens: list[int] = []

    case_rng = random.Random(rng.randint(0, 2**31 - 1))
    indices = list(valid_indices)
    case_rng.shuffle(indices)

    for idx in indices:
        # Measure current tokenization with a probe prompt appended
        text_before = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        ids_before = tokenizer(text_before, return_tensors="pt").input_ids[0]
        n_before = ids_before.shape[0]

        if n_before / max_ctx > fill_target:
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

        # The case begins at token n_before in the concatenated conversation
        case_start_tokens.append(n_before)

        conversation.append({"role": "user", "content": case_text})
        # Generate a short assistant placeholder (greedy would need a model pass).
        # For labeling purposes we insert the gold letter as the assistant response;
        # this keeps the conversation well-formed without a generation pass during
        # extraction. Accuracy labels for A2 come from a separate evaluation pass.
        conversation.append({"role": "assistant", "content": gold_letter})

    final_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    final_ids = tokenizer(final_text, return_tensors="pt").input_ids[0].tolist()
    return final_ids, case_start_tokens


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    evidence_db = load_evidence_db(args.evidence_db)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid_indices = [
        i
        for i in range(len(ds))
        if ds[i]["PATHOLOGY"] in [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:5]]
    ]

    capture = PerTokenResidualCapture(model, layers=layers)

    trace_rng = random.Random(args.seed)
    all_traces = []

    for trace_i in range(args.n_traces):
        print(f"\nTrace {trace_i + 1}/{args.n_traces}: building...")
        tokens, case_boundaries = build_trace(
            tokenizer, ds, valid_indices, evidence_db, args.max_ctx, args.fill_target, trace_rng
        )
        n_tokens = len(tokens)
        n_cases = len(case_boundaries)
        print(f"  tokens={n_tokens} cases={n_cases}")

        input_ids = torch.tensor([tokens], device=args.device)
        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        capture.enabled = False
        del _

        acts_by_layer = {li: capture.captured[li].clone() for li in layers}
        labels = label_trace(trace_length=n_tokens, case_boundaries=case_boundaries)

        all_traces.append(
            {
                "trace_id": trace_i,
                "tokens": tokens,
                "case_boundaries": case_boundaries,
                "labels": labels.to_dict(),
                "activations": acts_by_layer,
            }
        )

        torch.cuda.empty_cache()
        gc.collect()

    capture.remove()

    out_file = out_dir / "activations.pt"
    torch.save({"layers": layers, "traces": all_traces}, out_file)
    print(f"\nSaved {len(all_traces)} traces to {out_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the extraction on a small smoke configuration**

```bash
cd /workspace/temporal-awareness
HF_TOKEN=<token> uv run python -m scripts.probes.task_position.extract_activations --n-traces 2
```

Expected: runs to completion, saves `results/probes/task_position/gemma-9b-it/activations.pt`. The run may take a few minutes per trace.

- [ ] **Step 4: Verify the saved file structure**

```bash
uv run python -c "
import torch
d = torch.load('results/probes/task_position/gemma-9b-it/activations.pt', weights_only=False)
print('layers:', d['layers'])
print('num traces:', len(d['traces']))
t = d['traces'][0]
print('trace 0 tokens:', len(t['tokens']), 'cases:', len(t['case_boundaries']))
print('activation shape L20:', t['activations'][20].shape)
print('labels keys:', list(t['labels'].keys()))
"
```

Expected: activation shape is `(n_tokens, 3584)` for Gemma-9B. Labels contains `task_index`, `within_task_fraction`, `tokens_until_boundary`, etc.

- [ ] **Step 5: Commit**

```bash
git add scripts/probes/task_position/__init__.py scripts/probes/task_position/extract_activations.py
git commit -m "Add Gemma per-token DDXPlus extraction driver"
```

- [ ] **Step 6: Run the full extraction**

```bash
HF_TOKEN=<token> uv run python -m scripts.probes.task_position.extract_activations --n-traces 20
```

This produces the full dataset for training. Runtime estimate: ~30-60 min on one A100.

---

## Task 6: Ridge probe class with unit tests

**Files:**
- Create: `src/probes/task_position/probes.py`
- Create: `tests/probes/task_position/test_probes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/probes/task_position/test_probes.py`:

```python
"""Tests for ridge probes on synthetic data."""

import numpy as np
import pytest

from src.probes.task_position.probes import RidgeProbe


def test_ridge_recovers_linear_signal():
    """A linear probe should recover a linear signal from noisy features."""
    rng = np.random.default_rng(0)
    d = 16
    n = 500
    true_w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ true_w + 0.05 * rng.normal(size=n)

    probe = RidgeProbe(alpha=1.0)
    probe.fit(X[:400], y[:400])
    r2 = probe.score(X[400:], y[400:])
    assert r2 > 0.95, f"expected R² > 0.95, got {r2}"


def test_ridge_direction_is_unit_dim_d():
    probe = RidgeProbe()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 8))
    y = rng.normal(size=100)
    probe.fit(X, y)
    direction = probe.direction()
    assert direction.shape == (8,)


def test_ridge_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 8))
    y = rng.normal(size=100)

    probe = RidgeProbe(alpha=0.5)
    probe.fit(X, y)
    path = tmp_path / "probe.pkl"
    probe.save(path)

    loaded = RidgeProbe.load(path)
    assert np.allclose(probe.predict(X), loaded.predict(X))
    assert loaded.alpha == 0.5


def test_ridge_raises_if_not_fit():
    probe = RidgeProbe()
    with pytest.raises(RuntimeError, match="not fit"):
        probe.predict(np.zeros((1, 8)))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/probes/task_position/test_probes.py -v
```

Expected: ImportError — `probes` module doesn't exist yet.

- [ ] **Step 3: Implement `RidgeProbe`**

Create `src/probes/task_position/probes.py`:

```python
"""Probe training and inference for task-position targets."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


class RidgeProbe:
    """Ridge regression probe on residual-stream activations.

    Wraps sklearn's Ridge with save/load, a direction accessor, and R² scoring.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("probe not fit; call .fit(X, y) first")
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))

    def direction(self) -> np.ndarray:
        """Return the learned linear direction (coefficient vector)."""
        if self._model is None:
            raise RuntimeError("probe not fit; call .fit(X, y) first")
        return self._model.coef_.copy()

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"alpha": self.alpha, "model": self._model}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        with open(path, "rb") as f:
            data = pickle.load(f)
        probe = cls(alpha=data["alpha"])
        probe._model = data["model"]
        return probe
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/probes/task_position/test_probes.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/probes/task_position/probes.py tests/probes/task_position/test_probes.py
git commit -m "Add ridge probe with tests"
```

---

## Task 7: Training driver — fit all (target × layer) probes with by-trace splits

**Files:**
- Create: `scripts/probes/task_position/train_probes.py`

Uses the saved `activations.pt` from Task 5. Splits traces 80/20 by trace id (not token id) to avoid leakage. Fits one ridge probe per (target, layer). Saves probes and a metrics CSV.

- [ ] **Step 1: Write the training driver**

Create `scripts/probes/task_position/train_probes.py`:

```python
"""Train ridge probes for each (target, layer) on saved activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.probes.task_position.probes import RidgeProbe

TARGETS = ["task_index", "within_task_fraction", "tokens_until_boundary"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="results/probes/task_position/gemma-9b-it/activations.pt")
    p.add_argument("--out-dir", default="results/probes/task_position/gemma-9b-it/probes")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


def split_traces(n_traces: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traces)
    n_test = max(1, int(round(n_traces * test_frac)))
    test_ids = set(perm[:n_test].tolist())
    train_ids = set(perm[n_test:].tolist())
    return train_ids, test_ids


def build_matrix(traces: list, layer: int, target: str, trace_ids: set):
    """Stack per-token activations and labels for the given layer and target."""
    Xs, ys, trace_of_token = [], [], []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        act = t["activations"][layer].numpy()
        labels = t["labels"][target]
        if target == "tokens_until_boundary":
            labels = np.log1p(np.asarray(labels, dtype=np.float64))
        else:
            labels = np.asarray(labels, dtype=np.float64)
        Xs.append(act.astype(np.float32))
        ys.append(labels)
        trace_of_token.extend([t["trace_id"]] * act.shape[0])
    return np.vstack(Xs), np.concatenate(ys), np.array(trace_of_token)


def primary_metric(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if target == "task_index":
        return float(spearmanr(y_true, y_pred).correlation)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    layers = blob["layers"]
    traces = blob["traces"]
    n_traces = len(traces)
    train_ids, test_ids = split_traces(n_traces, args.test_frac, args.seed)
    print(f"n_traces={n_traces} train={len(train_ids)} test={len(test_ids)}")

    results = []
    for target in TARGETS:
        for layer in layers:
            X_train, y_train, _ = build_matrix(traces, layer, target, train_ids)
            X_test, y_test, _ = build_matrix(traces, layer, target, test_ids)

            probe = RidgeProbe(alpha=args.alpha)
            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            metric = primary_metric(target, y_test, y_pred)

            # Baseline 1: raw token position as sole feature
            from sklearn.linear_model import Ridge as SK
            raw_pos_train = np.arange(X_train.shape[0]).reshape(-1, 1).astype(np.float32)
            raw_pos_test = np.arange(X_test.shape[0]).reshape(-1, 1).astype(np.float32)
            baseline = SK(alpha=args.alpha).fit(raw_pos_train, y_train)
            baseline_metric = primary_metric(target, y_test, baseline.predict(raw_pos_test))

            probe_path = out_dir / f"{target}_L{layer}.pkl"
            probe.save(probe_path)

            results.append(
                {
                    "target": target,
                    "layer": layer,
                    "metric": metric,
                    "baseline_raw_pos": baseline_metric,
                    "delta": metric - baseline_metric,
                    "n_train_tokens": X_train.shape[0],
                    "n_test_tokens": X_test.shape[0],
                }
            )
            print(
                f"  {target:25s} L{layer:<3d} metric={metric:.4f} "
                f"baseline={baseline_metric:.4f} Δ={metric - baseline_metric:+.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "split.json", "w") as f:
        json.dump({"train_ids": sorted(train_ids), "test_ids": sorted(test_ids)}, f)
    print(f"\nSaved probes and metrics to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the training driver**

```bash
uv run python -m scripts.probes.task_position.train_probes
```

Expected: 3 targets × 5 layers = 15 probes trained, CSV written, per-probe metrics printed.

- [ ] **Step 3: Spot-check the metrics**

Verify: at least one (target, layer) combination has `delta > 0` (probe beats raw-position baseline).

- [ ] **Step 4: Commit**

```bash
git add scripts/probes/task_position/train_probes.py
git commit -m "Add task-position probe training driver"
```

---

## Task 8: Analysis A1 — orthogonality to context-fill

**Files:**
- Create: `src/probes/task_position/analysis.py`

A1 computes cosine similarity between each task-position probe direction and the context-fill direction. For v1, we define the context-fill direction as the ridge solution of the `raw_token_position` target on the same activations, at the same layer. This is a clean proxy: if a task-position direction is orthogonal to the pure-position direction, it's encoding something new.

- [ ] **Step 1: Implement `analysis_a1_orthogonality`**

Create `src/probes/task_position/analysis.py`:

```python
"""Analyses A1/A2/A4 for task-position probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.probes.task_position.probes import RidgeProbe


def _load_layer_matrix(traces: list, layer: int, target: str, trace_ids: set):
    Xs, ys = [], []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        Xs.append(t["activations"][layer].numpy().astype(np.float32))
        labels = t["labels"][target]
        if target == "tokens_until_boundary":
            labels = np.log1p(np.asarray(labels, dtype=np.float64))
        else:
            labels = np.asarray(labels, dtype=np.float64)
        ys.append(labels)
    return np.vstack(Xs), np.concatenate(ys)


def analysis_a1_orthogonality(
    traces: list,
    layers: list[int],
    train_ids: set,
    targets: list[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Cosine similarity between task-position and raw-position probe directions.

    Prediction from spec:
      - task_index aligns strongly with raw-position (both monotone in tokens)
      - within_task_fraction and tokens_until_boundary are substantially orthogonal
    """
    rows = []
    for layer in layers:
        # Fit raw-position probe at this layer
        X_train, _ = _load_layer_matrix(traces, layer, "task_index", train_ids)
        raw_pos = np.concatenate(
            [np.arange(t["activations"][layer].shape[0]) for t in traces if t["trace_id"] in train_ids]
        ).astype(np.float64)
        raw_probe = RidgeProbe(alpha=alpha).fit(X_train, raw_pos)
        raw_dir = raw_probe.direction()
        raw_dir /= np.linalg.norm(raw_dir) + 1e-12

        for target in targets:
            _, y_train = _load_layer_matrix(traces, layer, target, train_ids)
            probe = RidgeProbe(alpha=alpha).fit(X_train, y_train)
            d = probe.direction()
            d /= np.linalg.norm(d) + 1e-12
            cos = float(np.dot(d, raw_dir))
            rows.append({"layer": layer, "target": target, "cosine_with_raw_pos": cos})
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Commit**

```bash
git add src/probes/task_position/analysis.py
git commit -m "Add analysis A1 (orthogonality)"
```

---

## Task 9: Analysis A2 — upcoming failure prediction

**Files:**
- Modify: `src/probes/task_position/analysis.py`

A2 needs correctness labels, which require a separate model evaluation pass per case. The cleanest way: run the extraction driver once more with `--eval-correctness` that generates a short answer after each case and records whether it matched the gold letter. For the MVP, we add a lightweight correctness-evaluation script, saved as `correctness.json` alongside `activations.pt`, and A2 consumes it.

**However**, to keep this task tractable, we implement A2 as **conditional on the correctness file existing**. If it's absent, A2 returns a DataFrame with a placeholder row noting "correctness file missing — rerun extraction with --eval-correctness".

- [ ] **Step 1: Extend extraction driver to optionally evaluate correctness**

Add an `--eval-correctness` flag to `scripts/probes/task_position/extract_activations.py`. When set, after building the trace, generate a short answer (max 5 new tokens, greedy) for each case's MCQ prompt and record `(case_index, gold_letter, pred_letter, correct)`. Save to `results/probes/task_position/gemma-9b-it/correctness.json`.

Add to `extract_activations.py` inside `build_trace()` — actually, correctness must be evaluated *during* trace construction because it depends on the accumulating context. Add a second trace-walk mode: in eval mode, after appending each case's user turn, generate the answer instead of inserting the gold letter as assistant. Record the result, then continue.

Concretely, modify `build_trace()` to accept `model=None` and `max_new=5` parameters; when `model is not None`, greedily generate after each case and record correctness; the assistant turn uses the generated answer.

```python
# In build_trace, when model is provided:
if model is not None:
    chat_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    resp = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
    pred = extract_mcq_answer(resp) or "?"
    correctness_records.append({
        "case_index": len(case_start_tokens) - 1,
        "gold": gold_letter,
        "pred": pred,
        "correct": pred == gold_letter,
    })
    conversation.append({"role": "assistant", "content": resp})
else:
    conversation.append({"role": "assistant", "content": gold_letter})
```

Save `correctness.json` next to `activations.pt`. (Full code for this modification is left to the engineer following the inline snippet — the import additions needed are `from src.probes.ddxplus import extract_mcq_answer`.)

- [ ] **Step 2: Re-run extraction with correctness evaluation**

```bash
HF_TOKEN=<token> uv run python -m scripts.probes.task_position.extract_activations \
    --n-traces 20 --eval-correctness
```

- [ ] **Step 3: Implement `analysis_a2_failure_prediction`**

Append to `src/probes/task_position/analysis.py`:

```python
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def analysis_a2_failure_prediction(
    traces: list,
    layers: list[int],
    train_ids: set,
    test_ids: set,
    correctness_path: Path | str,
) -> pd.DataFrame:
    """Predict upcoming case correctness from residual stream + task-position features.

    Baseline: residual stream alone. Treatment: residual stream + concatenated
    task-position scalars (task_index, within_task_fraction, tokens_until_boundary).
    Compared at each layer; target AUC to beat is 0.68 (from the writeup).
    """
    if not Path(correctness_path).exists():
        return pd.DataFrame(
            [{"status": "correctness file missing — rerun extraction with --eval-correctness"}]
        )

    with open(correctness_path) as f:
        correctness_by_trace = json.load(f)

    rows = []
    for layer in layers:

        def gather(trace_ids):
            Xs, y_labels, tp_feats = [], [], []
            for t in traces:
                if t["trace_id"] not in trace_ids:
                    continue
                cr = correctness_by_trace.get(str(t["trace_id"]), [])
                if not cr:
                    continue
                boundaries = t["case_boundaries"]
                act = t["activations"][layer].numpy().astype(np.float32)
                labels = t["labels"]
                for case in cr:
                    ci = case["case_index"]
                    # Use the last token of the case's prompt as the prediction site
                    if ci + 1 < len(boundaries):
                        pos = boundaries[ci + 1] - 1
                    else:
                        pos = act.shape[0] - 1
                    Xs.append(act[pos])
                    y_labels.append(1 if case["correct"] else 0)
                    tp_feats.append(
                        [
                            labels["task_index"][pos],
                            labels["within_task_fraction"][pos],
                            np.log1p(labels["tokens_until_boundary"][pos]),
                        ]
                    )
            return np.array(Xs), np.array(y_labels), np.array(tp_feats)

        X_tr, y_tr, tp_tr = gather(train_ids)
        X_te, y_te, tp_te = gather(test_ids)
        if y_tr.size == 0 or y_te.size == 0 or len(np.unique(y_te)) < 2:
            rows.append({"layer": layer, "status": "insufficient data"})
            continue

        base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_te, base.predict_proba(X_te)[:, 1])

        X_tr_aug = np.hstack([X_tr, tp_tr])
        X_te_aug = np.hstack([X_te, tp_te])
        aug = LogisticRegression(max_iter=2000).fit(X_tr_aug, y_tr)
        aug_auc = roc_auc_score(y_te, aug.predict_proba(X_te_aug)[:, 1])

        rows.append(
            {
                "layer": layer,
                "baseline_auc": base_auc,
                "with_task_position_auc": aug_auc,
                "delta": aug_auc - base_auc,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Commit**

```bash
git add src/probes/task_position/analysis.py scripts/probes/task_position/extract_activations.py
git commit -m "Add analysis A2 (upcoming failure prediction)"
```

---

## Task 10: Analysis A4 — calibration gap vs probe readout

**Files:**
- Modify: `src/probes/task_position/analysis.py`

A4 measures whether overconfidence grows with the model's subjective sense of lateness. For each case in a trace with correctness available, compute:
- **Confidence:** from a second logits-capturing pass, take the softmax probability of the predicted letter among {A,B,C,D,E}
- **Correctness:** 0/1 from correctness.json
- **Subjective-lateness:** the trained `within_task_fraction` probe's readout at the case's last prompt token

Bin cases by subjective lateness decile, compute ECE per bin. Compare with the same binning by raw token position.

For v1, we simplify: confidence per case is taken as the argmax logit normalized over A-E options, captured during the correctness pass (requires modifying the eval to return per-case top-5 logits). To keep scope contained, Task 10 only plots correctness-rate vs bin — a proxy for calibration gap — and leaves full ECE to a follow-up.

- [ ] **Step 1: Extend correctness extraction to record top-5 option probabilities**

In the eval pass in `extract_activations.py`, after generating, also record the softmax probabilities of the first generated token restricted to the five option-letter token ids. Save these in `correctness.json` under each case as `option_probs: {"A": ..., "B": ..., ...}`.

```python
# Snippet for inside build_trace() eval branch
with torch.no_grad():
    out = model(ids, use_cache=False)
logits = out.logits[0, -1]
option_token_ids = {l: tokenizer(l, add_special_tokens=False).input_ids[0] for l in OPTION_LABELS}
option_logits = torch.tensor([logits[option_token_ids[l]].item() for l in OPTION_LABELS])
probs = torch.softmax(option_logits, dim=0).tolist()
option_probs = dict(zip(OPTION_LABELS, probs))
```

Then `generate` a short response as before for the text prediction and correctness check.

- [ ] **Step 2: Implement `analysis_a4_calibration_gap`**

Append to `src/probes/task_position/analysis.py`:

```python
def analysis_a4_calibration_gap(
    traces: list,
    layer: int,
    train_ids: set,
    test_ids: set,
    correctness_path: Path | str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Calibration gap binned by within_task_fraction and raw position.

    Fits a within_task_fraction probe on train traces, applies it to test cases
    at their last prompt-token position, bins cases by probe readout into
    n_bins equal-count bins, and reports (confidence, accuracy, gap) per bin.
    Also computes the same binning by raw token position for comparison.
    """
    if not Path(correctness_path).exists():
        return pd.DataFrame([{"status": "correctness file missing"}])
    with open(correctness_path) as f:
        correctness_by_trace = json.load(f)

    X_tr, y_tr = _load_layer_matrix(traces, layer, "within_task_fraction", train_ids)
    probe = RidgeProbe(alpha=1.0).fit(X_tr, y_tr)

    readouts, confidences, corrects, raw_positions = [], [], [], []
    for t in traces:
        if t["trace_id"] not in test_ids:
            continue
        cr = correctness_by_trace.get(str(t["trace_id"]), [])
        if not cr:
            continue
        act = t["activations"][layer].numpy().astype(np.float32)
        boundaries = t["case_boundaries"]
        for case in cr:
            ci = case["case_index"]
            pos = boundaries[ci + 1] - 1 if ci + 1 < len(boundaries) else act.shape[0] - 1
            readouts.append(probe.predict(act[pos : pos + 1])[0])
            pred = case["pred"]
            conf = case["option_probs"].get(pred, 0.0) if pred in case["option_probs"] else 0.0
            confidences.append(conf)
            corrects.append(1 if case["correct"] else 0)
            raw_positions.append(pos)

    df = pd.DataFrame(
        {
            "readout": readouts,
            "confidence": confidences,
            "correct": corrects,
            "raw_position": raw_positions,
        }
    )
    df["readout_bin"] = pd.qcut(df["readout"], q=n_bins, labels=False, duplicates="drop")
    df["raw_pos_bin"] = pd.qcut(df["raw_position"], q=n_bins, labels=False, duplicates="drop")

    grouped_readout = (
        df.groupby("readout_bin")
        .agg(confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
        .assign(binning="within_task_fraction_readout")
    )
    grouped_raw = (
        df.groupby("raw_pos_bin")
        .agg(confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
        .rename(columns={"raw_pos_bin": "readout_bin"})
        .assign(binning="raw_position")
    )
    combined = pd.concat([grouped_readout, grouped_raw], ignore_index=True)
    combined["gap"] = combined["confidence"] - combined["accuracy"]
    return combined
```

- [ ] **Step 3: Commit**

```bash
git add src/probes/task_position/analysis.py scripts/probes/task_position/extract_activations.py
git commit -m "Add analysis A4 (calibration gap)"
```

---

## Task 11: Analysis driver and results markdown

**Files:**
- Create: `scripts/probes/task_position/run_analyses.py`

- [ ] **Step 1: Write the driver**

Create `scripts/probes/task_position/run_analyses.py`:

```python
"""Run A1/A2/A4 analyses and write a results markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.probes.task_position.analysis import (
    analysis_a1_orthogonality,
    analysis_a2_failure_prediction,
    analysis_a4_calibration_gap,
)

TARGETS = ["task_index", "within_task_fraction", "tokens_until_boundary"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="results/probes/task_position/gemma-9b-it/activations.pt")
    p.add_argument("--correctness", default="results/probes/task_position/gemma-9b-it/correctness.json")
    p.add_argument("--split", default="results/probes/task_position/gemma-9b-it/probes/split.json")
    p.add_argument("--out", default="results/probes/task_position/2026-04-13-v1-results.md")
    return p.parse_args()


def main():
    args = parse_args()
    blob = torch.load(args.activations, weights_only=False)
    layers = blob["layers"]
    traces = blob["traces"]

    with open(args.split) as f:
        split = json.load(f)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])

    print("A1: orthogonality...")
    a1 = analysis_a1_orthogonality(traces, layers, train_ids, TARGETS)
    print(a1.to_string(index=False))

    print("\nA2: failure prediction...")
    a2 = analysis_a2_failure_prediction(traces, layers, train_ids, test_ids, args.correctness)
    print(a2.to_string(index=False))

    print("\nA4: calibration gap (layer 20)...")
    a4 = analysis_a4_calibration_gap(traces, 20, train_ids, test_ids, args.correctness)
    print(a4.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Task-Position Probes v1 Results (Gemma-9B-IT, DDXPlus)\n\n")
        f.write(f"Layers probed: {layers}\n\n")
        f.write(f"Train traces: {sorted(train_ids)}\nTest traces: {sorted(test_ids)}\n\n")
        f.write("## A1: Orthogonality between task-position and raw-position directions\n\n")
        f.write(a1.to_markdown(index=False) + "\n\n")
        f.write("## A2: Upcoming failure prediction\n\n")
        f.write("Target to beat: baseline AUC 0.67–0.68 from writeup Section 4.1.\n\n")
        f.write(a2.to_markdown(index=False) + "\n\n")
        f.write("## A4: Calibration gap vs within_task_fraction readout (L20)\n\n")
        f.write(a4.to_markdown(index=False) + "\n\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analyses**

```bash
uv run python -m scripts.probes.task_position.run_analyses
```

Expected: three tables printed, results markdown written.

- [ ] **Step 3: Review against success criteria from the spec**

Check the spec's success criteria against the results:

1. At least one probe beats both baselines on test split at best layer for at least one target?
2. A1 produces an interpretable orthogonality picture?
3. A4 produces a figure that supports or refutes the calibration-gap hypothesis?

Document the verdict in the results markdown under a new "## Verdict against v1 success criteria" section. If criterion 1 fails, stop and diagnose before proceeding to any stretch tasks.

- [ ] **Step 4: Commit**

```bash
git add scripts/probes/task_position/run_analyses.py results/probes/task_position/2026-04-13-v1-results.md
git commit -m "Add analysis driver and v1 results"
```

---

## Stretch tasks (v1 if time permits; v2 otherwise)

### Stretch task S1: MLP probe baseline

Add an `MLPProbe` class to `src/probes/task_position/probes.py` with the same interface as `RidgeProbe`. Single hidden layer width 128, ReLU, MSE loss, Adam, early stop on held-out within-train split. Run the training driver with `--probe-type mlp` and compare test metrics against ridge. **Expected outcome:** minimal lift over ridge; the point is to confirm the writeup's Section 4.1 finding extends here.

### Stretch task S2: Analysis A3 — F90871 ↔ tokens_until_boundary correlation

Requires the existing Gemma-2-9B-IT SAE hook from `scripts/context_fatigue/run_narrativeqa_gemma_sae.py` (layer 20, width 131k, `average_l0_81`). Re-run extraction with the SAE hook active, saving F90871 activation per token alongside the residual streams. Then compute per-trace sliding-window Pearson correlation between F90871 activation and the `tokens_until_boundary` probe readout; test whether correlation decays with context fill.

### Stretch task S3: Analysis A5 — base vs IT probe strength

Run `extract_activations.py` a second time with `--model google/gemma-2-9b` (base model). Retrain ridge probes. Compare (a) primary metric at best layer per target, (b) ridge solution norm, (c) effective rank via SVD of the solution direction. Write a comparison table in the results markdown.

---

## Self-review notes

**Spec coverage check:**

| Spec section | Plan task |
|---|---|
| Label schema (`task_index`, `within_task_fraction`, `tokens_until_boundary`) | Task 1, Task 2 |
| Gemma-9B-IT primary model | Task 5 |
| Layer sweep {0,10,20,30,41} | Task 5 (default `--layers`), Task 7 |
| Per-token residual extraction | Task 4, Task 5 |
| By-trace 80/20 split | Task 7 (`split_traces`) |
| Ridge probes | Task 6 |
| MLP probes | Stretch S1 |
| Raw-token-position baseline | Task 7 (inline in training driver) |
| Context-fill projection baseline | Task 8 (A1 uses raw-position probe as the proxy — documented choice) |
| A1 orthogonality | Task 8 |
| A2 upcoming-failure | Task 9 |
| A3 F90871 correlation | Stretch S2 |
| A4 calibration gap | Task 10 |
| A5 base vs IT | Stretch S3 |
| BaseSchema on dataclasses | Task 1 (`TaskPositionLabels`) |
| Auto-export in `__init__.py` | Task 1 |
| Tests on labeler | Task 2 |
| Code organization under `src/probes/task_position/` | Tasks 1, 3, 4, 6, 8, 9, 10 |
| Driver scripts in `scripts/probes/task_position/` | Tasks 5, 7, 11 |

**Placeholder scan:** Task 9 step 1 and Task 10 step 1 describe extensions to the extraction driver with code snippets but not a complete rewrite of the function. The snippets show exactly what to change and are self-contained; the engineer applies them to the `build_trace` branch added in Task 5. No TBDs or vague instructions.

**Type consistency check:** `RidgeProbe` signature (`fit`, `predict`, `score`, `direction`, `save`, `load`) is used identically in tasks 6, 7, 8, 9, 10. `TaskPositionLabels` field names match across labels.py, serialization, and every consumer. `build_trace` signature evolves across tasks (5 → 9 → 10) but each extension is explicit. `correctness.json` structure is defined once in Task 9 step 1 and referenced consistently in Tasks 9 and 10.

**Known scope tensions to flag to reviewer:**

1. Task 9 and Task 10 both mutate the extraction driver after Task 5 establishes it. An alternative structure would be to build the full extraction driver once in Task 5 including correctness and option-probs, but that front-loads complexity and makes Task 5 harder to smoke-test. The chosen ordering (MVP extraction first, then augment for A2, then augment for A4) is more TDD-shaped.
2. A1's "context-fill direction" is operationalized as the ridge solution on `raw_token_position`, not on a separately-trained context-fill probe as the spec phrased it. This is a deliberate simplification: the existing context-fill probe in the writeup was trained on last-token activations only, so it can't be reused here, and training a new one just to use as a direction is equivalent to training raw-position. Documented in Task 8's docstring.
