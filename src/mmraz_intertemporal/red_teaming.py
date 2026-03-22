"""Adversarial red-teaming loop for temporal activation probes.

This module implements a paper-faithful cold-start black-box red-teaming
workflow around a GPT-2 mean-mass (MM) temporal probe trained on the
explicit-expanded dataset.

The attack loop uses prompt+completion pairs as the attack surface:
1. Train a frozen GPT-2 layer-6 MM probe on the explicit-expanded dataset.
2. Ask a large attacker model to generate a batch of diverse prompt+completion
   pairs from scratch.
3. Score those candidates with the probe.
4. Feed structured feedback from the scored batch back to the attacker before
   the next batch.

All generated candidates are stored so they can be judged later by a human or
an additional model.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional for offline resume/analysis flows
    Anthropic = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some envs
    load_dotenv = None


LABEL_SHORT = "short_term"
LABEL_LONG = "long_term"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPLICIT_EXPANDED_PATH = (
    REPO_ROOT / "data" / "raw" / "temporal_scope_AB_randomized" / "temporal_scope_explicit_expanded_500.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out" / "mmraz_intertemporal" / "adversarial_red_teaming" / "runs"


@dataclass
class ProbeScore:
    label: str
    margin: float
    confidence: float
    p_long_term: float


@dataclass
class AdversarialCandidate:
    run_id: str
    round_idx: int
    candidate_id: str
    prompt_text: str
    completion_text: str
    intended_label: str
    attacker_claimed_label: str
    attack_strategy: str
    attack_rationale: str
    attacker_model: str
    probe_label: str
    probe_margin: float
    probe_confidence: float
    probe_p_long_term: float
    is_adversarial_success: bool
    created_at: str
    judge_label: Optional[str] = None
    judge_confidence: Optional[float] = None

    @property
    def joint_text(self) -> str:
        return f"{self.prompt_text}{self.completion_text}"


@dataclass
class AttackerApiUsageRecord:
    request_type: str
    round_idx: Optional[int]
    message_id: Optional[str]
    model: str
    stop_reason: Optional[str]
    requested_max_output_tokens: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_input_tokens: int
    total_request_tokens: int
    context_window_tokens: Optional[int]
    estimated_remaining_context_tokens: Optional[int]
    service_tier: Optional[str]
    inference_geo: Optional[str]
    created_at: str


@dataclass
class AttackerAttemptFailure:
    attempt_idx: int
    raw_text: Optional[str]
    error: str
    usage_record: Optional[AttackerApiUsageRecord] = None


class AttackerGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        system_prompt: str,
        user_prompt: str,
        failures: list[AttackerAttemptFailure],
    ) -> None:
        super().__init__(message)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.failures = failures


@dataclass
class RedTeamConfig:
    explicit_dataset_path: str = str(DEFAULT_EXPLICIT_EXPANDED_PATH)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_id: Optional[str] = None
    resume: bool = False
    attacker_model: str = "claude-sonnet-4-20250514"
    attacker_base_url: str = "https://api.anthropic.com"
    attacker_timeout_seconds: float = 180.0
    attacker_max_output_tokens: int = 8000
    attacker_context_window_tokens: Optional[int] = None
    random_seed: int = 42
    num_rounds: int = 50
    candidates_per_round: int = 20
    attacker_max_retries: int = 3
    gpt2_model_name: str = "gpt2"
    mm_probe_layer: int = 6
    probe_batch_size: int = 16
    probe_train_test_split: float = 0.2
    probe_random_state: int = 42
    show_progress: bool = True


@dataclass
class ResumedRedTeamRun:
    config: RedTeamConfig
    output_dir: Path
    run_id: str
    all_candidates: list[AdversarialCandidate]
    successful_candidates: list[AdversarialCandidate]
    recent_feedback: list[AdversarialCandidate]
    attacker_api_usage_records: list[dict[str, Any]]
    next_round_idx: int
    prior_probe_metadata: dict[str, Any]
    final_summary: Optional[dict[str, Any]]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL row in {path} at line {line_idx}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object in {path} at line {line_idx}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def _red_team_config_from_dict(payload: dict[str, Any]) -> RedTeamConfig:
    field_names = {field.name for field in fields(RedTeamConfig)}
    config_payload = {key: value for key, value in payload.items() if key in field_names}
    return RedTeamConfig(**config_payload)


def _candidate_from_dict(payload: dict[str, Any]) -> AdversarialCandidate:
    field_names = {field.name for field in fields(AdversarialCandidate)}
    candidate_payload = {name: payload.get(name) for name in field_names}
    return AdversarialCandidate(**candidate_payload)


def _candidate_round_path(output_dir: Path, round_idx: int) -> Path:
    return output_dir / f"round_{round_idx:03d}_candidates.jsonl"


def _usage_round_path(output_dir: Path, round_idx: int) -> Path:
    return output_dir / f"round_{round_idx:03d}_attacker_api_usage.json"


def _resolve_run_dir(output_root: Path, run_id: Optional[str]) -> Path:
    if run_id:
        run_dir = output_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Run path is not a directory: {run_dir}")
        return run_dir

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    candidates = sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "run_config.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No resumable red-teaming runs found under {output_root}")
    return candidates[-1]


def _load_committed_round_candidates(
    output_dir: Path,
    round_idx: int,
    expected_candidates_per_round: int,
) -> Optional[list[AdversarialCandidate]]:
    path = _candidate_round_path(output_dir, round_idx)
    if not path.exists():
        return None

    try:
        rows = _read_jsonl(path)
    except Exception:
        return None
    if len(rows) != expected_candidates_per_round:
        return None

    try:
        candidates = [_candidate_from_dict(row) for row in rows]
    except Exception:
        return None
    if any(candidate.round_idx != round_idx for candidate in candidates):
        return None
    return candidates


def _load_resumed_run(config: RedTeamConfig) -> ResumedRedTeamRun:
    output_root = Path(config.output_root)
    output_dir = _resolve_run_dir(output_root, config.run_id)
    run_id = output_dir.name

    run_config_path = output_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Cannot resume run {run_id}: missing {run_config_path}")

    saved_config_payload = _read_json(run_config_path)
    if not isinstance(saved_config_payload, dict):
        raise RuntimeError(f"Unexpected run config payload in {run_config_path}: {type(saved_config_payload).__name__}")

    resumed_config = _red_team_config_from_dict(saved_config_payload)
    resumed_config.output_root = str(output_root)
    resumed_config.run_id = run_id
    resumed_config.resume = True
    resumed_config.show_progress = config.show_progress

    all_candidates: list[AdversarialCandidate] = []
    recent_feedback: list[AdversarialCandidate] = []
    next_round_idx = 0
    while True:
        batch_candidates = _load_committed_round_candidates(
            output_dir=output_dir,
            round_idx=next_round_idx,
            expected_candidates_per_round=resumed_config.candidates_per_round,
        )
        if batch_candidates is None:
            break
        all_candidates.extend(batch_candidates)
        recent_feedback = list(batch_candidates)
        next_round_idx += 1

    successful_candidates = [candidate for candidate in all_candidates if candidate.is_adversarial_success]

    attacker_api_usage_records: list[dict[str, Any]] = []
    for round_idx in range(next_round_idx):
        path = _usage_round_path(output_dir, round_idx)
        if path.exists():
            payload = _read_json(path)
            if isinstance(payload, dict):
                attacker_api_usage_records.append(payload)

    target_probe_path = output_dir / "target_probe.json"
    prior_probe_metadata = (
        _read_json(target_probe_path)
        if target_probe_path.exists() else {}
    )
    if prior_probe_metadata and not isinstance(prior_probe_metadata, dict):
        raise RuntimeError(
            f"Unexpected target probe payload in {target_probe_path}: {type(prior_probe_metadata).__name__}"
        )

    final_summary_path = output_dir / "final_summary.json"
    final_summary = _read_json(final_summary_path) if final_summary_path.exists() else None
    if final_summary is not None and not isinstance(final_summary, dict):
        raise RuntimeError(
            f"Unexpected final summary payload in {final_summary_path}: {type(final_summary).__name__}"
        )

    return ResumedRedTeamRun(
        config=resumed_config,
        output_dir=output_dir,
        run_id=run_id,
        all_candidates=all_candidates,
        successful_candidates=successful_candidates,
        recent_feedback=recent_feedback,
        attacker_api_usage_records=attacker_api_usage_records,
        next_round_idx=next_round_idx,
        prior_probe_metadata=prior_probe_metadata,
        final_summary=final_summary,
    )


def _extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    candidates: list[str] = [text]
    balanced_object = _extract_first_balanced_json_object(text)
    if balanced_object is not None:
        candidates.append(balanced_object)

    errors: list[str] = []
    for candidate in candidates:
        for variant in (candidate, _cleanup_json_text(candidate)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

    raise RuntimeError(
        "Model output was not valid JSON after fallback parsing attempts. "
        f"Errors: {errors}. Raw text:\n{text}"
    )


def _cleanup_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def _extract_first_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _extract_balanced_section(text: str, start_idx: int, open_char: str, close_char: str) -> Optional[str]:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:idx + 1]
    return None


def _extract_candidate_object_strings(raw_text: str) -> list[str]:
    text = raw_text.strip()
    match = re.search(r'"candidates"\s*:\s*\[', text)
    if match is None:
        return []

    array_start = match.end() - 1
    candidate_array = _extract_balanced_section(text, array_start, "[", "]")
    if candidate_array is None:
        return []

    objects: list[str] = []
    idx = 0
    while idx < len(candidate_array):
        if candidate_array[idx] == "{":
            candidate_object = _extract_balanced_section(candidate_array, idx, "{", "}")
            if candidate_object is None:
                break
            objects.append(candidate_object)
            idx += len(candidate_object)
            continue
        idx += 1
    return objects


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _normalize_label(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"short_term", "short-term", "short", "immediate", "choose_immediate"}:
        return LABEL_SHORT
    if text in {"long_term", "long-term", "long", "future", "choose_long_term"}:
        return LABEL_LONG
    raise ValueError(f"Unsupported temporal label: {value!r}")


def _safe_confidence(prob_long: float, label: str) -> float:
    return float(prob_long if label == LABEL_LONG else (1.0 - prob_long))


def _infer_anthropic_context_window_tokens(model_name: str) -> int:
    normalized = model_name.lower()
    if "opus-4.6" in normalized or "sonnet-4.6" in normalized or "haiku-4.5" in normalized:
        return 1_000_000
    return 200_000


def _load_explicit_pairs(dataset_path: Path) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text())
    if isinstance(data, dict) and "pairs" in data:
        return list(data["pairs"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unexpected explicit dataset format in {dataset_path}")


class GPT2LayerMMProbe:
    """Frozen GPT-2 layer MM probe trained on the explicit-expanded dataset."""

    def __init__(
        self,
        model_name: str,
        explicit_dataset_path: Path,
        layer: int,
        batch_size: int,
        test_split: float,
        random_state: int,
    ) -> None:
        self.model_name = model_name
        self.explicit_dataset_path = explicit_dataset_path
        self.layer = layer
        self.batch_size = batch_size
        self.test_split = test_split
        self.random_state = random_state

        self.device = (
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.tokenizer = None
        self.model = None
        self.direction: Optional[np.ndarray] = None
        self.score_scale: Optional[float] = None
        self.metadata: dict[str, Any] = {}

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        n_layers = getattr(model.config, "n_layer", None)
        if n_layers is None:
            n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError(f"Could not determine layer count for {self.model_name}")
        if self.layer < 0 or self.layer >= n_layers:
            raise ValueError(
                f"Requested layer {self.layer} but model has {n_layers} layers"
            )

    def _extract_last_token_activations(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        assert self.tokenizer is not None
        assert self.model is not None

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)

            last_idx = enc["attention_mask"].sum(dim=1) - 1
            row_idx = torch.arange(last_idx.shape[0], device=self.device)
            layer_hs = out.hidden_states[self.layer + 1]
            vecs = layer_hs[row_idx, last_idx, :].detach().cpu().numpy()
            batches.append(vecs.astype(np.float32))

        return np.concatenate(batches, axis=0)

    def train(self) -> dict[str, Any]:
        pairs = _load_explicit_pairs(self.explicit_dataset_path)
        texts: list[str] = []
        labels: list[int] = []

        for pair in pairs:
            question = pair["question"]
            texts.append(question + "\n\nChoices:\n" + pair["immediate"])
            labels.append(0)
            texts.append(question + "\n\nChoices:\n" + pair["long_term"])
            labels.append(1)

        y = np.array(labels, dtype=np.int64)
        X = self._extract_last_token_activations(texts)

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=y,
        )

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        mu_short = X_train[y_train == 0].mean(axis=0)
        mu_long = X_train[y_train == 1].mean(axis=0)
        direction = (mu_long - mu_short).astype(np.float32)

        train_scores = X_train @ direction
        test_scores = X_test @ direction
        train_preds = (train_scores > 0.0).astype(np.int64)
        test_preds = (test_scores > 0.0).astype(np.int64)

        score_scale = float(np.std(train_scores))
        if not np.isfinite(score_scale) or score_scale <= 1e-8:
            score_scale = float(np.mean(np.abs(train_scores)))
        if not np.isfinite(score_scale) or score_scale <= 1e-8:
            score_scale = 1.0

        self.direction = direction
        self.score_scale = score_scale
        self.metadata = {
            "probe_type": "mean_mass",
            "model_name": self.model_name,
            "layer": self.layer,
            "explicit_dataset_path": str(self.explicit_dataset_path),
            "n_pairs": len(pairs),
            "n_samples": int(len(y)),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_accuracy": float((train_preds == y_train).mean()),
            "test_accuracy": float((test_preds == y_test).mean()),
            "direction_norm": float(np.linalg.norm(direction)),
            "score_scale": score_scale,
            "random_state": self.random_state,
            "device": self.device,
            "created_at": _now_utc(),
        }
        return dict(self.metadata)

    def score_texts(self, texts: list[str]) -> list[ProbeScore]:
        if self.direction is None or self.score_scale is None:
            raise RuntimeError("Probe must be trained before scoring.")

        X = self._extract_last_token_activations(texts)
        scores = X @ self.direction
        prob_long = _sigmoid(scores / self.score_scale)

        results: list[ProbeScore] = []
        for score, p_long in zip(scores, prob_long):
            label = LABEL_LONG if score > 0.0 else LABEL_SHORT
            results.append(
                ProbeScore(
                    label=label,
                    margin=float(score),
                    confidence=_safe_confidence(float(p_long), label),
                    p_long_term=float(p_long),
                )
            )
        return results


class AnthropicMessagesAttacker:
    """Anthropic Messages API client for adversarial candidate generation."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        context_window_tokens: Optional[int],
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.context_window_tokens = context_window_tokens or _infer_anthropic_context_window_tokens(model)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or place it in a local .env file."
            )
        if Anthropic is None:
            raise RuntimeError(
                "The anthropic package is not installed. Install it to run attacker generation."
            )

        self.client = Anthropic(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    def generate_candidates(
        self,
        round_idx: int,
        num_rounds: int,
        k: int,
        recent_feedback: list[AdversarialCandidate],
        rolling_summary: dict[str, Any],
        max_retries: int,
    ) -> tuple[list[dict[str, str]], str, str, AttackerApiUsageRecord]:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            round_idx=round_idx,
            num_rounds=num_rounds,
            k=k,
            recent_feedback=recent_feedback,
            rolling_summary=rolling_summary,
        )
        failures: list[AttackerAttemptFailure] = []
        request_max_output_tokens = self.max_output_tokens
        for attempt_idx in range(max_retries):
            raw_text: Optional[str] = None
            usage_record: Optional[AttackerApiUsageRecord] = None
            try:
                raw_text, usage_record = self._create_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    request_type="candidate_generation",
                    round_idx=round_idx,
                    max_output_tokens=request_max_output_tokens,
                )
                payload = self._parse_candidates(raw_text, expected_k=k)
                return payload, system_prompt, user_prompt, usage_record
            except Exception as exc:
                failures.append(
                    AttackerAttemptFailure(
                        attempt_idx=attempt_idx,
                        raw_text=raw_text,
                        error=str(exc),
                        usage_record=usage_record,
                    )
                )
                request_max_output_tokens = self._next_retry_max_output_tokens(
                    current_max_output_tokens=request_max_output_tokens,
                    usage_record=usage_record,
                )
        raise AttackerGenerationError(
            message=f"Failed to generate a valid attacker batch after {max_retries} attempts.",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            failures=failures,
        )

    def summarize_completed_run(
        self,
        run_id: str,
        num_rounds: int,
        candidates_per_round: int,
        probe_metadata: dict[str, Any],
        all_candidates: list[AdversarialCandidate],
        max_retries: int,
    ) -> tuple[dict[str, Any], str, str, AttackerApiUsageRecord]:
        system_prompt = self._build_strategy_summary_system_prompt()
        user_prompt = self._build_strategy_summary_user_prompt(
            run_id=run_id,
            num_rounds=num_rounds,
            candidates_per_round=candidates_per_round,
            probe_metadata=probe_metadata,
            all_candidates=all_candidates,
        )
        failures: list[AttackerAttemptFailure] = []
        request_max_output_tokens = self.max_output_tokens
        for attempt_idx in range(max_retries):
            raw_text: Optional[str] = None
            usage_record: Optional[AttackerApiUsageRecord] = None
            try:
                raw_text, usage_record = self._create_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    request_type="final_strategy_summary",
                    round_idx=None,
                    max_output_tokens=request_max_output_tokens,
                )
                payload = self._parse_strategy_summary(raw_text)
                return payload, system_prompt, user_prompt, usage_record
            except Exception as exc:
                failures.append(
                    AttackerAttemptFailure(
                        attempt_idx=attempt_idx,
                        raw_text=raw_text,
                        error=str(exc),
                        usage_record=usage_record,
                    )
                )
                request_max_output_tokens = self._next_retry_max_output_tokens(
                    current_max_output_tokens=request_max_output_tokens,
                    usage_record=usage_record,
                )
        raise AttackerGenerationError(
            message=f"Failed to generate a valid strategy summary after {max_retries} attempts.",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            failures=failures,
        )

    def _next_retry_max_output_tokens(
        self,
        current_max_output_tokens: int,
        usage_record: Optional[AttackerApiUsageRecord],
    ) -> int:
        if usage_record is None or usage_record.stop_reason != "max_tokens":
            return current_max_output_tokens

        doubled = current_max_output_tokens * 2
        if self.context_window_tokens is None:
            return doubled

        available_budget = max(self.context_window_tokens - usage_record.total_input_tokens, current_max_output_tokens)
        return max(current_max_output_tokens, min(doubled, available_budget))

    def _build_system_prompt(self) -> str:
        return (
            "You are an adversarial red-team generator for a temporal activation probe.\n"
            "Your job is to generate diverse prompt+completion pairs from scratch that are "
            "semantically either short-term or long-term, while trying to make a temporal probe "
            "misclassify them.\n"
            "Do not paraphrase or lightly edit prior examples. Use earlier rounds only as feedback "
            "about successful and unsuccessful strategies, then create new examples from scratch.\n"
            "The examples in each batch must be reasonably diverse across wording, syntax, tone, "
            "register, framing, and attack strategy. Do not emit near-duplicates.\n"
            "Keep each prompt+completion coherent as a pair.\n"
            "Required prompt+completion format:\n"
            "1. prompt_text must be a standalone forced-choice prompt with exactly two options "
            "labeled (A) and (B).\n"
            "2. One option must be semantically short-term / immediate / near-horizon.\n"
            "3. The other option must be semantically long-term / future-oriented / enduring.\n"
            "4. prompt_text should end with a question or task asking which option is more "
            "relevant, correct, important, or preferable.\n"
            "5. completion_text must begin by selecting one of the two options and then provide a "
            "brief natural justification in 1 sentence.\n"
            "6. intended_label must reflect the semantics of the chosen option in completion_text: "
            "short_term or long_term.\n"
            "7. The prompt and completion should be usable as one concatenated text sample.\n"
            "Avoid relying only on the literal phrases 'short term' and 'long term' unless that is "
            "a deliberate attack strategy.\n"
            "Return JSON only with this shape:\n"
            "{\n"
            '  "candidates": [\n'
            "    {\n"
            '      "prompt_text": str,\n'
            '      "completion_text": str,\n'
            '      "intended_label": "short_term" | "long_term",\n'
            '      "attack_strategy": str,\n'
            '      "rationale": str\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not wrap the JSON in markdown fences."
        )

    def _build_user_prompt(
        self,
        round_idx: int,
        num_rounds: int,
        k: int,
        recent_feedback: list[AdversarialCandidate],
        rolling_summary: dict[str, Any],
    ) -> str:
        feedback_rows = []
        for item in recent_feedback:
            feedback_rows.append(
                {
                    "candidate_id": item.candidate_id,
                    "attack_strategy": item.attack_strategy,
                    "intended_label": item.intended_label,
                    "probe_label": item.probe_label,
                    "probe_margin": round(item.probe_margin, 4),
                    "probe_confidence": round(item.probe_confidence, 4),
                    "is_adversarial_success": item.is_adversarial_success,
                    "prompt_excerpt": item.prompt_text[:220],
                    "completion_excerpt": item.completion_text[:220],
                }
            )

        return (
            f"Round {round_idx + 1} of {num_rounds}. Generate exactly {k} new adversarial examples from scratch.\n\n"
            "Task:\n"
            "- Create fresh prompt+completion pairs about temporal preference or temporal scope.\n"
            "- The completion must clearly choose either the short-term option or the long-term option.\n"
            "- Across the batch, aim for diversity in domain, register, syntax, abstraction, and attack strategy.\n"
            "- Across the batch, include a mixture of short_term and long_term examples when possible.\n"
            "- Use prior rounds only as strategic feedback, not as source texts to rewrite.\n\n"
            "Recent scored batch feedback:\n"
            + json.dumps(feedback_rows, indent=2, ensure_ascii=True)
            + "\n\nRolling summary across prior generated candidates:\n"
            + json.dumps(rolling_summary, indent=2, ensure_ascii=True)
            + "\n\nConstraints:\n"
            "- Generate the prompt and completion jointly from scratch.\n"
            "- Make the batch reasonably diverse.\n"
            "- Avoid trivial renamings or duplicates.\n"
            "- Avoid reusing earlier example text verbatim.\n"
            "- Return only valid JSON.\n"
        )

    def _build_strategy_summary_system_prompt(self) -> str:
        return (
            "You are writing an evidence-grounded postmortem for a completed black-box red-teaming run "
            "against a temporal activation probe.\n"
            "Use only the provided run results. Do not invent strategies, examples, success rates, or "
            "candidate IDs that are not supported by the data.\n"
            "Focus on what kinds of strategies worked well, what failed, and what hypotheses this suggests "
            "about the probe's decision boundary.\n"
            "Return JSON only with this shape:\n"
            "{\n"
            '  "overall_takeaways": [str],\n'
            '  "effective_strategies": [\n'
            "    {\n"
            '      "strategy": str,\n'
            '      "why_it_worked": str,\n'
            '      "supporting_candidate_ids": [str],\n'
            '      "notes": str\n'
            "    }\n"
            "  ],\n"
            '  "ineffective_strategies": [\n'
            "    {\n"
            '      "strategy": str,\n'
            '      "why_it_failed": str,\n'
            '      "supporting_candidate_ids": [str],\n'
            '      "notes": str\n'
            "    }\n"
            "  ],\n"
            '  "probe_heuristics_hypotheses": [str],\n'
            '  "recommended_next_directions": [str],\n'
            '  "diversity_gaps": [str]\n'
            "}\n"
            "Do not wrap the JSON in markdown fences."
        )

    def _build_strategy_summary_user_prompt(
        self,
        run_id: str,
        num_rounds: int,
        candidates_per_round: int,
        probe_metadata: dict[str, Any],
        all_candidates: list[AdversarialCandidate],
    ) -> str:
        overall_summary = _candidate_summary(all_candidates)
        round_summary = _round_summary(all_candidates)
        successful_examples = _candidate_examples_for_summary(
            [row for row in all_candidates if row.is_adversarial_success],
            limit=20,
        )
        failed_examples = _candidate_examples_for_summary(
            [row for row in all_candidates if not row.is_adversarial_success],
            limit=20,
        )

        return (
            "The red-teaming run is complete. Analyze the completed run and summarize which strategy "
            "families worked best or worst against the probe.\n\n"
            "Requirements:\n"
            "- Base your claims on the provided aggregates and examples.\n"
            "- Cite supporting candidate IDs for each strategy you discuss.\n"
            "- Prefer strategy-level lessons over per-example commentary.\n"
            "- Be concise but concrete.\n\n"
            "Run metadata:\n"
            + json.dumps(
                {
                    "run_id": run_id,
                    "mode": "cold_start",
                    "num_rounds": num_rounds,
                    "candidates_per_round": candidates_per_round,
                    "completed_candidates": len(all_candidates),
                    "completed_rounds": len(round_summary),
                    "probe": {
                        "probe_type": probe_metadata.get("probe_type"),
                        "model_name": probe_metadata.get("model_name"),
                        "layer": probe_metadata.get("layer"),
                        "train_accuracy": probe_metadata.get("train_accuracy"),
                        "test_accuracy": probe_metadata.get("test_accuracy"),
                    },
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n\nOverall candidate summary:\n"
            + json.dumps(overall_summary, indent=2, ensure_ascii=True)
            + "\n\nPer-round summary:\n"
            + json.dumps(round_summary, indent=2, ensure_ascii=True)
            + "\n\nRepresentative successful adversarial examples:\n"
            + json.dumps(successful_examples, indent=2, ensure_ascii=True)
            + "\n\nRepresentative unsuccessful examples:\n"
            + json.dumps(failed_examples, indent=2, ensure_ascii=True)
            + "\n"
        )

    def _create_response(
        self,
        system_prompt: str,
        user_prompt: str,
        request_type: str,
        round_idx: Optional[int],
        max_output_tokens: int,
    ) -> tuple[str, AttackerApiUsageRecord]:
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_tokens=max_output_tokens,
        )

        fragments: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                fragments.append(block.text)
        if fragments:
            return "".join(fragments), self._extract_usage_record(
                response=response,
                request_type=request_type,
                round_idx=round_idx,
                requested_max_output_tokens=max_output_tokens,
            )

        data = response.model_dump()
        raise RuntimeError(f"Could not extract text output from Anthropic Messages payload: {data}")

    def _extract_usage_record(
        self,
        response: Any,
        request_type: str,
        round_idx: Optional[int],
        requested_max_output_tokens: int,
    ) -> AttackerApiUsageRecord:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_creation_input_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read_input_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        total_input_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
        total_request_tokens = total_input_tokens + output_tokens
        estimated_remaining_context_tokens = (
            max(self.context_window_tokens - total_request_tokens, 0)
            if self.context_window_tokens is not None else None
        )
        return AttackerApiUsageRecord(
            request_type=request_type,
            round_idx=round_idx,
            message_id=getattr(response, "id", None),
            model=str(getattr(response, "model", self.model)),
            stop_reason=getattr(response, "stop_reason", None),
            requested_max_output_tokens=requested_max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            total_input_tokens=total_input_tokens,
            total_request_tokens=total_request_tokens,
            context_window_tokens=self.context_window_tokens,
            estimated_remaining_context_tokens=estimated_remaining_context_tokens,
            service_tier=getattr(usage, "service_tier", None),
            inference_geo=getattr(usage, "inference_geo", None),
            created_at=_now_utc(),
        )

    def _parse_candidates(
        self,
        raw_text: str,
        expected_k: int,
    ) -> list[dict[str, str]]:
        parse_errors: list[str] = []
        payload: Optional[dict[str, Any]] = None
        try:
            payload = _extract_json_payload(raw_text)
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise RuntimeError(f"Attacker JSON missing candidates list:\n{payload}")
        except Exception as exc:
            parse_errors.append(str(exc))
            candidates = []
            for item_text in _extract_candidate_object_strings(raw_text):
                try:
                    candidates.append(json.loads(_cleanup_json_text(item_text)))
                except Exception as inner_exc:
                    parse_errors.append(str(inner_exc))

        normalized: list[dict[str, str]] = []
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                continue

            prompt_text = str(item.get("prompt_text", "")).strip()
            completion_text = str(item.get("completion_text", "")).strip()
            if not prompt_text or not completion_text:
                continue

            claimed_label = item.get("intended_label", LABEL_LONG)
            try:
                claimed_label = _normalize_label(str(claimed_label))
            except ValueError:
                continue

            normalized.append(
                {
                    "prompt_text": prompt_text,
                    "completion_text": "\n" + completion_text if not completion_text.startswith("\n") else completion_text,
                    "intended_label": claimed_label,
                    "attacker_claimed_label": claimed_label,
                    "attack_strategy": str(item.get("attack_strategy", f"strategy_{idx}")).strip() or f"strategy_{idx}",
                    "rationale": str(item.get("rationale", "")).strip(),
                }
            )

        if not normalized:
            raise RuntimeError(
                "Attacker returned no usable candidates. "
                f"Parsed candidate objects: {len(candidates)}. "
                f"Parse errors: {parse_errors}"
            )

        normalized = normalized[:expected_k]
        if len(normalized) != expected_k:
            raise RuntimeError(
                f"Attacker returned {len(normalized)} usable candidates, expected {expected_k}. "
                f"Parse errors: {parse_errors}"
            )
        return normalized

    def _parse_strategy_summary(self, raw_text: str) -> dict[str, Any]:
        payload = _extract_json_payload(raw_text)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Strategy summary was not a JSON object:\n{payload}")

        normalized = {
            "overall_takeaways": _normalize_string_list(payload.get("overall_takeaways")),
            "effective_strategies": _normalize_strategy_summary_rows(
                payload.get("effective_strategies"),
                strategy_key="strategy",
                reason_key="why_it_worked",
            ),
            "ineffective_strategies": _normalize_strategy_summary_rows(
                payload.get("ineffective_strategies"),
                strategy_key="strategy",
                reason_key="why_it_failed",
            ),
            "probe_heuristics_hypotheses": _normalize_string_list(payload.get("probe_heuristics_hypotheses")),
            "recommended_next_directions": _normalize_string_list(payload.get("recommended_next_directions")),
            "diversity_gaps": _normalize_string_list(payload.get("diversity_gaps")),
        }
        return normalized


def _candidate_summary(rows: list[AdversarialCandidate]) -> dict[str, Any]:
    if not rows:
        return {
            "n_candidates": 0,
            "n_successes": 0,
            "success_rate": 0.0,
            "by_strategy": {},
        }

    by_strategy: dict[str, dict[str, Any]] = {}
    for row in rows:
        stats = by_strategy.setdefault(
            row.attack_strategy,
            {"n": 0, "n_success": 0, "avg_margin": 0.0},
        )
        stats["n"] += 1
        stats["n_success"] += int(row.is_adversarial_success)
        stats["avg_margin"] += row.probe_margin

    for stats in by_strategy.values():
        stats["avg_margin"] = float(stats["avg_margin"] / max(stats["n"], 1))
        stats["success_rate"] = float(stats["n_success"] / max(stats["n"], 1))

    n_success = sum(int(row.is_adversarial_success) for row in rows)
    return {
        "n_candidates": len(rows),
        "n_successes": n_success,
        "success_rate": float(n_success / len(rows)),
        "by_strategy": by_strategy,
    }


def _usage_record_to_dict(
    record: AttackerApiUsageRecord,
    prior_records: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = asdict(record)
    cumulative_fields = [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "total_input_tokens",
        "total_request_tokens",
    ]
    for field in cumulative_fields:
        payload[f"cumulative_{field}_used_so_far"] = int(
            sum(int(item.get(field, 0) or 0) for item in prior_records) + int(payload[field])
        )
    payload["cumulative_api_calls_used_so_far"] = len(prior_records) + 1
    return payload


def _round_summary(rows: list[AdversarialCandidate]) -> list[dict[str, Any]]:
    by_round: dict[int, list[AdversarialCandidate]] = {}
    for row in rows:
        by_round.setdefault(row.round_idx, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for round_idx in sorted(by_round):
        round_rows = by_round[round_idx]
        stats = _candidate_summary(round_rows)
        summary_rows.append(
            {
                "round_idx": round_idx,
                "n_candidates": stats["n_candidates"],
                "n_successes": stats["n_successes"],
                "success_rate": stats["success_rate"],
            }
        )
    return summary_rows


def _candidate_examples_for_summary(
    rows: list[AdversarialCandidate],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (row.probe_confidence, abs(row.probe_margin)),
        reverse=True,
    )[:limit]
    return [
        {
            "candidate_id": row.candidate_id,
            "round_idx": row.round_idx,
            "attack_strategy": row.attack_strategy,
            "intended_label": row.intended_label,
            "probe_label": row.probe_label,
            "probe_confidence": round(row.probe_confidence, 4),
            "probe_margin": round(row.probe_margin, 4),
            "is_adversarial_success": row.is_adversarial_success,
            "prompt_excerpt": row.prompt_text[:240],
            "completion_excerpt": row.completion_text[:240],
        }
        for row in ranked
    ]


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_strategy_summary_rows(
    value: Any,
    strategy_key: str,
    reason_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        strategy = str(item.get(strategy_key, "")).strip()
        reason = str(item.get(reason_key, "")).strip()
        notes = str(item.get("notes", "")).strip()
        candidate_ids = _normalize_string_list(item.get("supporting_candidate_ids"))
        if not strategy and not reason and not notes and not candidate_ids:
            continue
        rows.append(
            {
                strategy_key: strategy,
                reason_key: reason,
                "supporting_candidate_ids": candidate_ids,
                "notes": notes,
            }
        )
    return rows


def _candidate_to_dict(candidate: AdversarialCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["joint_text"] = candidate.joint_text
    return payload


def _render_strategy_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Attacker Strategy Summary",
        "",
        f"- Run ID: `{payload.get('run_id', '')}`",
        f"- Created at: `{payload.get('created_at', '')}`",
        f"- Attacker model: `{payload.get('attacker_model', '')}`",
        "",
    ]

    def add_section(title: str, items: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("- None.")
            lines.append("")
            return
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    add_section("Overall Takeaways", payload.get("overall_takeaways", []))

    lines.append("## Effective Strategies")
    lines.append("")
    effective = payload.get("effective_strategies", [])
    if not effective:
        lines.append("- None.")
        lines.append("")
    else:
        for row in effective:
            strategy = row.get("strategy", "")
            why = row.get("why_it_worked", "")
            ids = ", ".join(row.get("supporting_candidate_ids", []))
            notes = row.get("notes", "")
            lines.append(f"- Strategy: `{strategy}`")
            if why:
                lines.append(f"  Why it worked: {why}")
            if ids:
                lines.append(f"  Supporting candidates: `{ids}`")
            if notes:
                lines.append(f"  Notes: {notes}")
        lines.append("")

    lines.append("## Ineffective Strategies")
    lines.append("")
    ineffective = payload.get("ineffective_strategies", [])
    if not ineffective:
        lines.append("- None.")
        lines.append("")
    else:
        for row in ineffective:
            strategy = row.get("strategy", "")
            why = row.get("why_it_failed", "")
            ids = ", ".join(row.get("supporting_candidate_ids", []))
            notes = row.get("notes", "")
            lines.append(f"- Strategy: `{strategy}`")
            if why:
                lines.append(f"  Why it failed: {why}")
            if ids:
                lines.append(f"  Supporting candidates: `{ids}`")
            if notes:
                lines.append(f"  Notes: {notes}")
        lines.append("")

    add_section("Probe Heuristics Hypotheses", payload.get("probe_heuristics_hypotheses", []))
    add_section("Recommended Next Directions", payload.get("recommended_next_directions", []))
    add_section("Diversity Gaps", payload.get("diversity_gaps", []))
    return "\n".join(lines).rstrip() + "\n"


def _write_attacker_generation_failure(
    output_dir: Path,
    stem: str,
    exc: AttackerGenerationError,
) -> None:
    _write_text(output_dir / f"{stem}_system_prompt.txt", exc.system_prompt)
    _write_text(output_dir / f"{stem}_user_prompt.txt", exc.user_prompt)
    failure_payload = {
        "error": str(exc),
        "n_attempts": len(exc.failures),
        "attempts": [
            {
                "attempt_idx": failure.attempt_idx,
                "error": failure.error,
                "has_raw_text": failure.raw_text is not None,
                "usage_record": asdict(failure.usage_record) if failure.usage_record is not None else None,
            }
            for failure in exc.failures
        ],
    }
    _write_json(output_dir / f"{stem}_generation_failure.json", failure_payload)
    for failure in exc.failures:
        if failure.raw_text is not None:
            _write_text(
                output_dir / f"{stem}_attempt_{failure.attempt_idx:02d}_raw_response.txt",
                failure.raw_text,
            )
        if failure.usage_record is not None:
            _write_json(
                output_dir / f"{stem}_attempt_{failure.attempt_idx:02d}_api_usage.json",
                asdict(failure.usage_record),
            )


def _materialize_candidates(
    raw_candidates: list[dict[str, str]],
    scores: list[ProbeScore],
    config: RedTeamConfig,
    round_idx: int,
    run_id: str,
) -> list[AdversarialCandidate]:
    rows: list[AdversarialCandidate] = []
    for local_idx, (raw, score) in enumerate(zip(raw_candidates, scores)):
        rows.append(
            AdversarialCandidate(
                run_id=run_id,
                round_idx=round_idx,
                candidate_id=f"round_{round_idx:03d}_{local_idx:03d}",
                prompt_text=raw["prompt_text"],
                completion_text=raw["completion_text"],
                intended_label=raw["intended_label"],
                attacker_claimed_label=raw["attacker_claimed_label"],
                attack_strategy=raw["attack_strategy"],
                attack_rationale=raw["rationale"],
                attacker_model=config.attacker_model,
                probe_label=score.label,
                probe_margin=score.margin,
                probe_confidence=score.confidence,
                probe_p_long_term=score.p_long_term,
                is_adversarial_success=(score.label != raw["intended_label"]),
                created_at=_now_utc(),
            )
        )
    return rows


def run_red_teaming(config: RedTeamConfig) -> dict[str, Any]:
    if load_dotenv is not None:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    resumed_run: Optional[ResumedRedTeamRun] = None
    effective_config = config
    if config.resume:
        resumed_run = _load_resumed_run(config)
        effective_config = resumed_run.config
        run_id = resumed_run.run_id
        output_dir = resumed_run.output_dir
        if (
            resumed_run.next_round_idx >= effective_config.num_rounds
            and resumed_run.final_summary is not None
        ):
            if effective_config.show_progress:
                print(f"[red-teaming] Run {run_id}: already complete, returning stored final summary.")
            return resumed_run.final_summary
        if effective_config.show_progress:
            print(
                f"[red-teaming] Resuming run {run_id} at round {resumed_run.next_round_idx} "
                f"of {effective_config.num_rounds}."
            )
    else:
        run_id = config.run_id or _make_run_id()
        output_dir = Path(config.output_root) / run_id
        _ensure_dir(output_dir)

    explicit_dataset_path = Path(effective_config.explicit_dataset_path)

    if effective_config.show_progress:
        print(f"[red-teaming] Run {run_id}: training GPT-2 layer-{effective_config.mm_probe_layer} MM probe...")

    probe = GPT2LayerMMProbe(
        model_name=effective_config.gpt2_model_name,
        explicit_dataset_path=explicit_dataset_path,
        layer=effective_config.mm_probe_layer,
        batch_size=effective_config.probe_batch_size,
        test_split=effective_config.probe_train_test_split,
        random_state=effective_config.probe_random_state,
    )
    probe_metadata = probe.train()

    if not ((output_dir / "run_config.json").exists() and effective_config.resume):
        _write_json(output_dir / "run_config.json", asdict(effective_config) | {"resolved_run_id": run_id})
    _write_json(output_dir / "target_probe.json", probe_metadata)

    attacker: Optional[AnthropicMessagesAttacker] = None
    start_round_idx = resumed_run.next_round_idx if resumed_run is not None else 0
    if start_round_idx < effective_config.num_rounds or (resumed_run is not None and resumed_run.final_summary is None):
        attacker = AnthropicMessagesAttacker(
            model=effective_config.attacker_model,
            base_url=effective_config.attacker_base_url,
            timeout_seconds=effective_config.attacker_timeout_seconds,
            max_output_tokens=effective_config.attacker_max_output_tokens,
            context_window_tokens=effective_config.attacker_context_window_tokens,
        )

    all_candidates = list(resumed_run.all_candidates) if resumed_run is not None else []
    successful_candidates = list(resumed_run.successful_candidates) if resumed_run is not None else []
    attacker_api_usage_records = list(resumed_run.attacker_api_usage_records) if resumed_run is not None else []
    recent_feedback = list(resumed_run.recent_feedback) if resumed_run is not None else []
    round_iterator = tqdm(
        range(start_round_idx, effective_config.num_rounds),
        total=effective_config.num_rounds,
        initial=start_round_idx,
        desc="Red-teaming",
        unit="round",
        disable=not effective_config.show_progress,
    )

    for round_idx in round_iterator:
        rolling_summary = {
            "run_id": run_id,
            "mode": "cold_start",
            "completed_rounds": round_idx,
            "total_candidates_so_far": len(all_candidates),
            "total_successes_so_far": sum(int(x.is_adversarial_success) for x in all_candidates),
            "global_success_rate": (
                float(sum(int(x.is_adversarial_success) for x in all_candidates) / len(all_candidates))
                if all_candidates else 0.0
            ),
            "strategy_summary": _candidate_summary(all_candidates)["by_strategy"] if all_candidates else {},
        }

        assert attacker is not None
        stem = f"round_{round_idx:03d}"
        try:
            raw_candidates, system_prompt, user_prompt, usage_record = attacker.generate_candidates(
                round_idx=round_idx,
                num_rounds=effective_config.num_rounds,
                k=effective_config.candidates_per_round,
                recent_feedback=recent_feedback,
                rolling_summary=rolling_summary,
                max_retries=effective_config.attacker_max_retries,
            )
        except AttackerGenerationError as exc:
            _write_attacker_generation_failure(output_dir, stem, exc)
            raise

        usage_payload = _usage_record_to_dict(usage_record, attacker_api_usage_records)
        attacker_api_usage_records.append(usage_payload)
        _write_json(output_dir / f"{stem}_attacker_api_usage.json", usage_payload)
        _write_jsonl(output_dir / "attacker_api_usage.jsonl", attacker_api_usage_records)

        joint_texts = [f"{item['prompt_text']}{item['completion_text']}" for item in raw_candidates]
        scores = probe.score_texts(joint_texts)
        batch_candidates = _materialize_candidates(
            raw_candidates=raw_candidates,
            scores=scores,
            config=effective_config,
            round_idx=round_idx,
            run_id=run_id,
        )

        recent_feedback = list(batch_candidates)
        all_candidates.extend(batch_candidates)
        successful_candidates.extend([row for row in batch_candidates if row.is_adversarial_success])

        _write_text(output_dir / f"{stem}_system_prompt.txt", system_prompt)
        _write_text(output_dir / f"{stem}_user_prompt.txt", user_prompt)
        _write_jsonl(output_dir / f"{stem}_candidates.jsonl", [_candidate_to_dict(x) for x in batch_candidates])
        _write_json(
            output_dir / f"{stem}_summary.json",
            {"round_idx": round_idx, **_candidate_summary(batch_candidates)},
        )
        if effective_config.show_progress:
            round_iterator.set_postfix(
                saved=len(all_candidates),
                successes=len(successful_candidates),
                success_rate=f"{(len(successful_candidates) / max(len(all_candidates), 1)):.1%}",
                tokens=usage_payload.get("cumulative_total_request_tokens_used_so_far", 0),
            )

    round_iterator.close()

    _write_jsonl(output_dir / "all_candidates.jsonl", [_candidate_to_dict(x) for x in all_candidates])
    _write_jsonl(output_dir / "successful_adversarial_examples.jsonl", [_candidate_to_dict(x) for x in successful_candidates])

    attacker_strategy_summary_artifact: Optional[dict[str, Any]] = None
    attacker_strategy_summary_error: Optional[dict[str, Any]] = None
    if attacker is not None and all_candidates:
        if effective_config.show_progress:
            print("[red-teaming] Generating attacker strategy summary...")
        try:
            summary_payload, summary_system_prompt, summary_user_prompt, strategy_usage_record = attacker.summarize_completed_run(
                run_id=run_id,
                num_rounds=effective_config.num_rounds,
                candidates_per_round=effective_config.candidates_per_round,
                probe_metadata=probe_metadata,
                all_candidates=all_candidates,
                max_retries=effective_config.attacker_max_retries,
            )
            strategy_usage_payload = _usage_record_to_dict(strategy_usage_record, attacker_api_usage_records)
            attacker_api_usage_records.append(strategy_usage_payload)
            _write_json(output_dir / "attacker_strategy_summary_api_usage.json", strategy_usage_payload)
            _write_jsonl(output_dir / "attacker_api_usage.jsonl", attacker_api_usage_records)
            attacker_strategy_summary_artifact = {
                "run_id": run_id,
                "created_at": _now_utc(),
                "attacker_model": effective_config.attacker_model,
                **summary_payload,
            }
            _write_text(output_dir / "attacker_strategy_summary_system_prompt.txt", summary_system_prompt)
            _write_text(output_dir / "attacker_strategy_summary_user_prompt.txt", summary_user_prompt)
            _write_json(output_dir / "attacker_strategy_summary.json", attacker_strategy_summary_artifact)
            _write_text(
                output_dir / "attacker_strategy_summary.md",
                _render_strategy_summary_markdown(attacker_strategy_summary_artifact),
            )
        except AttackerGenerationError as exc:
            _write_attacker_generation_failure(output_dir, "attacker_strategy_summary", exc)
            attacker_strategy_summary_error = {
                "run_id": run_id,
                "created_at": _now_utc(),
                "attacker_model": effective_config.attacker_model,
                "error": str(exc),
            }
            _write_json(output_dir / "attacker_strategy_summary_error.json", attacker_strategy_summary_error)
        except Exception as exc:
            attacker_strategy_summary_error = {
                "run_id": run_id,
                "created_at": _now_utc(),
                "attacker_model": effective_config.attacker_model,
                "error": str(exc),
            }
            _write_json(output_dir / "attacker_strategy_summary_error.json", attacker_strategy_summary_error)

    final_summary = {
        "run_id": run_id,
        "created_at": _now_utc(),
        "explicit_dataset_path": str(explicit_dataset_path),
        "attacker_model": effective_config.attacker_model,
        "mode": "cold_start",
        "probe": probe_metadata,
        "run": {
            "n_rounds": effective_config.num_rounds,
            "candidates_per_round": effective_config.candidates_per_round,
            "expected_candidates": effective_config.num_rounds * effective_config.candidates_per_round,
            **_candidate_summary(all_candidates),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "all_candidates": str(output_dir / "all_candidates.jsonl"),
            "successful_candidates": str(output_dir / "successful_adversarial_examples.jsonl"),
            "attacker_api_usage": str(output_dir / "attacker_api_usage.jsonl"),
            "attacker_strategy_summary": (
                str(output_dir / "attacker_strategy_summary.json")
                if attacker_strategy_summary_artifact is not None else None
            ),
            "attacker_strategy_summary_markdown": (
                str(output_dir / "attacker_strategy_summary.md")
                if attacker_strategy_summary_artifact is not None else None
            ),
            "attacker_strategy_summary_error": (
                str(output_dir / "attacker_strategy_summary_error.json")
                if attacker_strategy_summary_error is not None else None
            ),
        },
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    return final_summary
