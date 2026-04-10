#!/usr/bin/env python3
"""Run the current multi-turn red-teaming loop against a Qwen3-32B question-only probe.

This keeps the same attacker/judge methodology and artifact layout as
`scripts/mmraz_intertemporal/run_adversarial_probe_red_team.py`, but swaps in a
probe scorer backed by the saved Qwen3-32B question-only probe artifacts.

The top-level `DEFAULT_CONFIG` is intended to be easy to edit before a run,
especially `train_regime` and `target_layer`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mmraz_intertemporal.red_teaming as rt


REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION = 1


DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "explicit_path": None,
    "implicit_path": None,
    "artifact_search_roots": None,
    "artifact_path": None,
    "metadata_path": None,
    "train_regime": "explicit_train_only",
    "feature_name": "mean_answer_tokens",
    "vector_key": "mm_probe_vectors",
    "target_layer": 40,
    "use_chat_template": True,
    "disable_thinking_trace": True,
    "probe_batch_size": 1,
    "require_cuda": True,
    "output_root": "out/qwen3_32b/adversarial_red_teaming/runs",
    "run_id": None,
    "attacker_model": "claude-sonnet-4-20250514",
    "attacker_base_url": "https://api.anthropic.com",
    "attacker_timeout_seconds": 180.0,
    "attacker_max_output_tokens": 8000,
    "attacker_context_window_tokens": None,
    "judge_model": "claude-sonnet-4-20250514",
    "judge_base_url": "https://api.anthropic.com",
    "judge_timeout_seconds": 180.0,
    "judge_max_output_tokens": 4000,
    "judge_context_window_tokens": None,
    "num_rounds": 20,
    "candidates_per_round": 10,
    "attacker_max_retries": 3,
    "judge_max_retries": 3,
    "random_seed": 42,
    "show_progress": True,
}


@dataclass
class Qwen32BRedTeamConfig:
    model_name: str = DEFAULT_CONFIG["model_name"]
    explicit_path: Optional[str] = DEFAULT_CONFIG["explicit_path"]
    implicit_path: Optional[str] = DEFAULT_CONFIG["implicit_path"]
    artifact_search_roots: Optional[list[str]] = None
    artifact_path: Optional[str] = DEFAULT_CONFIG["artifact_path"]
    metadata_path: Optional[str] = DEFAULT_CONFIG["metadata_path"]
    train_regime: str = DEFAULT_CONFIG["train_regime"]
    feature_name: str = DEFAULT_CONFIG["feature_name"]
    vector_key: str = DEFAULT_CONFIG["vector_key"]
    target_layer: int = DEFAULT_CONFIG["target_layer"]
    use_chat_template: bool = DEFAULT_CONFIG["use_chat_template"]
    disable_thinking_trace: bool = DEFAULT_CONFIG["disable_thinking_trace"]
    probe_batch_size: int = DEFAULT_CONFIG["probe_batch_size"]
    require_cuda: bool = DEFAULT_CONFIG["require_cuda"]
    output_root: str = DEFAULT_CONFIG["output_root"]
    run_id: Optional[str] = DEFAULT_CONFIG["run_id"]
    attacker_model: str = DEFAULT_CONFIG["attacker_model"]
    attacker_base_url: str = DEFAULT_CONFIG["attacker_base_url"]
    attacker_timeout_seconds: float = DEFAULT_CONFIG["attacker_timeout_seconds"]
    attacker_max_output_tokens: int = DEFAULT_CONFIG["attacker_max_output_tokens"]
    attacker_context_window_tokens: Optional[int] = DEFAULT_CONFIG["attacker_context_window_tokens"]
    judge_model: str = DEFAULT_CONFIG["judge_model"]
    judge_base_url: str = DEFAULT_CONFIG["judge_base_url"]
    judge_timeout_seconds: float = DEFAULT_CONFIG["judge_timeout_seconds"]
    judge_max_output_tokens: int = DEFAULT_CONFIG["judge_max_output_tokens"]
    judge_context_window_tokens: Optional[int] = DEFAULT_CONFIG["judge_context_window_tokens"]
    num_rounds: int = DEFAULT_CONFIG["num_rounds"]
    candidates_per_round: int = DEFAULT_CONFIG["candidates_per_round"]
    attacker_max_retries: int = DEFAULT_CONFIG["attacker_max_retries"]
    judge_max_retries: int = DEFAULT_CONFIG["judge_max_retries"]
    random_seed: int = DEFAULT_CONFIG["random_seed"]
    show_progress: bool = DEFAULT_CONFIG["show_progress"]


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from current working directory.")


def pick_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + str([str(path) for path in paths]))


def load_pairs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "pairs" in data:
        return data.get("metadata", {}), data["pairs"]
    return {}, data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_cuda_runtime(*, require_cuda: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this run, but torch.cuda.is_available() is False. "
            "Refusing to fall back to CPU/MPS."
        )


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def decode_str_array(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def resolve_metadata_path_for_artifact(artifact_path: Path, metadata_candidates):
    candidate = artifact_path.with_name(
        artifact_path.name.replace("_probe_artifacts_", "_probe_metadata_").replace(".npz", ".json")
    )
    if candidate.exists():
        return candidate
    for metadata_path in metadata_candidates:
        if metadata_path.stem.replace("_probe_metadata_", "_probe_artifacts_") == artifact_path.stem:
            return metadata_path
    return None


def load_probe_metadata(metadata_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if metadata_path is None or not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_is_compatible(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return False
    if int(metadata.get("artifact_format_version", 0)) < REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION:
        return False
    if metadata.get("prompt_family") != "question_only_teacher_forced_answers":
        return False
    if metadata.get("explicit_split_granularity") != "question":
        return False
    if metadata.get("implicit_split_granularity") != "question":
        return False
    if not bool(metadata.get("probe_prompt_use_chat_template", metadata.get("use_chat_template", False))):
        return False
    if not bool(metadata.get("probe_prompt_disable_thinking_trace", metadata.get("disable_thinking_trace", False))):
        return False
    return True


def artifact_contains_train_regime(artifact_path: Path, required_train_regime: Optional[str]) -> bool:
    if required_train_regime is None:
        return True
    try:
        with np.load(artifact_path) as bundle:
            train_regimes = decode_str_array(bundle["train_regimes"])
    except Exception:
        return False
    return required_train_regime in train_regimes


def locate_latest_probe_artifacts(
    search_roots,
    artifact_path_override: Optional[str] = None,
    metadata_path_override: Optional[str] = None,
    required_train_regime: Optional[str] = None,
):
    if artifact_path_override is not None:
        artifact_path = Path(artifact_path_override).expanduser().resolve()
        if metadata_path_override is not None:
            metadata_path = Path(metadata_path_override).expanduser().resolve()
        else:
            metadata_path = resolve_metadata_path_for_artifact(artifact_path, [])
        if not artifact_contains_train_regime(artifact_path, required_train_regime):
            raise ValueError(
                f"Artifact {artifact_path} does not contain requested train_regime={required_train_regime!r}."
            )
        return artifact_path, metadata_path

    artifact_candidates = []
    metadata_candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        artifact_candidates.extend(sorted(root.rglob("qwen3_32b_question_only_probe_artifacts_*.npz")))
        metadata_candidates.extend(sorted(root.rglob("qwen3_32b_question_only_probe_metadata_*.json")))

    artifact_candidates = unique_paths(artifact_candidates)
    metadata_candidates = unique_paths(metadata_candidates)

    if not artifact_candidates:
        roots_text = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            "Could not find any Qwen3-32B probe-artifact bundles under: "
            f"{roots_text}. Set artifact_path explicitly if needed."
        )

    compatible_candidates = []
    for artifact_path in reversed(artifact_candidates):
        metadata_path = resolve_metadata_path_for_artifact(artifact_path, metadata_candidates)
        metadata = load_probe_metadata(metadata_path)
        if metadata_is_compatible(metadata) and artifact_contains_train_regime(artifact_path, required_train_regime):
            compatible_candidates.append((artifact_path, metadata_path))
    if not compatible_candidates:
        roots_text = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            "Found Qwen3-32B probe artifacts, but none were compatible with the current question-only + question-split + chat-template format "
            f"(required artifact_format_version >= {REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION}). Search roots: {roots_text}"
        )
    return compatible_candidates[0]


def extract_option_letter(option_text: str) -> str:
    match = re.search(r"\(([ABab])\)", option_text or "")
    if match is None:
        raise ValueError(f"Could not parse option letter from: {option_text!r}")
    return match.group(1).upper()


def strip_option_label(option_text: str) -> str:
    return re.sub(r"^\s*\([ABab]\)\s*", "", option_text or "").strip()


def build_question_only_probe_prompt(question_text: str) -> str:
    question_text = (question_text or "").strip()
    if not question_text:
        raise ValueError("Question text is empty; cannot build question-only probe prompt.")
    return question_text


def get_pair_option_payload(pair: dict[str, Any]) -> dict[str, str]:
    immediate_letter = extract_option_letter(pair["immediate"])
    long_term_letter = extract_option_letter(pair["long_term"])
    if not immediate_letter or not long_term_letter or immediate_letter == long_term_letter:
        raise ValueError(f"Could not resolve A/B option ordering for pair: {pair!r}")
    return {
        "candidate_immediate_text": strip_option_label(pair["immediate"]),
        "candidate_long_term_text": strip_option_label(pair["long_term"]),
    }


def build_teacher_forced_examples_from_pairs(pairs, strip_option_letters: bool = True):
    examples = []
    labels = []
    for question_idx, pair in enumerate(pairs):
        option_payload = get_pair_option_payload(pair)
        prompt = build_question_only_probe_prompt(pair["question"])
        immediate_continuation = option_payload["candidate_immediate_text"]
        long_term_continuation = option_payload["candidate_long_term_text"]
        if not strip_option_letters:
            immediate_continuation = pair["immediate"]
            long_term_continuation = pair["long_term"]

        examples.append(
            {
                "prompt": prompt,
                "continuation": immediate_continuation,
                "label": 0,
                "question_idx": int(question_idx),
            }
        )
        labels.append(0)
        examples.append(
            {
                "prompt": prompt,
                "continuation": long_term_continuation,
                "label": 1,
                "question_idx": int(question_idx),
            }
        )
        labels.append(1)

    return examples, np.asarray(labels, dtype=np.int64)


def question_indices_to_example_indices(question_idx: np.ndarray) -> np.ndarray:
    if len(question_idx) == 0:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate([2 * question_idx, 2 * question_idx + 1]).astype(np.int64))


def format_prompt_for_model(tokenizer, user_prompt: str, *, use_chat_template: bool, disable_thinking_trace: bool) -> str:
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not expose apply_chat_template, but use_chat_template=True was requested.")
        messages = [{"role": "user", "content": user_prompt}]
        if disable_thinking_trace:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return templated + "<think>\n</think>\n\n"
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def assert_model_on_cuda(model) -> torch.device:
    model_device = get_model_device(model)
    if model_device.type != "cuda":
        raise RuntimeError(
            f"Model is not on CUDA. Expected CUDA execution, but model device is {model_device}."
        )
    return model_device


def move_batch_to_model_device(model, batch):
    model_device = assert_model_on_cuda(model)
    return {key: value.to(model_device) for key, value in batch.items()}


def _extract_api_key_from_text(text: str) -> Optional[str]:
    for pattern in (
        r'ANTHROPIC_API_KEY\s*=\s*"([^"]+)"',
        r"ANTHROPIC_API_KEY\s*=\s*'([^']+)'",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _load_api_key_from_vscode_history() -> Optional[str]:
    history_roots = [
        Path.home() / "Library" / "Application Support" / "Code" / "User" / "History",
        Path.home() / ".config" / "Code" / "User" / "History",
    ]
    for history_root in history_roots:
        if not history_root.exists():
            continue
        for path in history_root.rglob("*"):
            if path.suffix not in {".ipynb", ".py"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            api_key = _extract_api_key_from_text(text)
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                return api_key
    return None


def resolve_anthropic_api_key(repo_root: Path) -> tuple[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key, "environment"

    env_path = repo_root / ".env"
    if env_path.exists() and load_dotenv is not None:
        load_dotenv(env_path)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return api_key, "repo_dotenv"

    api_key = _load_api_key_from_vscode_history()
    if api_key:
        return api_key, "vscode_history_fallback"

    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Searched process environment, repo .env, and local VS Code history fallback."
    )


def build_question_only_probe_example_from_candidate(prompt_text: str, completion_text: str) -> dict[str, str]:
    question_text, option_a_text, option_b_text = rt._parse_probe_prompt_parts(prompt_text)
    chosen_option_letter = rt._parse_chosen_option_letter(completion_text)
    chosen_option_text = option_a_text if chosen_option_letter == "A" else option_b_text
    return {
        "question_text": question_text,
        "option_a_text": strip_option_label(option_a_text),
        "option_b_text": strip_option_label(option_b_text),
        "chosen_option_letter": chosen_option_letter,
        "probe_prompt": build_question_only_probe_prompt(question_text),
        "probe_completion": strip_option_label(chosen_option_text),
    }


class QuestionOnlyAnthropicMessagesAttacker(rt.AnthropicMessagesAttacker):
    def _parse_candidates(self, raw_text: str, expected_k: int) -> list[dict[str, str]]:
        parse_errors: list[str] = []
        payload: Optional[dict[str, Any]] = None
        try:
            payload = rt._extract_json_payload(raw_text)
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise RuntimeError(f"Attacker JSON missing candidates list:\n{payload}")
        except Exception as exc:
            parse_errors.append(str(exc))
            candidates = []
            for item_text in rt._extract_candidate_object_strings(raw_text):
                try:
                    candidates.append(json.loads(rt._cleanup_json_text(item_text)))
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

            stored_completion_text = "\n" + completion_text if not completion_text.startswith("\n") else completion_text

            claimed_label = item.get("intended_label", rt.LABEL_LONG)
            try:
                claimed_label = rt._normalize_label(str(claimed_label))
            except ValueError:
                continue

            try:
                probe_example = build_question_only_probe_example_from_candidate(
                    prompt_text=prompt_text,
                    completion_text=stored_completion_text,
                )
            except Exception as exc:
                parse_errors.append(f"candidate_{idx}: {exc}")
                continue

            normalized.append(
                {
                    "prompt_text": prompt_text,
                    "completion_text": stored_completion_text,
                    "question_text": probe_example["question_text"],
                    "option_a_text": probe_example["option_a_text"],
                    "option_b_text": probe_example["option_b_text"],
                    "chosen_option_letter": probe_example["chosen_option_letter"],
                    "probe_prompt": probe_example["probe_prompt"],
                    "probe_completion": probe_example["probe_completion"],
                    "intended_label": claimed_label,
                    "attacker_claimed_label": claimed_label,
                    "attack_strategy": str(item.get("attack_strategy", f"strategy_{idx}")).strip() or f"strategy_{idx}",
                    "rationale": str(item.get("rationale", "")).strip(),
                }
            )

        if not normalized:
            raise RuntimeError(
                "Attacker returned no usable candidates. "
                f"Parsed candidate objects: {len(candidates)}. Parse errors: {parse_errors}"
            )

        normalized = normalized[:expected_k]
        if len(normalized) != expected_k:
            raise RuntimeError(
                f"Attacker returned {len(normalized)} usable candidates, expected {expected_k}. Parse errors: {parse_errors}"
            )
        return normalized


class Qwen32BQuestionOnlyArtifactProbe:
    def __init__(
        self,
        *,
        model_name: str,
        artifact_path: Path,
        metadata_path: Optional[Path],
        train_regime: str,
        feature_name: str,
        vector_key: str,
        target_layer: int,
        probe_batch_size: int,
        use_chat_template: bool,
        disable_thinking_trace: bool,
        explicit_pairs,
        implicit_pairs,
        explicit_dataset_path: Path,
        implicit_dataset_path: Path,
        explicit_dataset_sha256: str,
        implicit_dataset_sha256: str,
    ) -> None:
        self.model_name = model_name
        self.artifact_path = artifact_path
        self.metadata_path = metadata_path
        self.train_regime = train_regime
        self.feature_name = feature_name
        self.vector_key = vector_key
        self.target_layer = int(target_layer)
        self.probe_batch_size = int(probe_batch_size)
        self.use_chat_template = bool(use_chat_template)
        self.disable_thinking_trace = bool(disable_thinking_trace)

        self.explicit_pairs = explicit_pairs
        self.implicit_pairs = implicit_pairs
        self.explicit_dataset_path = explicit_dataset_path
        self.implicit_dataset_path = implicit_dataset_path
        self.explicit_dataset_sha256 = explicit_dataset_sha256
        self.implicit_dataset_sha256 = implicit_dataset_sha256

        self.tokenizer = None
        self.model = None
        self.direction: Optional[np.ndarray] = None
        self.raw_direction: Optional[np.ndarray] = None
        self.score_scale: Optional[float] = None
        self.wmm_mean_train: Optional[np.ndarray] = None
        self.metadata: dict[str, Any] = {}
        self.train_question_indices: np.ndarray = np.asarray([], dtype=np.int64)
        self.test_question_indices: np.ndarray = np.asarray([], dtype=np.int64)

        self._load_artifact()

    def _load_artifact(self) -> None:
        bundle = np.load(self.artifact_path)
        try:
            metadata = load_probe_metadata(self.metadata_path) or {}
            if not metadata_is_compatible(metadata):
                raise ValueError(
                    f"Artifact metadata at {self.metadata_path} is not compatible with this Qwen3-32B question-only red-team script."
                )

            train_regimes = decode_str_array(bundle["train_regimes"])
            feature_names = decode_str_array(bundle["feature_names"])
            available_layers = bundle["layers"].astype(int)

            regime_matches = [idx for idx, name in enumerate(train_regimes) if name == self.train_regime]
            feature_matches = [idx for idx, name in enumerate(feature_names) if name == self.feature_name]
            if len(regime_matches) != 1:
                raise ValueError(f"Train regime {self.train_regime!r} not found exactly once. Available: {train_regimes}")
            if len(feature_matches) != 1:
                raise ValueError(f"Feature name {self.feature_name!r} not found exactly once. Available: {feature_names}")
            layer_matches = np.where(available_layers == int(self.target_layer))[0]
            if layer_matches.size != 1:
                raise ValueError(
                    f"Layer {self.target_layer} not found exactly once in artifact. Available layers: {available_layers.tolist()}"
                )
            if self.vector_key not in bundle.files:
                raise KeyError(f"{self.vector_key} not found in {self.artifact_path}. Available keys: {bundle.files}")

            regime_idx = int(regime_matches[0])
            feature_idx = int(feature_matches[0])
            layer_idx = int(layer_matches[0])

            raw_key = None
            if self.vector_key == "mm_probe_vectors" and "mm_raw_directions" in bundle.files:
                raw_key = "mm_raw_directions"
            elif self.vector_key == "wmm_probe_vectors" and "wmm_effective_directions" in bundle.files:
                raw_key = "wmm_effective_directions"

            self.direction = bundle[self.vector_key][regime_idx, feature_idx, layer_idx, :].astype(np.float32)
            self.raw_direction = (
                bundle[raw_key][regime_idx, feature_idx, layer_idx, :].astype(np.float32)
                if raw_key is not None
                else self.direction.copy()
            )
            if self.vector_key == "wmm_probe_vectors":
                if "wmm_mean_train" not in bundle.files:
                    raise KeyError(f"wmm_mean_train not found in {self.artifact_path}. Available keys: {bundle.files}")
                self.wmm_mean_train = bundle["wmm_mean_train"][regime_idx, feature_idx, layer_idx, :].astype(np.float32)

            if self.train_regime == "explicit_train_only":
                self.train_question_indices = bundle["explicit_train_question_indices"].astype(np.int64)
                self.test_question_indices = bundle["explicit_test_question_indices"].astype(np.int64)
                target_pairs = self.explicit_pairs
                target_domain = "explicit"
                target_dataset_path = self.explicit_dataset_path
            elif self.train_regime == "implicit_train_only":
                self.train_question_indices = bundle["implicit_train_question_indices"].astype(np.int64)
                self.test_question_indices = bundle["implicit_test_question_indices"].astype(np.int64)
                target_pairs = self.implicit_pairs
                target_domain = "implicit"
                target_dataset_path = self.implicit_dataset_path
            else:
                raise ValueError(
                    f"Unsupported train_regime={self.train_regime!r}. Expected explicit_train_only or implicit_train_only."
                )

            train_examples, train_labels = self._build_question_only_train_examples(target_pairs, self.train_question_indices)
            self.score_scale, calibration_payload = self._estimate_score_scale(train_examples, train_labels)
            metric_payload = self._load_metric_payload()

            self.metadata = {
                "probe_type": "artifact_backed_mean_mass",
                "model_name": self.model_name,
                "layer": int(self.target_layer),
                "train_regime": self.train_regime,
                "feature_name": self.feature_name,
                "vector_key": self.vector_key,
                "probe_format": "question_only_teacher_forced_answers",
                "answer_pooling": self.feature_name,
                "artifact_path": str(self.artifact_path),
                "metadata_path": str(self.metadata_path) if self.metadata_path is not None else None,
                "direction_norm": float(np.linalg.norm(self.raw_direction)),
                "score_scale": float(self.score_scale),
                "train_question_count": int(len(self.train_question_indices)),
                "test_question_count": int(len(self.test_question_indices)),
                "train_example_count": int(len(train_examples)),
                "probe_domain": target_domain,
                "probe_dataset_path": str(target_dataset_path),
                "explicit_dataset_path": str(self.explicit_dataset_path),
                "implicit_dataset_path": str(self.implicit_dataset_path),
                "explicit_expanded_sha256": self.explicit_dataset_sha256,
                "implicit_expanded_sha256": self.implicit_dataset_sha256,
                "artifact_format_version": int(metadata.get("artifact_format_version", 0)),
                "prompt_family": metadata.get("prompt_family"),
                "probe_prompt_use_chat_template": bool(metadata.get("probe_prompt_use_chat_template", False)),
                "probe_prompt_disable_thinking_trace": bool(metadata.get("probe_prompt_disable_thinking_trace", False)),
                "created_at": rt._now_utc(),
            }
            self.metadata.update(calibration_payload)
            self.metadata.update(metric_payload)
        finally:
            bundle.close()

    def _load_metric_payload(self) -> dict[str, Any]:
        metrics_path = self.artifact_path.with_name(
            self.artifact_path.name.replace("_probe_artifacts_", "_probe_metrics_").replace(".npz", ".csv")
        )
        if not metrics_path.exists():
            return {}

        metrics_df = pd.read_csv(metrics_path)
        matches = metrics_df[
            (metrics_df["train_dataset"] == self.train_regime)
            & (metrics_df["feature_name"] == self.feature_name)
            & (pd.to_numeric(metrics_df["layer"], errors="coerce") == int(self.target_layer))
        ]
        if len(matches) != 1:
            return {"metrics_path": str(metrics_path)}

        row = matches.iloc[0].to_dict()
        payload = {"metrics_path": str(metrics_path)}
        family = "mm" if self.vector_key == "mm_probe_vectors" else "wmm"
        for key in [
            f"{family}_train_acc",
            f"{family}_in_domain_holdout_acc",
            f"{family}_cross_domain_holdout_acc",
            f"{family}_explicit_holdout_acc",
            f"{family}_implicit_holdout_acc",
            f"{family}_explicit_full_acc",
            f"{family}_implicit_full_acc",
        ]:
            if key in row and pd.notna(row[key]):
                payload[key] = float(row[key])
        return payload

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = model.to("cuda")
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        model_device = assert_model_on_cuda(model)
        cuda_device = torch.cuda.current_device()
        cuda_props = torch.cuda.get_device_properties(cuda_device)
        hidden_size = int(model.config.hidden_size)
        n_layers = len(model.model.layers)
        print(
            "[cuda] confirmed GPU execution:",
            f"device={model_device}",
            f"| name={cuda_props.name}",
            f"| total_memory_gb={cuda_props.total_memory / (1024 ** 3):.1f}",
            f"| device_count={torch.cuda.device_count()}",
        )
        print("Loaded model | n_layers =", n_layers, "| hidden_size =", hidden_size)

    def _build_question_only_train_examples(self, pairs, question_indices: np.ndarray):
        all_examples, all_labels = build_teacher_forced_examples_from_pairs(pairs, strip_option_letters=True)
        example_indices = question_indices_to_example_indices(question_indices)
        selected_examples = [all_examples[int(idx)] for idx in example_indices]
        selected_labels = all_labels[example_indices]
        return selected_examples, selected_labels

    def _extract_feature_matrix(self, examples, *, progress_desc: str) -> np.ndarray:
        self._load_model()
        assert self.model is not None
        assert self.tokenizer is not None

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        model_device = assert_model_on_cuda(self.model)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if model_device.type == "cuda"
            else nullcontext()
        )

        pooled_rows = []
        progress = tqdm(
            range(0, len(examples), self.probe_batch_size),
            total=(len(examples) + self.probe_batch_size - 1) // self.probe_batch_size,
            desc=progress_desc,
            unit="batch",
        )
        for start in progress:
            batch_examples = examples[start:start + self.probe_batch_size]
            prompt_ids_batch = []
            full_ids_batch = []
            seq_lengths = []
            answer_spans = []

            for example in batch_examples:
                model_prompt = format_prompt_for_model(
                    self.tokenizer,
                    example["prompt"],
                    use_chat_template=self.use_chat_template,
                    disable_thinking_trace=self.disable_thinking_trace,
                )
                prompt_ids = self.tokenizer(
                    model_prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                full_ids = self.tokenizer(
                    model_prompt + example["continuation"],
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                continuation_token_count = int(full_ids.shape[0] - prompt_ids.shape[0])
                if continuation_token_count <= 0:
                    raise ValueError(f"Empty continuation for example: {example!r}")
                prompt_ids_batch.append(prompt_ids)
                full_ids_batch.append(full_ids)
                seq_lengths.append(int(full_ids.shape[0]))

            max_seq_len = max(seq_lengths)
            input_ids = torch.full((len(batch_examples), max_seq_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch_examples), max_seq_len), dtype=torch.long)

            for row_idx, (prompt_ids, seq) in enumerate(zip(prompt_ids_batch, full_ids_batch)):
                seq_len = int(seq.shape[0])
                answer_start = int(prompt_ids.shape[0])
                answer_end = seq_len
                input_ids[row_idx, :seq_len] = seq
                attention_mask[row_idx, :seq_len] = 1
                answer_spans.append((answer_start, answer_end))

            batch = move_batch_to_model_device(self.model, {"input_ids": input_ids, "attention_mask": attention_mask})
            batch_devices = {value.device.type for value in batch.values()}
            if batch_devices != {"cuda"}:
                raise RuntimeError(f"Tokenized batch is not fully on CUDA: {batch_devices}")

            with torch.inference_mode():
                with autocast_ctx:
                    outputs = self.model(**batch, output_hidden_states=True, use_cache=False)

            hidden = outputs.hidden_states[self.target_layer + 1]
            for row_idx, (answer_start, answer_end) in enumerate(answer_spans):
                answer_hidden = hidden[row_idx, answer_start:answer_end, :]
                if self.feature_name == "mean_answer_tokens":
                    pooled = answer_hidden.mean(dim=0)
                elif self.feature_name == "last_answer_token":
                    pooled = answer_hidden[-1, :]
                else:
                    raise ValueError(f"Unsupported feature_name={self.feature_name!r}")
                pooled_rows.append(pooled.detach().float().cpu().numpy().astype(np.float32))

            del outputs
            clear_gpu_cache()

        return np.stack(pooled_rows, axis=0).astype(np.float32)

    def _scores_from_features(self, X: np.ndarray) -> np.ndarray:
        assert self.direction is not None
        if self.vector_key == "wmm_probe_vectors":
            if self.wmm_mean_train is None:
                raise RuntimeError("wmm_mean_train is required when scoring wmm_probe_vectors.")
            X = X - self.wmm_mean_train
        return X @ self.direction

    def _estimate_score_scale(self, train_examples, train_labels: np.ndarray) -> tuple[float, dict[str, Any]]:
        X_train = self._extract_feature_matrix(
            train_examples,
            progress_desc=f"Calibrating {self.train_regime} layer {self.target_layer}",
        )
        scores = self._scores_from_features(X_train)
        preds = (scores > 0.0).astype(np.int64)
        family = "mm" if self.vector_key == "mm_probe_vectors" else "wmm"
        calibration_payload = {
            "calibrated_train_accuracy": float((preds == train_labels).mean()),
            f"{family}_score_std_train": float(np.std(scores)),
        }
        scale = float(np.std(scores))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.mean(np.abs(scores)))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        return scale, calibration_payload

    def score_probe_examples(self, examples: list[dict[str, str]]) -> list[rt.ProbeScore]:
        if self.direction is None or self.score_scale is None:
            raise RuntimeError("Probe direction and score scale must be initialized before scoring.")
        X = self._extract_feature_matrix(
            examples,
            progress_desc=f"Scoring round candidates layer {self.target_layer}",
        )
        scores = self._scores_from_features(X)
        prob_long = rt._sigmoid(scores / self.score_scale)

        results = []
        for score, p_long in zip(scores, prob_long):
            label = rt.LABEL_LONG if score > 0.0 else rt.LABEL_SHORT
            results.append(
                rt.ProbeScore(
                    label=label,
                    margin=float(score),
                    confidence=rt._safe_confidence(float(p_long), label),
                    p_long_term=float(p_long),
                )
            )
        return results


def default_artifact_search_roots(root: Path):
    return [
        root / "results" / "qwen3_32b" / "question_only_probe_variations",
        Path("/workspace/results/qwen3_32b/question_only_probe_variations"),
    ]


def resolve_output_root(root: Path, output_root: str) -> Path:
    path = Path(output_root)
    if not path.is_absolute():
        path = root / path
    return path


def build_parser() -> argparse.ArgumentParser:
    defaults = Qwen32BRedTeamConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run the current multi-turn Anthropic-attacker + Anthropic-judge red-teaming loop against a "
            "saved Qwen3-32B question-only probe artifact."
        )
    )
    parser.add_argument("--artifact-path", type=str, default=defaults.artifact_path, help="Optional explicit probe artifact .npz path.")
    parser.add_argument("--metadata-path", type=str, default=defaults.metadata_path, help="Optional explicit probe metadata .json path.")
    parser.add_argument("--explicit-path", type=str, default=defaults.explicit_path, help="Path to explicit-expanded dataset JSON.")
    parser.add_argument("--implicit-path", type=str, default=defaults.implicit_path, help="Path to implicit-expanded dataset JSON.")
    parser.add_argument("--train-regime", type=str, default=defaults.train_regime, help="Probe train regime: explicit_train_only or implicit_train_only.")
    parser.add_argument("--feature-name", type=str, default=defaults.feature_name, help="Probe feature name: mean_answer_tokens or last_answer_token.")
    parser.add_argument("--vector-key", type=str, default=defaults.vector_key, help="Probe vector key. Default is mm_probe_vectors.")
    parser.add_argument("--target-layer", type=int, default=defaults.target_layer, help="Layer to target inside the saved probe artifact.")
    parser.add_argument("--output-root", type=str, default=defaults.output_root, help="Output root for run artifacts.")
    parser.add_argument("--run-id", type=str, default=defaults.run_id, help="Optional explicit run id.")
    parser.add_argument("--attacker-model", type=str, default=defaults.attacker_model, help="Anthropic attacker model.")
    parser.add_argument("--judge-model", type=str, default=defaults.judge_model, help="Anthropic judge model.")
    parser.add_argument("--attacker-max-output-tokens", type=int, default=defaults.attacker_max_output_tokens, help="Max attacker output tokens.")
    parser.add_argument("--judge-max-output-tokens", type=int, default=defaults.judge_max_output_tokens, help="Max judge output tokens.")
    parser.add_argument("--num-rounds", type=int, default=defaults.num_rounds, help="Number of red-team rounds.")
    parser.add_argument("--candidates-per-round", type=int, default=defaults.candidates_per_round, help="Candidates per round.")
    parser.add_argument("--attacker-max-retries", type=int, default=defaults.attacker_max_retries, help="Retries per round if generation/parsing fails.")
    parser.add_argument("--judge-max-retries", type=int, default=defaults.judge_max_retries, help="Retries per round if judging/parsing fails.")
    parser.add_argument("--probe-batch-size", type=int, default=defaults.probe_batch_size, help="Batch size for Qwen3-32B activation extraction during probe scoring.")
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed, help="Random seed for reproducibility metadata.")
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="Show tqdm progress output.")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Hide tqdm progress output.")
    parser.set_defaults(show_progress=defaults.show_progress)
    return parser


def build_config_from_args(args: argparse.Namespace) -> Qwen32BRedTeamConfig:
    return Qwen32BRedTeamConfig(
        artifact_path=args.artifact_path,
        metadata_path=args.metadata_path,
        explicit_path=args.explicit_path,
        implicit_path=args.implicit_path,
        train_regime=args.train_regime,
        feature_name=args.feature_name,
        vector_key=args.vector_key,
        target_layer=args.target_layer,
        output_root=args.output_root,
        run_id=args.run_id,
        attacker_model=args.attacker_model,
        judge_model=args.judge_model,
        attacker_max_output_tokens=args.attacker_max_output_tokens,
        judge_max_output_tokens=args.judge_max_output_tokens,
        num_rounds=args.num_rounds,
        candidates_per_round=args.candidates_per_round,
        attacker_max_retries=args.attacker_max_retries,
        judge_max_retries=args.judge_max_retries,
        probe_batch_size=args.probe_batch_size,
        random_seed=args.random_seed,
        show_progress=args.show_progress,
    )


def run_red_teaming(config: Qwen32BRedTeamConfig) -> dict[str, Any]:
    root = find_repo_root(Path.cwd())
    require_cuda_runtime(require_cuda=bool(config.require_cuda))
    _, api_key_source = resolve_anthropic_api_key(root)

    run_id = config.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_root = resolve_output_root(root, config.output_root)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_dataset_path = (
        Path(config.explicit_path).expanduser().resolve()
        if config.explicit_path
        else pick_first_existing(
            [
                root / "data" / "raw" / "temporal_scope_AB_randomized" / "temporal_scope_explicit_expanded_500.json",
                root / "data" / "raw" / "temporal_scope" / "temporal_scope_explicit_expanded_500.json",
                root / "data" / "raw" / "temporal_scope_explicit_expanded_500.json",
            ]
        )
    )
    implicit_dataset_path = (
        Path(config.implicit_path).expanduser().resolve()
        if config.implicit_path
        else pick_first_existing(
            [
                root / "data" / "raw" / "temporal_scope_AB_randomized" / "temporal_scope_implicit_expanded_300.json",
                root / "data" / "raw" / "temporal_scope" / "temporal_scope_implicit_expanded_300.json",
                root / "data" / "raw" / "temporal_scope_implicit_expanded_300.json",
            ]
        )
    )
    explicit_metadata, explicit_pairs = load_pairs(explicit_dataset_path)
    implicit_metadata, implicit_pairs = load_pairs(implicit_dataset_path)
    explicit_dataset_sha256 = sha256(explicit_dataset_path)
    implicit_dataset_sha256 = sha256(implicit_dataset_path)

    artifact_search_roots = config.artifact_search_roots or [str(path) for path in default_artifact_search_roots(root)]
    artifact_path, metadata_path = locate_latest_probe_artifacts(
        artifact_search_roots,
        artifact_path_override=config.artifact_path,
        metadata_path_override=config.metadata_path,
        required_train_regime=config.train_regime,
    )
    artifact_metadata = load_probe_metadata(metadata_path) or {}

    if artifact_metadata.get("model_name") and artifact_metadata.get("model_name") != config.model_name:
        print("Warning: probe artifact model_name differs from current model_name:", artifact_metadata.get("model_name"))

    artifact_explicit_sha = artifact_metadata.get("explicit_expanded_sha256")
    if artifact_explicit_sha and artifact_explicit_sha != explicit_dataset_sha256:
        raise ValueError(
            "Probe artifact explicit dataset SHA does not match the current explicit dataset file. "
            f"artifact={artifact_explicit_sha} current={explicit_dataset_sha256}"
        )

    artifact_implicit_sha = artifact_metadata.get("implicit_expanded_sha256")
    if artifact_implicit_sha and artifact_implicit_sha != implicit_dataset_sha256:
        raise ValueError(
            "Probe artifact implicit dataset SHA does not match the current implicit dataset file. "
            f"artifact={artifact_implicit_sha} current={implicit_dataset_sha256}"
        )

    if config.show_progress:
        print(
            f"[red-teaming] Run {run_id}: loading Qwen3-32B layer-{config.target_layer} "
            f"{config.train_regime} {config.feature_name} {config.vector_key} probe artifact from {artifact_path}..."
        )
        print("Anthropic key source:", api_key_source)

    probe = Qwen32BQuestionOnlyArtifactProbe(
        model_name=config.model_name,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        train_regime=config.train_regime,
        feature_name=config.feature_name,
        vector_key=config.vector_key,
        target_layer=config.target_layer,
        probe_batch_size=config.probe_batch_size,
        use_chat_template=config.use_chat_template,
        disable_thinking_trace=config.disable_thinking_trace,
        explicit_pairs=explicit_pairs,
        implicit_pairs=implicit_pairs,
        explicit_dataset_path=explicit_dataset_path,
        implicit_dataset_path=implicit_dataset_path,
        explicit_dataset_sha256=explicit_dataset_sha256,
        implicit_dataset_sha256=implicit_dataset_sha256,
    )

    attacker = QuestionOnlyAnthropicMessagesAttacker(
        actor="attacker",
        model=config.attacker_model,
        base_url=config.attacker_base_url,
        timeout_seconds=config.attacker_timeout_seconds,
        max_output_tokens=config.attacker_max_output_tokens,
        context_window_tokens=config.attacker_context_window_tokens,
    )
    judge = rt.AnthropicMessagesJudge(
        actor="judge",
        model=config.judge_model,
        base_url=config.judge_base_url,
        timeout_seconds=config.judge_timeout_seconds,
        max_output_tokens=config.judge_max_output_tokens,
        context_window_tokens=config.judge_context_window_tokens,
    )

    run_config_payload = asdict(config)
    run_config_payload.update(
        {
            "resolved_run_id": run_id,
            "resolved_output_dir": str(output_dir),
            "resolved_artifact_path": str(artifact_path),
            "resolved_metadata_path": str(metadata_path) if metadata_path is not None else None,
            "explicit_dataset_path": str(explicit_dataset_path),
            "implicit_dataset_path": str(implicit_dataset_path),
            "explicit_dataset_sha256": explicit_dataset_sha256,
            "implicit_dataset_sha256": implicit_dataset_sha256,
            "anthropic_api_key_source": api_key_source,
            "explicit_metadata": explicit_metadata,
            "implicit_metadata": implicit_metadata,
        }
    )
    rt._write_json(output_dir / "run_config.json", run_config_payload)
    rt._write_json(output_dir / "target_probe.json", probe.metadata)

    attacker_system_prompt = rt._build_attacker_system_prompt()
    attacker_messages: list[dict[str, str]] = [
        {"role": "user", "content": rt._build_initial_attacker_user_prompt(config)}
    ]
    rt._write_text(output_dir / "attacker_system_prompt.txt", attacker_system_prompt)
    rt._write_text(output_dir / "attacker_initial_user_prompt.txt", attacker_messages[0]["content"])
    rt._write_text(output_dir / "judge_system_prompt.txt", rt._build_judge_system_prompt())

    all_candidates: list[rt.AdversarialCandidate] = []
    successful_candidates: list[rt.AdversarialCandidate] = []
    attacker_api_usage_records: list[dict[str, Any]] = []
    judge_api_usage_records: list[dict[str, Any]] = []

    round_iterator = tqdm(
        range(config.num_rounds),
        total=config.num_rounds,
        desc="Red-teaming",
        unit="round",
        disable=not config.show_progress,
    )

    for round_idx in round_iterator:
        stem = f"round_{round_idx:03d}"
        rt._write_json(output_dir / f"{stem}_attacker_request_messages.json", attacker_messages)

        try:
            raw_candidates, attacker_raw_text, attacker_usage_record = attacker.generate_candidates(
                system_prompt=attacker_system_prompt,
                request_messages=attacker_messages,
                round_idx=round_idx,
                expected_k=config.candidates_per_round,
                max_retries=config.attacker_max_retries,
            )
        except rt.MessagesGenerationError as exc:
            rt._write_generation_failure(output_dir, f"{stem}_attacker", exc)
            raise

        attacker_usage_payload = rt._usage_record_to_dict(attacker_usage_record, attacker_api_usage_records)
        attacker_api_usage_records.append(attacker_usage_payload)
        rt._write_json(output_dir / f"{stem}_attacker_api_usage.json", attacker_usage_payload)
        rt._write_jsonl(output_dir / "attacker_api_usage.jsonl", attacker_api_usage_records)
        rt._write_text(output_dir / f"{stem}_attacker_response.txt", attacker_raw_text)

        rt._assign_candidate_ids(raw_candidates, round_idx)

        probe_examples = [
            {"prompt": item["probe_prompt"], "continuation": item["probe_completion"]}
            for item in raw_candidates
        ]
        scores = probe.score_probe_examples(probe_examples)

        try:
            judge_results, judge_system_prompt, judge_user_prompt, judge_raw_text, judge_usage_record = judge.judge_candidates(
                round_idx=round_idx,
                raw_candidates=raw_candidates,
                max_retries=config.judge_max_retries,
            )
        except rt.MessagesGenerationError as exc:
            rt._write_generation_failure(output_dir, f"{stem}_judge", exc)
            raise

        judge_usage_payload = rt._usage_record_to_dict(judge_usage_record, judge_api_usage_records)
        judge_api_usage_records.append(judge_usage_payload)
        rt._write_json(output_dir / f"{stem}_judge_api_usage.json", judge_usage_payload)
        rt._write_jsonl(output_dir / "judge_api_usage.jsonl", judge_api_usage_records)
        rt._write_text(output_dir / f"{stem}_judge_system_prompt.txt", judge_system_prompt)
        rt._write_text(output_dir / f"{stem}_judge_user_prompt.txt", judge_user_prompt)
        rt._write_text(output_dir / f"{stem}_judge_raw_response.txt", judge_raw_text)
        rt._write_json(output_dir / f"{stem}_judge_evaluations.json", [asdict(row) for row in judge_results])

        batch_candidates = rt._materialize_candidates(
            raw_candidates=raw_candidates,
            scores=scores,
            judge_results=judge_results,
            config=config,
            round_idx=round_idx,
            run_id=run_id,
        )

        feedback_prompt = rt._build_feedback_user_prompt(
            round_idx=round_idx,
            num_rounds=config.num_rounds,
            candidates_per_round=config.candidates_per_round,
            batch_candidates=batch_candidates,
        )
        attacker_messages.append({"role": "assistant", "content": attacker_raw_text})
        attacker_messages.append({"role": "user", "content": feedback_prompt})

        all_candidates.extend(batch_candidates)
        successful_candidates.extend([row for row in batch_candidates if row.is_adversarial_success])

        rt._write_text(output_dir / f"{stem}_feedback_to_attacker.txt", feedback_prompt)
        rt._write_jsonl(output_dir / f"{stem}_candidates.jsonl", [rt._candidate_to_dict(x) for x in batch_candidates])
        rt._write_json(
            output_dir / f"{stem}_summary.json",
            {"round_idx": round_idx, **rt._candidate_summary(batch_candidates)},
        )

        if config.show_progress:
            round_iterator.set_postfix(
                saved=len(all_candidates),
                successes=len(successful_candidates),
                success_rate=f"{(len(successful_candidates) / max(len(all_candidates), 1)):.1%}",
                attacker_tokens=attacker_usage_payload.get("cumulative_total_request_tokens_used_so_far", 0),
                judge_tokens=judge_usage_payload.get("cumulative_total_request_tokens_used_so_far", 0),
            )

    round_iterator.close()

    rt._write_jsonl(output_dir / "all_candidates.jsonl", [rt._candidate_to_dict(x) for x in all_candidates])
    rt._write_jsonl(
        output_dir / "successful_adversarial_examples.jsonl",
        [rt._candidate_to_dict(x) for x in successful_candidates],
    )
    rt._write_json(
        output_dir / "attacker_conversation.json",
        {
            "system_prompt": attacker_system_prompt,
            "messages": attacker_messages,
        },
    )

    attacker_strategy_summary_artifact: Optional[dict[str, Any]] = None
    attacker_strategy_summary_error: Optional[dict[str, Any]] = None

    if all_candidates:
        if config.show_progress:
            print("[red-teaming] Generating attacker strategy summary...")

        summary_prompt = rt._build_summary_request_prompt()
        summary_messages = list(attacker_messages) + [{"role": "user", "content": summary_prompt}]
        rt._write_json(output_dir / "attacker_strategy_summary_request_messages.json", summary_messages)
        try:
            summary_payload, summary_raw_text, strategy_usage_record = attacker.summarize_conversation(
                system_prompt=attacker_system_prompt,
                request_messages=summary_messages,
                max_retries=config.attacker_max_retries,
            )
            strategy_usage_payload = rt._usage_record_to_dict(strategy_usage_record, attacker_api_usage_records)
            attacker_api_usage_records.append(strategy_usage_payload)
            rt._write_json(output_dir / "attacker_strategy_summary_api_usage.json", strategy_usage_payload)
            rt._write_jsonl(output_dir / "attacker_api_usage.jsonl", attacker_api_usage_records)
            attacker_strategy_summary_artifact = {
                "run_id": run_id,
                "created_at": rt._now_utc(),
                "attacker_model": config.attacker_model,
                **summary_payload,
            }
            rt._write_text(output_dir / "attacker_strategy_summary_user_prompt.txt", summary_prompt)
            rt._write_text(output_dir / "attacker_strategy_summary_raw_response.txt", summary_raw_text)
            rt._write_json(output_dir / "attacker_strategy_summary.json", attacker_strategy_summary_artifact)
            rt._write_text(
                output_dir / "attacker_strategy_summary.md",
                rt._render_strategy_summary_markdown(attacker_strategy_summary_artifact),
            )
        except rt.MessagesGenerationError as exc:
            rt._write_generation_failure(output_dir, "attacker_strategy_summary", exc)
            attacker_strategy_summary_error = {
                "run_id": run_id,
                "created_at": rt._now_utc(),
                "attacker_model": config.attacker_model,
                "error": str(exc),
            }
            rt._write_json(output_dir / "attacker_strategy_summary_error.json", attacker_strategy_summary_error)
        except Exception as exc:
            attacker_strategy_summary_error = {
                "run_id": run_id,
                "created_at": rt._now_utc(),
                "attacker_model": config.attacker_model,
                "error": str(exc),
            }
            rt._write_json(output_dir / "attacker_strategy_summary_error.json", attacker_strategy_summary_error)

    final_summary = {
        "run_id": run_id,
        "created_at": rt._now_utc(),
        "explicit_dataset_path": str(explicit_dataset_path),
        "implicit_dataset_path": str(implicit_dataset_path),
        "attacker_model": config.attacker_model,
        "judge_model": config.judge_model,
        "mode": "full_history_with_judge",
        "probe": probe.metadata,
        "run": {
            "n_rounds": config.num_rounds,
            "candidates_per_round": config.candidates_per_round,
            "expected_candidates": config.num_rounds * config.candidates_per_round,
            **rt._candidate_summary(all_candidates),
            "round_summary": rt._round_summary(all_candidates),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "all_candidates": str(output_dir / "all_candidates.jsonl"),
            "successful_candidates": str(output_dir / "successful_adversarial_examples.jsonl"),
            "attacker_api_usage": str(output_dir / "attacker_api_usage.jsonl"),
            "judge_api_usage": str(output_dir / "judge_api_usage.jsonl"),
            "attacker_conversation": str(output_dir / "attacker_conversation.json"),
            "attacker_strategy_summary": (
                str(output_dir / "attacker_strategy_summary.json")
                if attacker_strategy_summary_artifact is not None
                else None
            ),
            "attacker_strategy_summary_markdown": (
                str(output_dir / "attacker_strategy_summary.md")
                if attacker_strategy_summary_artifact is not None
                else None
            ),
            "attacker_strategy_summary_error": (
                str(output_dir / "attacker_strategy_summary_error.json")
                if attacker_strategy_summary_error is not None
                else None
            ),
        },
    }
    rt._write_json(output_dir / "final_summary.json", final_summary)
    return final_summary


def main() -> int:
    args = build_parser().parse_args()
    config = build_config_from_args(args)
    summary = run_red_teaming(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
