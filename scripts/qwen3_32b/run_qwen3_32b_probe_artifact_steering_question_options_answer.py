#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION = 1


DEFAULT_CONFIG = {
    'model_name': 'Qwen/Qwen3-32B',
    'use_chat_template': True,
    'disable_thinking_trace': True,
    'dataset_source': 'expanded',
    'split_random_state': 42,
    'patch_prompt_last_only': True,
    'patch_generation_tokens': True,
    'max_new_tokens': 32,
    'quick_mode': False,
    'max_prompts_explicit': None,
    'max_prompts_implicit': None,
    'artifact_search_roots': None,
    'artifact_path': None,
    'metadata_path': None,
    'train_regime': 'explicit_train_only',
    'feature_name': 'mean_answer_tokens',
    'vector_key': 'mm_probe_vectors',
    'layers_to_test': [24, 28, 32, 36, 40, 44, 48],
    'strengths': [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    'output_root_name': 'qwen3_32b/probe_artifact_steering_question_options_answer_vast',
    'require_cuda': True,
    'reuse_existing_results': True,
    'reuse_result_search_roots': None,
    'run_id': None,
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'notebooks').exists():
            return candidate
    raise RuntimeError('Could not locate repo root from current working directory.')


def pick_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError('None of these paths exist: ' + str([str(p) for p in paths]))


def load_pairs(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'pairs' in data:
        return data.get('metadata', {}), data['pairs']
    return {}, data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_cuda_runtime(*, require_cuda: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required for this run, but torch.cuda.is_available() is False. '
            'Refusing to fall back to CPU/MPS.'
        )


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
            decoded.append(value.decode('utf-8'))
        else:
            decoded.append(str(value))
    return decoded


def resolve_metadata_path_for_artifact(artifact_path: Path, metadata_candidates):
    candidate = artifact_path.with_name(artifact_path.name.replace('_probe_artifacts_', '_probe_metadata_').replace('.npz', '.json'))
    if candidate.exists():
        return candidate
    for metadata_path in metadata_candidates:
        if metadata_path.stem.replace('_probe_metadata_', '_probe_artifacts_') == artifact_path.stem:
            return metadata_path
    return None


def load_probe_metadata(metadata_path):
    if metadata_path is None or not Path(metadata_path).exists():
        return None
    return json.loads(Path(metadata_path).read_text(encoding='utf-8'))


def metadata_is_compatible(metadata):
    if not metadata:
        return False
    if int(metadata.get('artifact_format_version', 0)) < REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION:
        return False
    if metadata.get('prompt_family') != 'question_only_teacher_forced_answers':
        return False
    if metadata.get('explicit_split_granularity') != 'question':
        return False
    if metadata.get('implicit_split_granularity') != 'question':
        return False
    if not bool(metadata.get('probe_prompt_use_chat_template', metadata.get('use_chat_template', False))):
        return False
    if not bool(metadata.get('probe_prompt_disable_thinking_trace', metadata.get('disable_thinking_trace', False))):
        return False
    return True


def artifact_contains_train_regime(artifact_path, required_train_regime):
    if required_train_regime is None:
        return True
    try:
        with np.load(artifact_path) as bundle:
            train_regimes = decode_str_array(bundle['train_regimes'])
    except Exception:
        return False
    return required_train_regime in train_regimes


def locate_latest_probe_artifacts(search_roots, artifact_path_override=None, metadata_path_override=None, required_train_regime=None):
    if artifact_path_override is not None:
        artifact_path = Path(artifact_path_override).expanduser().resolve()
        if metadata_path_override is not None:
            metadata_path = Path(metadata_path_override).expanduser().resolve()
        else:
            metadata_path = resolve_metadata_path_for_artifact(artifact_path, [])
        if not artifact_contains_train_regime(artifact_path, required_train_regime):
            raise ValueError(
                f'Artifact {artifact_path} does not contain requested train_regime={required_train_regime!r}.'
            )
        return artifact_path, metadata_path

    artifact_candidates = []
    metadata_candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        artifact_candidates.extend(sorted(root.rglob('qwen3_32b_question_only_probe_artifacts_*.npz')))
        metadata_candidates.extend(sorted(root.rglob('qwen3_32b_question_only_probe_metadata_*.json')))

    artifact_candidates = unique_paths(artifact_candidates)
    metadata_candidates = unique_paths(metadata_candidates)

    if not artifact_candidates:
        roots_text = ', '.join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            'Could not find any Qwen3 probe-artifact bundles under: '
            f'{roots_text}. Set artifact_path explicitly if needed.'
        )

    compatible_candidates = []
    for artifact_path in reversed(artifact_candidates):
        metadata_path = resolve_metadata_path_for_artifact(artifact_path, metadata_candidates)
        metadata = load_probe_metadata(metadata_path)
        if metadata_is_compatible(metadata) and artifact_contains_train_regime(artifact_path, required_train_regime):
            compatible_candidates.append((artifact_path, metadata_path))
    if not compatible_candidates:
        roots_text = ', '.join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            'Found Qwen3-32B probe artifacts, but none were compatible with the current question-only + question-split + chat-template format '
            f'(required artifact_format_version >= {REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION}). '
            f'Search roots: {roots_text}'
        )
    return compatible_candidates[0]


def extract_option_letter(option_text: str) -> str:
    match = re.search(r'\(([ABab])\)', option_text or '')
    if match is None:
        raise ValueError(f'Could not parse option letter from: {option_text!r}')
    return match.group(1).upper()


def strip_option_label(option_text: str) -> str:
    return re.sub(r'^\s*\([ABab]\)\s*', '', option_text or '').strip()


def normalize_for_match(text: str) -> str:
    text = (text or '').lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


_MATCH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'is', 'are', 'be',
    'this', 'that', 'it', 'we', 'what', 'which', 'will', 'would', 'should', 'can', 'our', 'your',
}


def content_words(text: str):
    return [
        tok for tok in normalize_for_match(text).split()
        if len(tok) >= 3 and tok not in _MATCH_STOPWORDS
    ]


def option_anchor_phrases(option_text: str):
    words = content_words(strip_option_label(option_text))
    phrases = []
    for n in (5, 4, 3):
        if len(words) >= n:
            phrases.append(' '.join(words[:n]))
            phrases.append(' '.join(words[-n:]))
    unique = []
    for phrase in phrases:
        if phrase and phrase not in unique:
            unique.append(phrase)
    return unique[:4]


def get_pair_option_payload(pair):
    immediate_letter = extract_option_letter(pair['immediate'])
    long_term_letter = extract_option_letter(pair['long_term'])
    if immediate_letter == long_term_letter:
        raise ValueError(f'Immediate and long-term options cannot share the same letter: {pair!r}')

    option_a_text = pair['immediate'] if immediate_letter == 'A' else pair['long_term']
    option_b_text = pair['immediate'] if immediate_letter == 'B' else pair['long_term']
    return {
        'immediate_letter': immediate_letter,
        'long_term_letter': long_term_letter,
        'option_a_text': option_a_text,
        'option_b_text': option_b_text,
        'candidate_immediate_text': strip_option_label(pair['immediate']),
        'candidate_long_term_text': strip_option_label(pair['long_term']),
    }


def build_probe_prompt(question_text, option_a_text, option_b_text):
    question_text = (question_text or '').strip()
    option_a_text = strip_option_label(option_a_text)
    option_b_text = strip_option_label(option_b_text)
    return (
        f'{question_text}\n'
        'Options:\n'
        f'  {option_a_text}\n'
        f'  {option_b_text}\n'
        '  Answer:\n'
    )


def format_prompt_for_model(tokenizer, user_prompt, use_chat_template=True, disable_thinking_trace=True):
    if use_chat_template:
        if not hasattr(tokenizer, 'apply_chat_template'):
            raise RuntimeError('Tokenizer does not expose apply_chat_template, but use_chat_template=True was requested.')
        messages = [{'role': 'user', 'content': user_prompt}]
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


def encode_prompt_and_full_sequence(tokenizer, prompt_text, continuation_text=None, use_chat_template=True, disable_thinking_trace=True):
    model_prompt = format_prompt_for_model(
        tokenizer,
        prompt_text,
        use_chat_template=use_chat_template,
        disable_thinking_trace=disable_thinking_trace,
    )
    prompt_ids = tokenizer(model_prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    full_text = model_prompt + (continuation_text or '')
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    return model_prompt, prompt_ids, full_ids


def build_dataset_packet(dataset_name, pairs):
    packet = []
    for idx, pair in enumerate(pairs):
        option_payload = get_pair_option_payload(pair)
        option_a_text = option_payload['option_a_text']
        option_b_text = option_payload['option_b_text']
        prompt = build_probe_prompt(pair['question'], option_a_text, option_b_text)
        packet.append({
            'dataset': dataset_name,
            'prompt_idx': idx,
            'pair': pair,
            'prompt': prompt,
            'immediate_letter': option_payload['immediate_letter'],
            'long_term_letter': option_payload['long_term_letter'],
            'option_a_text': option_a_text,
            'option_b_text': option_b_text,
            'candidate_immediate_text': option_payload['candidate_immediate_text'],
            'candidate_long_term_text': option_payload['candidate_long_term_text'],
            'option_a_normalized': normalize_for_match(option_a_text),
            'option_b_normalized': normalize_for_match(option_b_text),
            'option_a_stripped_normalized': normalize_for_match(strip_option_label(option_a_text)),
            'option_b_stripped_normalized': normalize_for_match(strip_option_label(option_b_text)),
            'a_semantic': 'choose_immediate' if option_payload['immediate_letter'] == 'A' else 'choose_long_term',
            'b_semantic': 'choose_immediate' if option_payload['immediate_letter'] == 'B' else 'choose_long_term',
            'option_a_phrases': option_anchor_phrases(option_a_text),
            'option_b_phrases': option_anchor_phrases(option_b_text),
            'option_a_content_words': content_words(option_a_text),
            'option_b_content_words': content_words(option_b_text),
        })
    return packet


def select_eval_pairs_for_train_regime(
    *,
    train_regime,
    explicit_pairs,
    implicit_pairs,
    explicit_test_question_idx,
    implicit_test_question_idx,
    max_prompts_explicit,
    max_prompts_implicit,
):
    explicit_test_pairs_full = [explicit_pairs[i] for i in explicit_test_question_idx]
    implicit_test_pairs_full = [implicit_pairs[i] for i in implicit_test_question_idx]
    explicit_pairs_full = list(explicit_pairs)
    implicit_pairs_full = list(implicit_pairs)

    if train_regime == 'explicit_train_only':
        return {
            'explicit_test': build_dataset_packet(
                'explicit_test',
                explicit_test_pairs_full[:max_prompts_explicit] if max_prompts_explicit is not None else explicit_test_pairs_full,
            ),
            'implicit_full': build_dataset_packet(
                'implicit_full',
                implicit_pairs_full[:max_prompts_implicit] if max_prompts_implicit is not None else implicit_pairs_full,
            ),
        }
    if train_regime == 'implicit_train_only':
        return {
            'implicit_test': build_dataset_packet(
                'implicit_test',
                implicit_test_pairs_full[:max_prompts_implicit] if max_prompts_implicit is not None else implicit_test_pairs_full,
            ),
            'explicit_full': build_dataset_packet(
                'explicit_full',
                explicit_pairs_full[:max_prompts_explicit] if max_prompts_explicit is not None else explicit_pairs_full,
            ),
        }
    raise ValueError(
        f"Unsupported train_regime={train_regime!r}. Expected one of ['explicit_train_only', 'implicit_train_only']."
    )


def parse_ab_from_text(text):
    stripped = (text or '').strip()
    if not stripped:
        return None

    strong = re.match(r'^[\s\n]*[\(\[]?\s*([ABab12])\s*[\)\].,:;\- ]?', stripped)
    if strong:
        ch = strong.group(1).upper()
        return {'1': 'A', '2': 'B'}.get(ch, ch)

    upper = stripped.upper()
    patterns = [
        r'\b(?:ANSWER\s*[:\-]?\s*)([AB12])\b',
        r'\bOPTION\s*([AB12])\b',
        r'\bCHOOSE\s*([AB12])\b',
        r'\(([AB12])\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            ch = match.group(1).upper()
            return {'1': 'A', '2': 'B'}.get(ch, ch)
    return None


def semantic_from_letter(packet_item, letter):
    if letter == packet_item['immediate_letter']:
        return 'choose_immediate'
    if letter == packet_item['long_term_letter']:
        return 'choose_long_term'
    return None


def common_prefix_token_count(a_tokens, b_tokens):
    count = 0
    for a_tok, b_tok in zip(a_tokens, b_tokens):
        if a_tok != b_tok:
            break
        count += 1
    return count


def literal_option_match_score(normalized_continuation, continuation_tokens, option_normalized, option_stripped_normalized, option_phrases, option_content_words):
    best_score = 0.0
    best_method = None

    candidates = [
        ('literal_exact_full', option_normalized),
        ('literal_exact_stripped', option_stripped_normalized),
    ]
    for method_name, candidate in candidates:
        if not candidate:
            continue
        candidate_tokens = candidate.split()
        if normalized_continuation == candidate:
            score = 5000.0 + len(candidate_tokens)
        elif normalized_continuation.startswith(candidate):
            score = 4500.0 + len(candidate_tokens)
        elif len(continuation_tokens) >= 3 and candidate.startswith(normalized_continuation):
            score = 4000.0 + len(continuation_tokens)
        elif candidate in normalized_continuation:
            score = 3500.0 + len(candidate_tokens)
        else:
            prefix_count = common_prefix_token_count(continuation_tokens, candidate_tokens)
            if prefix_count >= 3:
                score = 2500.0 + prefix_count
            else:
                score = 0.0
        if score > best_score:
            best_score = score
            best_method = method_name

    phrase_hits = sum(phrase in normalized_continuation for phrase in option_phrases)
    if phrase_hits > 0:
        phrase_score = 1000.0 + float(phrase_hits)
        if phrase_score > best_score:
            best_score = phrase_score
            best_method = 'literal_anchor_match'

    continuation_content_words = [tok for tok in continuation_tokens if tok not in _MATCH_STOPWORDS]
    content_prefix = common_prefix_token_count(continuation_content_words, option_content_words)
    if content_prefix >= 2:
        content_score = 800.0 + float(content_prefix)
        if content_score > best_score:
            best_score = content_score
            best_method = 'literal_content_prefix'

    return best_score, best_method


def parse_preference_completion(packet_item, continuation):
    normalized = normalize_for_match(continuation)
    continuation_tokens = normalized.split()

    if normalized:
        a_score, a_method = literal_option_match_score(
            normalized,
            continuation_tokens,
            packet_item['option_a_normalized'],
            packet_item['option_a_stripped_normalized'],
            packet_item['option_a_phrases'],
            packet_item['option_a_content_words'],
        )
        b_score, b_method = literal_option_match_score(
            normalized,
            continuation_tokens,
            packet_item['option_b_normalized'],
            packet_item['option_b_stripped_normalized'],
            packet_item['option_b_phrases'],
            packet_item['option_b_content_words'],
        )
        if max(a_score, b_score) > 0 and a_score != b_score:
            parsed_letter = 'A' if a_score > b_score else 'B'
            return {
                'parsed_letter': parsed_letter,
                'parsed_semantic': semantic_from_letter(packet_item, parsed_letter),
                'parse_method': a_method if a_score > b_score else b_method,
                'fallback_used': False,
                'score_a': float(a_score),
                'score_b': float(b_score),
            }

    parsed_letter = parse_ab_from_text(continuation)
    if parsed_letter is not None:
        return {
            'parsed_letter': parsed_letter,
            'parsed_semantic': semantic_from_letter(packet_item, parsed_letter),
            'parse_method': 'letter_parse',
            'fallback_used': False,
            'score_a': np.nan,
            'score_b': np.nan,
        }

    return {
        'parsed_letter': None,
        'parsed_semantic': None,
        'parse_method': 'unparsed',
        'fallback_used': True,
        'score_a': np.nan,
        'score_b': np.nan,
    }


def resolve_heatmap_range(matrix, default_vmin, default_vmax, mode='data'):
    if mode == 'fixed':
        return default_vmin, default_vmax
    finite_vals = matrix[np.isfinite(matrix)]
    if finite_vals.size == 0:
        return default_vmin, default_vmax
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))
    if np.isclose(vmin, vmax):
        pad = 1e-6 if np.isclose(vmin, 0.0) else max(abs(vmin) * 0.01, 1e-6)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def draw_steering_heatmap(ax, matrix, title, x_labels, y_labels, vmin=0.0, vmax=1.0, cmap='viridis', range_mode='data'):
    vmin, vmax = resolve_heatmap_range(matrix, vmin, vmax, mode=range_mode)
    im = ax.imshow(matrix, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Signed strength')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([f'{v:g}' for v in y_labels])

    finite_vals = matrix[np.isfinite(matrix)]
    midpoint = float((vmin + vmax) / 2.0) if np.isfinite(vmin) and np.isfinite(vmax) else 0.5
    if finite_vals.size:
        midpoint = float(np.nanmedian(finite_vals)) if not (np.isfinite(vmin) and np.isfinite(vmax)) else midpoint

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = '--'
                text_color = 'black'
            else:
                label = f'{value:.2f}'
                text_color = 'white' if value < midpoint else 'black'
            ax.text(col_idx, row_idx, label, ha='center', va='center', color=text_color, fontsize=8)
    return im


def pivot_metric(df, dataset_name, value_col, signed_strength_grid, layers_to_test):
    dataset_df = df[(df['dataset'] == dataset_name) & (df['layer'] >= 0)].copy()
    dataset_df = dataset_df.groupby(['signed_strength', 'layer'], as_index=False)[value_col].mean()
    pivot = dataset_df.pivot(index='signed_strength', columns='layer', values=value_col)
    return pivot.reindex(index=signed_strength_grid, columns=layers_to_test)


def load_probe_slice(artifact_path, metadata_path, train_regime, feature_name, vector_key, layers_to_test):
    bundle = np.load(artifact_path)
    metadata = load_probe_metadata(metadata_path) or {}
    if int(metadata.get('artifact_format_version', 0)) < REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f'Artifact metadata at {metadata_path} has artifact_format_version='
            f"{metadata.get('artifact_format_version')} but version >= {REQUIRED_PROBE_ARTIFACT_FORMAT_VERSION} is required."
        )
    if metadata.get('prompt_family') != 'question_only_teacher_forced_answers':
        raise ValueError(
            f'Artifact metadata at {metadata_path} uses prompt_family='
            f"{metadata.get('prompt_family')!r}; expected 'question_only_teacher_forced_answers'."
        )
    if metadata.get('explicit_split_granularity') != 'question':
        raise ValueError(
            f'Artifact metadata at {metadata_path} uses explicit_split_granularity='
            f"{metadata.get('explicit_split_granularity')!r}; expected 'question'."
        )
    if metadata.get('implicit_split_granularity') != 'question':
        raise ValueError(
            f'Artifact metadata at {metadata_path} uses implicit_split_granularity='
            f"{metadata.get('implicit_split_granularity')!r}; expected 'question'."
        )
    if not bool(metadata.get('probe_prompt_use_chat_template', metadata.get('use_chat_template', False))):
        raise ValueError(
            f'Artifact metadata at {metadata_path} does not indicate chat-template probe encoding; '
            'rerun the updated probe-training notebook first.'
        )
    if not bool(metadata.get('probe_prompt_disable_thinking_trace', metadata.get('disable_thinking_trace', False))):
        raise ValueError(
            f'Artifact metadata at {metadata_path} does not indicate thinking-trace suppression; '
            'rerun the updated probe-training notebook first.'
        )

    train_regimes = decode_str_array(bundle['train_regimes'])
    feature_names = decode_str_array(bundle['feature_names'])
    available_layers = bundle['layers'].astype(int)
    required_bundle_keys = {
        'explicit_train_question_indices',
        'explicit_test_question_indices',
        'implicit_train_question_indices',
        'implicit_test_question_indices',
    }
    missing_bundle_keys = required_bundle_keys.difference(bundle.files)
    if missing_bundle_keys:
        raise KeyError(
            f'Artifact bundle {artifact_path} is missing required keys for question-level steering reuse: '
            f'{sorted(missing_bundle_keys)}'
        )

    regime_matches = [idx for idx, name in enumerate(train_regimes) if name == train_regime]
    feature_matches = [idx for idx, name in enumerate(feature_names) if name == feature_name]
    if len(regime_matches) != 1:
        raise ValueError(f'Train regime {train_regime!r} not found exactly once. Available: {train_regimes}')
    if len(feature_matches) != 1:
        raise ValueError(f'Feature name {feature_name!r} not found exactly once. Available: {feature_names}')
    if vector_key not in bundle.files:
        raise KeyError(f'{vector_key} not found in {artifact_path}. Available keys: {bundle.files}')

    raw_key = None
    if vector_key == 'mm_probe_vectors' and 'mm_raw_directions' in bundle.files:
        raw_key = 'mm_raw_directions'
    elif vector_key == 'wmm_probe_vectors' and 'wmm_effective_directions' in bundle.files:
        raw_key = 'wmm_effective_directions'

    vector_tensor = bundle[vector_key]
    raw_tensor = bundle[raw_key] if raw_key is not None else None

    direction_store = {}
    probe_slice_rows = []
    regime_idx = int(regime_matches[0])
    feature_idx = int(feature_matches[0])

    for layer in layers_to_test:
        layer_matches = np.where(available_layers == int(layer))[0]
        if layer_matches.size != 1:
            raise ValueError(f'Layer {layer} not found exactly once in artifact. Available layers: {available_layers.tolist()}')
        layer_idx = int(layer_matches[0])
        steering_vector = vector_tensor[regime_idx, feature_idx, layer_idx, :].astype(np.float32)
        raw_vector = raw_tensor[regime_idx, feature_idx, layer_idx, :].astype(np.float32) if raw_tensor is not None else steering_vector
        direction_store[int(layer)] = {
            'steering_vector': steering_vector,
            'raw_direction': raw_vector,
            'raw_norm': float(np.linalg.norm(raw_vector)),
            'steering_norm': float(np.linalg.norm(steering_vector)),
        }
        probe_slice_rows.append({
            'train_regime': train_regime,
            'feature_name': feature_name,
            'vector_key': vector_key,
            'layer': int(layer),
            'raw_norm': float(np.linalg.norm(raw_vector)),
            'steering_norm': float(np.linalg.norm(steering_vector)),
        })

    return {
        'bundle': bundle,
        'metadata': metadata,
        'available_layers': available_layers,
        'explicit_train_question_indices': bundle['explicit_train_question_indices'].astype(np.int64),
        'explicit_test_question_indices': bundle['explicit_test_question_indices'].astype(np.int64),
        'implicit_train_question_indices': bundle['implicit_train_question_indices'].astype(np.int64),
        'implicit_test_question_indices': bundle['implicit_test_question_indices'].astype(np.int64),
        'direction_store': direction_store,
        'probe_slice_df': pd.DataFrame(probe_slice_rows).sort_values('layer').reset_index(drop=True),
    }


def coerce_existing_logs_df(df):
    if df is None or not len(df):
        return pd.DataFrame()

    df = df.copy()
    required_cols = ['dataset', 'condition', 'layer', 'strength', 'prompt_idx']
    if any(col not in df.columns for col in required_cols):
        return pd.DataFrame()

    df['dataset'] = df['dataset'].astype(str)
    df['condition'] = df['condition'].astype(str)
    df['layer'] = pd.to_numeric(df['layer'], errors='coerce')
    df['strength'] = pd.to_numeric(df['strength'], errors='coerce')
    df['prompt_idx'] = pd.to_numeric(df['prompt_idx'], errors='coerce')
    df = df.dropna(subset=['layer', 'strength', 'prompt_idx']).copy()
    df['layer'] = df['layer'].astype(int)
    df['strength'] = df['strength'].astype(float)
    df['prompt_idx'] = df['prompt_idx'].astype(int)
    return df


def load_existing_result_logs(search_roots):
    log_paths = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        log_paths.extend(sorted(root.rglob('mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_logs_*.csv')))
        log_paths.extend(sorted(root.rglob('partial/*_logs.csv')))

    log_paths = unique_paths(log_paths)
    if not log_paths:
        return pd.DataFrame()

    def mtime_or_zero(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    log_paths = sorted(log_paths, key=mtime_or_zero, reverse=True)
    frames = []
    for path in log_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f'Warning: failed to read existing logs from {path}: {exc}')
            continue
        df = coerce_existing_logs_df(df)
        if not len(df):
            continue
        df['source_path'] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_point_key(dataset_name, condition, layer, strength):
    return (str(dataset_name), str(condition), int(layer), float(strength))


def prepare_reused_logs_for_export(reused_logs, probe_variant, condition, layer, strength, signed_strength, raw_vector_norm, steering_vector_norm, source_path):
    logs_df = reused_logs.copy()
    logs_df['probe_variant'] = probe_variant
    logs_df['condition'] = condition
    logs_df['layer'] = int(layer)
    logs_df['strength'] = float(strength)
    logs_df['signed_strength'] = float(signed_strength)
    logs_df['raw_vector_norm'] = raw_vector_norm
    logs_df['steering_vector_norm'] = steering_vector_norm
    logs_df['result_source'] = 'reused_existing'
    logs_df['reused_from_path'] = str(source_path)
    return logs_df.reset_index(drop=True)


def find_reusable_logs_for_point(existing_logs_df, dataset_packet, dataset_name, condition, layer, strength):
    if existing_logs_df is None or not len(existing_logs_df):
        return None, None

    expected_by_prompt_idx = {int(item['prompt_idx']): item for item in dataset_packet}
    expected_prompt_idx = set(expected_by_prompt_idx)
    if not expected_prompt_idx:
        return None, None

    point_df = existing_logs_df[
        (existing_logs_df['dataset'] == str(dataset_name))
        & (existing_logs_df['condition'] == str(condition))
        & (existing_logs_df['layer'] == int(layer))
        & np.isclose(existing_logs_df['strength'].astype(float), float(strength))
    ].copy()
    if not len(point_df):
        return None, None

    for source_path, source_df in point_df.groupby('source_path', sort=False):
        candidate_df = source_df[source_df['prompt_idx'].isin(expected_prompt_idx)].copy()
        if not len(candidate_df):
            continue
        candidate_df = candidate_df.drop_duplicates(subset=['prompt_idx'], keep='first').sort_values('prompt_idx').reset_index(drop=True)
        if len(candidate_df) != len(expected_prompt_idx):
            continue
        if set(candidate_df['prompt_idx']) != expected_prompt_idx:
            continue

        matches_expected = True
        for _, row in candidate_df.iterrows():
            expected = expected_by_prompt_idx[int(row['prompt_idx'])]
            for row_col, expected_value in [
                ('question', expected['pair']['question']),
                ('immediate', expected['pair']['immediate']),
                ('long_term', expected['pair']['long_term']),
                ('prompt', expected['prompt']),
            ]:
                actual_value = row[row_col] if row_col in row and pd.notna(row[row_col]) else ''
                if str(actual_value) != str(expected_value):
                    matches_expected = False
                    break
            if not matches_expected:
                break
        if matches_expected:
            return candidate_df, source_path

    return None, None


def run_experiment(config_overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    root = find_repo_root(Path.cwd())
    require_cuda_runtime(require_cuda=bool(cfg.get('require_cuda', True)))
    explicit_expanded_path = pick_first_existing([
        root / 'data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded_500.json',
        root / 'data/raw/temporal_scope/temporal_scope_explicit_expanded_500.json',
        root / 'data/raw/temporal_scope_AB_randomized/temporal_scope_explicit_expanded.json',
        root / 'data/raw/temporal_scope/temporal_scope_explicit_expanded.json',
    ])
    implicit_expanded_path = pick_first_existing([
        root / 'data/raw/temporal_scope_AB_randomized/temporal_scope_implicit_expanded_300.json',
        root / 'data/raw/temporal_scope/temporal_scope_implicit_expanded_300.json',
        root / 'data/raw/temporal_scope_AB_randomized/temporal_scope_implicit_expanded.json',
        root / 'data/raw/temporal_scope/temporal_scope_implicit_expanded.json',
    ])

    artifact_search_roots = cfg['artifact_search_roots'] or [
        root / 'results' / 'qwen3_32b' / 'question_only_probe_variations',
        root / 'results' / 'qwen3_32b',
        root / 'results',
        Path('/workspace/results/qwen3_32b/question_only_probe_variations'),
        Path('/workspace/results/qwen3_32b'),
        Path('/workspace/results'),
        Path('/workspace'),
    ]
    artifact_path, metadata_path = locate_latest_probe_artifacts(
        artifact_search_roots,
        artifact_path_override=cfg.get('artifact_path'),
        metadata_path_override=cfg.get('metadata_path'),
        required_train_regime=cfg.get('train_regime'),
    )
    probe_payload = load_probe_slice(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        train_regime=cfg['train_regime'],
        feature_name=cfg['feature_name'],
        vector_key=cfg['vector_key'],
        layers_to_test=[int(layer) for layer in cfg['layers_to_test']],
    )
    probe_metadata = probe_payload['metadata']

    model_name = cfg['model_name'] or probe_metadata.get('model_name', DEFAULT_CONFIG['model_name'])
    split_random_state = int(probe_metadata.get('split_random_state', cfg['split_random_state']))
    quick_mode = bool(cfg['quick_mode'])
    if quick_mode:
        max_prompts_explicit = 8 if cfg['max_prompts_explicit'] is None else int(cfg['max_prompts_explicit'])
        max_prompts_implicit = 20 if cfg['max_prompts_implicit'] is None else int(cfg['max_prompts_implicit'])
    else:
        max_prompts_explicit = None if cfg['max_prompts_explicit'] is None else int(cfg['max_prompts_explicit'])
        max_prompts_implicit = None if cfg['max_prompts_implicit'] is None else int(cfg['max_prompts_implicit'])

    expd_meta, explicit_pairs = load_pairs(explicit_expanded_path)
    impd_meta, implicit_pairs = load_pairs(implicit_expanded_path)
    explicit_sha = sha256(explicit_expanded_path)
    implicit_sha = sha256(implicit_expanded_path)
    if probe_metadata.get('explicit_expanded_sha256') and probe_metadata.get('explicit_expanded_sha256') != explicit_sha:
        raise ValueError(
            'Probe artifact explicit dataset SHA does not match the current explicit dataset file. '
            f"artifact={probe_metadata.get('explicit_expanded_sha256')} current={explicit_sha}"
        )
    if probe_metadata.get('implicit_expanded_sha256') and probe_metadata.get('implicit_expanded_sha256') != implicit_sha:
        raise ValueError(
            'Probe artifact implicit dataset SHA does not match the current implicit dataset file. '
            f"artifact={probe_metadata.get('implicit_expanded_sha256')} current={implicit_sha}"
        )

    explicit_train_question_idx = np.sort(probe_payload['explicit_train_question_indices'].astype(np.int64))
    explicit_test_question_idx = np.sort(probe_payload['explicit_test_question_indices'].astype(np.int64))
    implicit_train_question_idx = np.sort(probe_payload['implicit_train_question_indices'].astype(np.int64))
    implicit_test_question_idx = np.sort(probe_payload['implicit_test_question_indices'].astype(np.int64))

    if np.intersect1d(explicit_train_question_idx, explicit_test_question_idx).size != 0:
        raise ValueError('Artifact explicit train/test question indices overlap.')
    if np.intersect1d(implicit_train_question_idx, implicit_test_question_idx).size != 0:
        raise ValueError('Artifact implicit train/test question indices overlap.')

    dataset_packets = select_eval_pairs_for_train_regime(
        train_regime=cfg['train_regime'],
        explicit_pairs=explicit_pairs,
        implicit_pairs=implicit_pairs,
        explicit_test_question_idx=explicit_test_question_idx,
        implicit_test_question_idx=implicit_test_question_idx,
        max_prompts_explicit=max_prompts_explicit,
        max_prompts_implicit=max_prompts_implicit,
    )

    output_root = root / 'results' / cfg['output_root_name']
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.get('run_id') or time.strftime('%Y%m%d-%H%M%S'))
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / 'partial'
    partial_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    reuse_existing_results = bool(cfg.get('reuse_existing_results', True))
    reuse_result_search_roots = cfg.get('reuse_result_search_roots') or [
        output_dir,
        partial_dir,
        root / 'results' / 'qwen3_32b' / 'probe_artifact_steering_question_options_answer_vast',
        root / 'results' / 'qwen3_32b' / 'probe_artifact_steering_question_options_answer_colab',
        root / 'results' / 'qwen3_32b',
        root / 'results',
        Path('/workspace/results/qwen3_32b/probe_artifact_steering_question_options_answer_vast'),
        Path('/workspace/results/qwen3_32b/probe_artifact_steering_question_options_answer_colab'),
        Path('/workspace/results/qwen3_32b'),
        Path('/workspace/results'),
        Path('/workspace'),
    ]
    existing_logs_df = load_existing_result_logs(reuse_result_search_roots) if reuse_existing_results else pd.DataFrame()

    planned_points = []
    for dataset_name, dataset_packet in dataset_packets.items():
        planned_points.append({
            'dataset': dataset_name,
            'condition': 'baseline',
            'layer': -1,
            'strength': 0.0,
            'signed_strength': 0.0,
            'n_prompts': len(dataset_packet),
        })
        for layer in [int(layer) for layer in cfg['layers_to_test']]:
            for strength in [float(value) for value in cfg['strengths']]:
                planned_points.extend([
                    {
                        'dataset': dataset_name,
                        'condition': 'steer_long_term',
                        'layer': int(layer),
                        'strength': float(strength),
                        'signed_strength': float(strength),
                        'n_prompts': len(dataset_packet),
                    },
                    {
                        'dataset': dataset_name,
                        'condition': 'steer_immediate',
                        'layer': int(layer),
                        'strength': float(strength),
                        'signed_strength': -float(strength),
                        'n_prompts': len(dataset_packet),
                    },
                ])

    reusable_logs_by_point = {}
    reuse_coverage_rows = []
    for point in planned_points:
        reused_logs, reused_source_path = find_reusable_logs_for_point(
            existing_logs_df,
            dataset_packet=dataset_packets[point['dataset']],
            dataset_name=point['dataset'],
            condition=point['condition'],
            layer=point['layer'],
            strength=point['strength'],
        )
        point_key = make_point_key(point['dataset'], point['condition'], point['layer'], point['strength'])
        if reused_logs is not None:
            reusable_logs_by_point[point_key] = {
                'logs': reused_logs,
                'source_path': reused_source_path,
            }
        reuse_coverage_rows.append({
            **point,
            'reused': reused_logs is not None,
            'source_path': reused_source_path,
        })
    reuse_coverage_df = pd.DataFrame(reuse_coverage_rows)
    reuse_coverage_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_reuse_coverage_{run_id}.csv'
    reuse_coverage_df.to_csv(reuse_coverage_path, index=False)
    missing_points = reuse_coverage_df.loc[~reuse_coverage_df['reused']].copy()

    print('Repo root:', root)
    print('Device:', device)
    print('Model:', model_name)
    print('Expanded explicit dataset:', explicit_expanded_path)
    print('Expanded implicit dataset:', implicit_expanded_path)
    print('Probe artifact:', artifact_path)
    print('Probe metadata:', metadata_path)
    print('Dataset source:', cfg['dataset_source'])
    print('Use chat template:', bool(cfg['use_chat_template']))
    print('Disable thinking trace:', bool(cfg['disable_thinking_trace']))
    print('Quick mode:', quick_mode)
    print(
        'Artifact split metadata:',
        probe_metadata.get('explicit_split_strategy'),
        '/',
        probe_metadata.get('implicit_split_strategy'),
        '| version =',
        probe_metadata.get('artifact_format_version'),
    )
    print('Explicit train questions from artifact:', len(explicit_train_question_idx))
    print('Explicit test questions from artifact :', len(explicit_test_question_idx))
    print('Implicit train questions from artifact:', len(implicit_train_question_idx))
    print('Implicit test questions from artifact :', len(implicit_test_question_idx))
    for dataset_name, dataset_packet in dataset_packets.items():
        print(f'{dataset_name} eval size:', len(dataset_packet))
    print('Reuse existing results:', reuse_existing_results)
    print('Existing result search roots:', [str(Path(p)) for p in reuse_result_search_roots])
    print('Reusable points:', int(reuse_coverage_df['reused'].sum()), '/', len(reuse_coverage_df))
    for dataset_name, dataset_group in reuse_coverage_df.groupby('dataset'):
        print(
            f"  {dataset_name}: reusable {int(dataset_group['reused'].sum())}/{len(dataset_group)} points"
        )

    tokenizer = None
    model = None
    a_ids = []
    b_ids = []

    if len(missing_points):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {'trust_remote_code': True, 'torch_dtype': torch.float16}

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = model.to('cuda')
        model.eval()
        n_layers = len(model.model.layers)
        hidden_size = int(model.config.hidden_size)
        cuda_device = torch.cuda.current_device()
        cuda_props = torch.cuda.get_device_properties(cuda_device)
        print(
            '[cuda] confirmed GPU execution:',
            f'device=cuda:{cuda_device}',
            f'| name={cuda_props.name}',
            f'| total_memory_gb={cuda_props.total_memory / (1024 ** 3):.1f}',
            f'| device_count={torch.cuda.device_count()}',
        )
        print('Loaded model | n_layers =', n_layers, '| hidden_size =', hidden_size)
    else:
        print('All planned points already exist in prior artifacts. Skipping model load.')

    if probe_metadata.get('model_name') and probe_metadata.get('model_name') != model_name:
        print('Warning: probe artifact model_name differs from current model_name:', probe_metadata.get('model_name'))

    def move_batch_to_model_device(batch):
        if model is None:
            raise RuntimeError('Model is not loaded; no missing steering points required fresh generation.')
        model_device = next(model.parameters()).device
        return {key: value.to(model_device) for key, value in batch.items()}

    def get_single_token_ids_for_label(label):
        variants = [
            label,
            f' {label}',
            f'({label})',
            f' ({label})',
            f'{label})',
            f' {label})',
        ]
        ids = set()
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(token_ids[0])
        return sorted(ids)

    if tokenizer is not None:
        a_ids = get_single_token_ids_for_label('A')
        b_ids = get_single_token_ids_for_label('B')
        if not a_ids or not b_ids:
            raise RuntimeError(f'Could not identify single-token IDs for A/B. A={a_ids}, B={b_ids}')
        print('A token IDs:', a_ids)
        print('B token IDs:', b_ids)

    def maybe_register_steering_hook(layer, direction, strength, prompt_len, patch_decode_tokens=False):
        if model is None:
            raise RuntimeError('Model is not loaded; no missing steering points required fresh generation.')
        if layer is None or direction is None or abs(float(strength)) == 0:
            return None
        vector = torch.tensor(direction, device=next(model.parameters()).device, dtype=torch.float32)
        target_layer = model.model.layers[int(layer)]

        def steering_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_mod = hidden.clone()
            delta = (float(strength) * vector).to(hidden_mod.dtype)

            if hidden_mod.shape[1] >= prompt_len:
                if cfg['patch_prompt_last_only']:
                    hidden_mod[:, prompt_len - 1, :] = hidden_mod[:, prompt_len - 1, :] + delta
                else:
                    hidden_mod[:, :prompt_len, :] = hidden_mod[:, :prompt_len, :] + delta
            elif patch_decode_tokens and cfg['patch_generation_tokens']:
                if cfg['patch_prompt_last_only']:
                    hidden_mod[:, -1, :] = hidden_mod[:, -1, :] + delta
                else:
                    hidden_mod = hidden_mod + delta

            if isinstance(output, tuple):
                return (hidden_mod,) + output[1:]
            return hidden_mod

        return target_layer.register_forward_hook(steering_hook)

    def score_candidate_logprobs(packet_item, layer=None, direction=None, strength=0.0):
        if tokenizer is None or model is None:
            raise RuntimeError('Model/tokenizer are not loaded; cannot score missing steering points.')
        candidate_texts = [packet_item['candidate_immediate_text'], packet_item['candidate_long_term_text']]
        _, prompt_ids, _ = encode_prompt_and_full_sequence(
            tokenizer,
            packet_item['prompt'],
            continuation_text=None,
            use_chat_template=bool(cfg['use_chat_template']),
            disable_thinking_trace=bool(cfg['disable_thinking_trace']),
        )
        prompt_len = int(prompt_ids.shape[0])
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        full_sequence_ids = []
        seq_lengths = []
        candidate_token_counts = []
        for text in candidate_texts:
            _, _, full_ids = encode_prompt_and_full_sequence(
                tokenizer,
                packet_item['prompt'],
                continuation_text=text,
                use_chat_template=bool(cfg['use_chat_template']),
                disable_thinking_trace=bool(cfg['disable_thinking_trace']),
            )
            candidate_token_count = int(full_ids.shape[0] - prompt_len)
            if candidate_token_count <= 0:
                raise ValueError(f'Candidate continuation did not add any tokens: {text!r}')
            full_sequence_ids.append(full_ids)
            seq_lengths.append(int(full_ids.shape[0]))
            candidate_token_counts.append(candidate_token_count)
        max_seq_len = max(seq_lengths)

        input_ids = torch.full((len(full_sequence_ids), max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(full_sequence_ids), max_seq_len), dtype=torch.long)
        for row_idx, seq in enumerate(full_sequence_ids):
            seq_len = int(seq.shape[0])
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1

        batch = move_batch_to_model_device({'input_ids': input_ids, 'attention_mask': attention_mask})
        model_device = next(model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()
        hook = maybe_register_steering_hook(layer=layer, direction=direction, strength=strength, prompt_len=prompt_len, patch_decode_tokens=False)
        try:
            with torch.no_grad():
                with autocast_ctx:
                    outputs = model(**batch, use_cache=False)
            logits = outputs.logits.float()
        finally:
            if hook is not None:
                hook.remove()

        score_rows = []
        for row_idx, cand_len in enumerate(candidate_token_counts):
            token_logits = logits[row_idx, prompt_len - 1:prompt_len - 1 + cand_len, :]
            target_ids = batch['input_ids'][row_idx, prompt_len:prompt_len + cand_len]
            token_logprobs = torch.log_softmax(token_logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            score_rows.append({
                'sum_logprob': float(token_logprobs.sum().item()),
                'avg_logprob': float(token_logprobs.mean().item()),
                'token_count': int(cand_len),
            })

        immediate_scores, long_term_scores = score_rows
        return {
            'immediate_sum_logprob': immediate_scores['sum_logprob'],
            'immediate_avg_logprob': immediate_scores['avg_logprob'],
            'immediate_token_count': immediate_scores['token_count'],
            'long_term_sum_logprob': long_term_scores['sum_logprob'],
            'long_term_avg_logprob': long_term_scores['avg_logprob'],
            'long_term_token_count': long_term_scores['token_count'],
            'long_minus_immediate_sum_logprob': long_term_scores['sum_logprob'] - immediate_scores['sum_logprob'],
            'long_minus_immediate_avg_logprob': long_term_scores['avg_logprob'] - immediate_scores['avg_logprob'],
            'logprob_prefers_long_term': float(long_term_scores['avg_logprob'] >= immediate_scores['avg_logprob']),
        }

    def score_ab_logits(enc, layer=None, direction=None, strength=0.0):
        if model is None:
            raise RuntimeError('Model is not loaded; cannot score missing steering points.')
        prompt_len = int(enc['input_ids'].shape[1])
        model_device = next(model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()
        hook = maybe_register_steering_hook(layer=layer, direction=direction, strength=strength, prompt_len=prompt_len, patch_decode_tokens=False)
        try:
            with torch.no_grad():
                with autocast_ctx:
                    logits = model(**enc, use_cache=False).logits[0, -1, :].float()
        finally:
            if hook is not None:
                hook.remove()
        score_a = float(torch.max(logits[a_ids]).item())
        score_b = float(torch.max(logits[b_ids]).item())
        return score_a, score_b

    def run_preference(packet_item, layer=None, direction=None, strength=0.0):
        if tokenizer is None or model is None:
            raise RuntimeError('Model/tokenizer are not loaded; cannot generate missing steering points.')
        model_input_text = format_prompt_for_model(
            tokenizer,
            packet_item['prompt'],
            use_chat_template=bool(cfg['use_chat_template']),
            disable_thinking_trace=bool(cfg['disable_thinking_trace']),
        )
        enc = tokenizer(model_input_text, return_tensors='pt')
        enc = move_batch_to_model_device(enc)
        prompt_len = int(enc['input_ids'].shape[1])

        hook = maybe_register_steering_hook(
            layer=layer,
            direction=direction,
            strength=strength,
            prompt_len=prompt_len,
            patch_decode_tokens=cfg['patch_generation_tokens'],
        )
        try:
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=int(cfg['max_new_tokens']),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_ids = generated[0, enc['input_ids'].shape[1]:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            if hook is not None:
                hook.remove()

        parsed = parse_preference_completion(packet_item, continuation)
        if parsed['parsed_letter'] is None:
            score_a, score_b = score_ab_logits(enc, layer=layer, direction=direction, strength=strength)
            parsed_letter = 'A' if score_a >= score_b else 'B'
            parsed = {
                'parsed_letter': parsed_letter,
                'parsed_semantic': semantic_from_letter(packet_item, parsed_letter),
                'parse_method': 'ab_logit_fallback',
                'fallback_used': True,
                'score_a': score_a,
                'score_b': score_b,
            }

        logprob_stats = score_candidate_logprobs(packet_item, layer=layer, direction=direction, strength=strength)
        return {
            'continuation': continuation,
            **parsed,
            **logprob_stats,
        }

    def summarize_packet_predictions(pred_rows):
        if not pred_rows:
            return {
                'n_prompts': 0,
                'n_no_fallback_prompts': 0,
                'prop_choose_long_term': np.nan,
                'prop_choose_immediate': np.nan,
                'prop_choose_long_term_no_fallback': np.nan,
                'prop_choose_immediate_no_fallback': np.nan,
                'fallback_rate': np.nan,
                'direct_parse_rate': np.nan,
                'mean_long_minus_immediate_avg_logprob': np.nan,
                'mean_long_minus_immediate_sum_logprob': np.nan,
                'prop_logprob_prefers_long_term': np.nan,
            }

        semantics = np.array([row['parsed_semantic'] for row in pred_rows], dtype=object)
        methods = np.array([row['parse_method'] for row in pred_rows], dtype=object)
        no_fallback_mask = methods != 'ab_logit_fallback'
        semantics_no_fallback = semantics[no_fallback_mask]
        avg_logprob_margins = np.array([row.get('long_minus_immediate_avg_logprob', np.nan) for row in pred_rows], dtype=float)
        sum_logprob_margins = np.array([row.get('long_minus_immediate_sum_logprob', np.nan) for row in pred_rows], dtype=float)
        logprob_prefers_long_term = np.array([row.get('logprob_prefers_long_term', np.nan) for row in pred_rows], dtype=float)

        return {
            'n_prompts': int(len(pred_rows)),
            'n_no_fallback_prompts': int(no_fallback_mask.sum()),
            'prop_choose_long_term': float(np.mean(semantics == 'choose_long_term')),
            'prop_choose_immediate': float(np.mean(semantics == 'choose_immediate')),
            'prop_choose_long_term_no_fallback': float(np.mean(semantics_no_fallback == 'choose_long_term')) if len(semantics_no_fallback) else np.nan,
            'prop_choose_immediate_no_fallback': float(np.mean(semantics_no_fallback == 'choose_immediate')) if len(semantics_no_fallback) else np.nan,
            'fallback_rate': float(np.mean(methods == 'ab_logit_fallback')),
            'direct_parse_rate': float(np.mean(methods != 'ab_logit_fallback')),
            'mean_long_minus_immediate_avg_logprob': float(np.nanmean(avg_logprob_margins)),
            'mean_long_minus_immediate_sum_logprob': float(np.nanmean(sum_logprob_margins)),
            'prop_logprob_prefers_long_term': float(np.nanmean(logprob_prefers_long_term)),
        }

    probe_variant = {
        'mm_probe_vectors': 'mm',
        'wmm_probe_vectors': 'wmm',
    }.get(cfg['vector_key'], cfg['vector_key'])
    strength_schedule_by_layer = {int(layer): [float(value) for value in cfg['strengths']] for layer in cfg['layers_to_test']}
    signed_strength_grid = sorted({-float(value) for value in cfg['strengths']} | {float(value) for value in cfg['strengths']})

    summary_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_summary_{run_id}.csv'
    logs_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_logs_{run_id}.csv'
    probe_slice_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_probe_slice_{run_id}.csv'
    config_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_config_{run_id}.json'
    artifact_index_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_artifacts_{run_id}.csv'

    probe_slice_df = probe_payload['probe_slice_df']
    probe_slice_df.to_csv(probe_slice_path, index=False)

    def checkpoint_results(summary_parts, log_parts, note=''):
        summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
        logs_df = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame()
        if len(summary_df):
            summary_df.to_csv(summary_path, index=False)
        if len(logs_df):
            logs_df.to_csv(logs_path, index=False)
        if note:
            print(f'Checkpoint saved ({note}):', summary_path)
        else:
            print('Checkpoint saved:', summary_path)
        return summary_df, logs_df

    def write_partial_point(summary_row, logs_df, dataset_name, probe_variant_name, layer, strength, condition):
        pd.DataFrame([summary_row]).to_csv(
            partial_dir / f'{dataset_name}_{probe_variant_name}_layer{int(layer)}_strength{int(strength)}_{condition}_summary.csv',
            index=False,
        )
        logs_df.to_csv(
            partial_dir / f'{dataset_name}_{probe_variant_name}_layer{int(layer)}_strength{int(strength)}_{condition}_logs.csv',
            index=False,
        )

    def evaluate_baseline_packet(dataset_packet, progress_bar=None):
        rows = []
        for packet_item in dataset_packet:
            pred = run_preference(packet_item, layer=None, direction=None, strength=0.0)
            rows.append({
                'dataset': packet_item['dataset'],
                'probe_variant': 'baseline',
                'condition': 'baseline',
                'layer': -1,
                'strength': 0.0,
                'signed_strength': 0.0,
                'raw_vector_norm': np.nan,
                'steering_vector_norm': np.nan,
                'prompt_idx': packet_item['prompt_idx'],
                'question': packet_item['pair']['question'],
                'immediate': packet_item['pair']['immediate'],
                'long_term': packet_item['pair']['long_term'],
                'prompt': packet_item['prompt'],
                'result_source': 'computed_now',
                'reused_from_path': None,
                **pred,
            })
            if progress_bar is not None:
                progress_bar.update(1)
        return summarize_packet_predictions(rows), pd.DataFrame(rows)

    baseline_results = {}
    baseline_summary_rows = []
    baseline_logs_parts = []
    baseline_start = time.time()
    baseline_total_generations = sum(len(packet) for packet in dataset_packets.values())
    with tqdm(total=baseline_total_generations, desc='baseline generations', unit='gen') as baseline_pbar:
        for dataset_name, dataset_packet in dataset_packets.items():
            point_key = make_point_key(dataset_name, 'baseline', -1, 0.0)
            reused_payload = reusable_logs_by_point.get(point_key)
            if reused_payload is not None:
                baseline_logs = prepare_reused_logs_for_export(
                    reused_payload['logs'],
                    probe_variant='baseline',
                    condition='baseline',
                    layer=-1,
                    strength=0.0,
                    signed_strength=0.0,
                    raw_vector_norm=np.nan,
                    steering_vector_norm=np.nan,
                    source_path=reused_payload['source_path'],
                )
                baseline_summary = summarize_packet_predictions(baseline_logs.to_dict('records'))
                baseline_pbar.update(len(dataset_packet))
                print(f'[{dataset_name}] reused baseline from {reused_payload["source_path"]}')
                baseline_result_source = 'reused_existing'
                baseline_reused_from_path = str(reused_payload['source_path'])
            else:
                baseline_summary, baseline_logs = evaluate_baseline_packet(dataset_packet, progress_bar=baseline_pbar)
                baseline_result_source = 'computed_now'
                baseline_reused_from_path = None
            baseline_results[dataset_name] = {
                'summary': baseline_summary,
                'logs': baseline_logs,
            }
            baseline_logs_parts.append(baseline_logs)
            baseline_summary_rows.append({
                'dataset': dataset_name,
                'probe_variant': 'baseline',
                'condition': 'baseline',
                'layer': -1,
                'strength': 0.0,
                'signed_strength': 0.0,
                'raw_vector_norm': np.nan,
                'steering_vector_norm': np.nan,
                'baseline_prop_choose_long_term': baseline_summary['prop_choose_long_term'],
                'baseline_prop_choose_immediate': baseline_summary['prop_choose_immediate'],
                'baseline_prop_choose_long_term_no_fallback': baseline_summary['prop_choose_long_term_no_fallback'],
                'baseline_prop_choose_immediate_no_fallback': baseline_summary['prop_choose_immediate_no_fallback'],
                'baseline_mean_long_minus_immediate_avg_logprob': baseline_summary['mean_long_minus_immediate_avg_logprob'],
                'baseline_prop_logprob_prefers_long_term': baseline_summary['prop_logprob_prefers_long_term'],
                'delta_long_term_vs_baseline': 0.0,
                'delta_immediate_vs_baseline': 0.0,
                'delta_long_term_vs_baseline_no_fallback': 0.0,
                'delta_immediate_vs_baseline_no_fallback': 0.0,
                'delta_long_minus_immediate_avg_logprob_vs_baseline': 0.0,
                'delta_prop_logprob_prefers_long_term_vs_baseline': 0.0,
                'steering_success': np.nan,
                'steering_success_no_fallback': np.nan,
                'result_source': baseline_result_source,
                'reused_from_path': baseline_reused_from_path,
                **baseline_summary,
            })
    baseline_summary_df = pd.DataFrame(baseline_summary_rows)
    baseline_logs_df = pd.concat(baseline_logs_parts, ignore_index=True)
    checkpoint_results([baseline_summary_df], [baseline_logs_df], note='after baseline')
    print('Baseline generation time (min):', round((time.time() - baseline_start) / 60.0, 2))

    steering_summary_parts = [baseline_summary_df]
    steering_logs_parts = [baseline_logs_df]
    run_start = time.time()
    steering_total_generations = sum(len(packet) for packet in dataset_packets.values()) * len(cfg['layers_to_test']) * len(cfg['strengths']) * 2
    with tqdm(total=steering_total_generations, desc='steering generations', unit='gen') as steering_pbar:
        for dataset_name, dataset_packet in dataset_packets.items():
            for layer in [int(layer) for layer in cfg['layers_to_test']]:
                direction_record = probe_payload['direction_store'][int(layer)]
                for strength in [float(value) for value in cfg['strengths']]:
                    for condition, signed_strength, target_semantic in [
                        ('steer_long_term', float(strength), 'choose_long_term'),
                        ('steer_immediate', -float(strength), 'choose_immediate'),
                    ]:
                        point_key = make_point_key(dataset_name, condition, layer, strength)
                        reused_payload = reusable_logs_by_point.get(point_key)
                        steering_pbar.set_postfix(dataset=dataset_name, layer=layer, signed_strength=f'{signed_strength:g}')
                        if reused_payload is not None:
                            log_rows_df = prepare_reused_logs_for_export(
                                reused_payload['logs'],
                                probe_variant=probe_variant,
                                condition=condition,
                                layer=int(layer),
                                strength=float(strength),
                                signed_strength=float(signed_strength),
                                raw_vector_norm=float(direction_record['raw_norm']),
                                steering_vector_norm=float(direction_record['steering_norm']),
                                source_path=reused_payload['source_path'],
                            )
                            pred_rows = log_rows_df.to_dict('records')
                            steering_pbar.update(len(dataset_packet))
                            result_source = 'reused_existing'
                            reused_from_path = str(reused_payload['source_path'])
                        else:
                            pred_rows = []
                            log_rows = []
                            for packet_item in dataset_packet:
                                pred = run_preference(
                                    packet_item,
                                    layer=layer,
                                    direction=direction_record['steering_vector'],
                                    strength=signed_strength,
                                )
                                row = {
                                    'dataset': dataset_name,
                                    'probe_variant': probe_variant,
                                    'condition': condition,
                                    'layer': int(layer),
                                    'strength': float(strength),
                                    'signed_strength': float(signed_strength),
                                    'raw_vector_norm': float(direction_record['raw_norm']),
                                    'steering_vector_norm': float(direction_record['steering_norm']),
                                    'prompt_idx': packet_item['prompt_idx'],
                                    'question': packet_item['pair']['question'],
                                    'immediate': packet_item['pair']['immediate'],
                                    'long_term': packet_item['pair']['long_term'],
                                    'prompt': packet_item['prompt'],
                                    'result_source': 'computed_now',
                                    'reused_from_path': None,
                                    **pred,
                                }
                                pred_rows.append(row)
                                log_rows.append(row)
                                steering_pbar.update(1)
                            log_rows_df = pd.DataFrame(log_rows)
                            result_source = 'computed_now'
                            reused_from_path = None

                        summary = summarize_packet_predictions(pred_rows)
                        achieved = summary['prop_choose_long_term'] if target_semantic == 'choose_long_term' else summary['prop_choose_immediate']
                        achieved_no_fallback = summary['prop_choose_long_term_no_fallback'] if target_semantic == 'choose_long_term' else summary['prop_choose_immediate_no_fallback']
                        summary_row = {
                            'dataset': dataset_name,
                            'probe_variant': probe_variant,
                            'condition': condition,
                            'layer': int(layer),
                            'strength': float(strength),
                            'signed_strength': float(signed_strength),
                            'raw_vector_norm': float(direction_record['raw_norm']),
                            'steering_vector_norm': float(direction_record['steering_norm']),
                            'baseline_prop_choose_long_term': baseline_results[dataset_name]['summary']['prop_choose_long_term'],
                            'baseline_prop_choose_immediate': baseline_results[dataset_name]['summary']['prop_choose_immediate'],
                            'baseline_prop_choose_long_term_no_fallback': baseline_results[dataset_name]['summary']['prop_choose_long_term_no_fallback'],
                            'baseline_prop_choose_immediate_no_fallback': baseline_results[dataset_name]['summary']['prop_choose_immediate_no_fallback'],
                            'baseline_mean_long_minus_immediate_avg_logprob': baseline_results[dataset_name]['summary']['mean_long_minus_immediate_avg_logprob'],
                            'baseline_prop_logprob_prefers_long_term': baseline_results[dataset_name]['summary']['prop_logprob_prefers_long_term'],
                            'delta_long_term_vs_baseline': summary['prop_choose_long_term'] - baseline_results[dataset_name]['summary']['prop_choose_long_term'],
                            'delta_immediate_vs_baseline': summary['prop_choose_immediate'] - baseline_results[dataset_name]['summary']['prop_choose_immediate'],
                            'delta_long_term_vs_baseline_no_fallback': summary['prop_choose_long_term_no_fallback'] - baseline_results[dataset_name]['summary']['prop_choose_long_term_no_fallback'],
                            'delta_immediate_vs_baseline_no_fallback': summary['prop_choose_immediate_no_fallback'] - baseline_results[dataset_name]['summary']['prop_choose_immediate_no_fallback'],
                            'delta_long_minus_immediate_avg_logprob_vs_baseline': summary['mean_long_minus_immediate_avg_logprob'] - baseline_results[dataset_name]['summary']['mean_long_minus_immediate_avg_logprob'],
                            'delta_prop_logprob_prefers_long_term_vs_baseline': summary['prop_logprob_prefers_long_term'] - baseline_results[dataset_name]['summary']['prop_logprob_prefers_long_term'],
                            'steering_success': achieved,
                            'steering_success_no_fallback': achieved_no_fallback,
                            'result_source': result_source,
                            'reused_from_path': reused_from_path,
                            **summary,
                        }
                        steering_summary_parts.append(pd.DataFrame([summary_row]))
                        steering_logs_parts.append(log_rows_df)
                        write_partial_point(summary_row, log_rows_df, dataset_name, probe_variant, layer, strength, condition)
                        checkpoint_results(steering_summary_parts, steering_logs_parts, note=f'{dataset_name}-layer{layer}-{condition}-{strength:g}')
                        print(
                            f"[{dataset_name}] {probe_variant} layer={layer:02d} strength={strength:g} {condition} ({result_source}): "
                            f"long={summary['prop_choose_long_term']:.3f} imm={summary['prop_choose_immediate']:.3f} "
                            f"fallback={summary['fallback_rate']:.3f} logprob_margin={summary['mean_long_minus_immediate_avg_logprob']:.4f}"
                        )
    print('Steering generation time (min):', round((time.time() - run_start) / 60.0, 2))

    steering_summary_df, steering_logs_df = checkpoint_results(steering_summary_parts, steering_logs_parts, note='final')

    plot_rows = []
    for dataset_name in dataset_packets:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        panels = [
            ('prop_choose_long_term', 'Prop choose long-term', 'coolwarm', 0.0, 1.0),
            ('steering_success', 'Steering success', 'viridis', 0.0, 1.0),
            ('fallback_rate', 'Fallback rate', 'Greys', 0.0, 1.0),
            ('mean_long_minus_immediate_avg_logprob', 'Mean long-minus-immediate avg logprob', 'PiYG', -0.1, 0.1),
        ]
        for ax, (value_col, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            pivot = pivot_metric(
                steering_summary_df,
                dataset_name=dataset_name,
                value_col=value_col,
                signed_strength_grid=signed_strength_grid,
                layers_to_test=[int(layer) for layer in cfg['layers_to_test']],
            )
            image = draw_steering_heatmap(
                ax,
                pivot.to_numpy(dtype=float),
                f'{dataset_name}: {title}',
                x_labels=list(pivot.columns),
                y_labels=list(pivot.index),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                range_mode='data' if value_col == 'mean_long_minus_immediate_avg_logprob' else 'fixed',
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f'{model_name} | {probe_variant} steering | {cfg["train_regime"]} | {cfg["feature_name"]} | '
            f'Question-only probes, Question + Options + Answer eval',
            y=1.02,
        )
        plot_path = output_dir / f'mmraz_qwen3_32b_probe_artifact_steering_question_options_answer_{dataset_name}_heatmaps_{run_id}.png'
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        plot_rows.append({'artifact': f'{dataset_name}_heatmap_png', 'path': str(plot_path)})

    config_payload = {
        'run_id': run_id,
        'model_name': model_name,
        'device': str(next(model.parameters()).device) if model is not None else device,
        'model_loaded': model is not None,
        'probe_artifact_path': str(artifact_path),
        'probe_metadata_path': str(metadata_path) if metadata_path is not None else None,
        'train_regime': cfg['train_regime'],
        'feature_name': cfg['feature_name'],
        'vector_key': cfg['vector_key'],
        'layers_to_test': [int(layer) for layer in cfg['layers_to_test']],
        'strengths': [float(value) for value in cfg['strengths']],
        'signed_strengths': signed_strength_grid,
        'dataset_source': cfg['dataset_source'],
        'split_random_state': split_random_state,
        'explicit_split_strategy': probe_metadata.get('explicit_split_strategy', 'question_level_80_20'),
        'explicit_split_granularity': 'question',
        'implicit_split_strategy': probe_metadata.get('implicit_split_strategy', 'question_level_70_30'),
        'implicit_split_granularity': 'question',
        'probe_training_split_note': (
            'Steering uses vectors trained on question-only probes. '
            'For explicit-trained probes, evaluation uses explicit_test + implicit_full; '
            'for implicit-trained probes, evaluation uses implicit_test + explicit_full.'
        ),
        'prompt_format': 'question_then_options_then_answer_with_stripped_option_labels_eval',
        'probe_prompt_family': probe_metadata.get('prompt_family'),
        'use_chat_template': bool(cfg['use_chat_template']),
        'disable_thinking_trace': bool(cfg['disable_thinking_trace']),
        'patch_prompt_last_only': bool(cfg['patch_prompt_last_only']),
        'patch_generation_tokens': bool(cfg['patch_generation_tokens']),
        'do_sample': False,
        'max_new_tokens': int(cfg['max_new_tokens']),
        'explicit_expanded_path': str(explicit_expanded_path),
        'implicit_expanded_path': str(implicit_expanded_path),
        'explicit_expanded_sha256': explicit_sha,
        'implicit_expanded_sha256': implicit_sha,
        'probe_artifact_format_version': int(probe_metadata.get('artifact_format_version', 0)),
        'explicit_train_question_indices': explicit_train_question_idx.tolist(),
        'explicit_test_question_indices': explicit_test_question_idx.tolist(),
        'implicit_train_question_indices': implicit_train_question_idx.tolist(),
        'implicit_test_question_indices': implicit_test_question_idx.tolist(),
        'evaluation_datasets': {name: len(packet) for name, packet in dataset_packets.items()},
        'expd_metadata': expd_meta,
        'impd_metadata': impd_meta,
        'summary_path': str(summary_path),
        'logs_path': str(logs_path),
        'probe_slice_path': str(probe_slice_path),
        'partial_dir': str(partial_dir),
        'reuse_existing_results': reuse_existing_results,
        'reuse_result_search_roots': [str(Path(p)) for p in reuse_result_search_roots],
        'reuse_coverage_path': str(reuse_coverage_path),
        'reused_point_count': int(reuse_coverage_df['reused'].sum()),
        'total_planned_point_count': int(len(reuse_coverage_df)),
    }
    config_path.write_text(json.dumps(config_payload, indent=2) + '\n', encoding='utf-8')

    artifact_rows = [
        {'artifact': 'summary_csv', 'path': str(summary_path)},
        {'artifact': 'logs_csv', 'path': str(logs_path)},
        {'artifact': 'probe_slice_csv', 'path': str(probe_slice_path)},
        {'artifact': 'reuse_coverage_csv', 'path': str(reuse_coverage_path)},
        {'artifact': 'config_json', 'path': str(config_path)},
        *plot_rows,
    ]
    artifact_index_df = pd.DataFrame(artifact_rows)
    artifact_index_df.to_csv(artifact_index_path, index=False)

    print('Saved summary    :', summary_path)
    print('Saved logs       :', logs_path)
    print('Saved probe slice:', probe_slice_path)
    print('Saved reuse map  :', reuse_coverage_path)
    print('Saved config     :', config_path)
    print('Saved artifacts  :', artifact_index_path)
    print('Output dir       :', output_dir)

    return {
        'output_dir': output_dir,
        'summary_path': summary_path,
        'logs_path': logs_path,
        'probe_slice_path': probe_slice_path,
        'reuse_coverage_path': reuse_coverage_path,
        'config_path': config_path,
        'artifact_index_path': artifact_index_path,
        'summary_df': steering_summary_df,
        'logs_df': steering_logs_df,
        'probe_slice_df': probe_slice_df,
        'baseline_summary_df': baseline_summary_df,
        'reuse_coverage_df': reuse_coverage_df,
        'artifact_index_df': artifact_index_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Qwen3-32B probe-artifact steering experiment on question+options+answer prompts.'
    )
    parser.add_argument('--train-regime', type=str, default=None, help='Probe train regime to load, e.g. explicit_train_only or implicit_train_only.')
    parser.add_argument('--feature-name', type=str, default=None, help='Probe feature name, e.g. mean_answer_tokens or last_answer_token.')
    parser.add_argument('--vector-key', type=str, default=None, help='Artifact vector key, e.g. mm_probe_vectors or wmm_probe_vectors.')
    parser.add_argument('--artifact-path', type=str, default=None, help='Optional explicit probe artifact .npz path.')
    parser.add_argument('--metadata-path', type=str, default=None, help='Optional explicit probe metadata .json path.')
    parser.add_argument('--quick-mode', action='store_true', help='Use small eval subsets for faster debugging.')
    parser.add_argument('--max-prompts-explicit', type=int, default=None, help='Optional cap for explicit evaluation prompts.')
    parser.add_argument('--max-prompts-implicit', type=int, default=None, help='Optional cap for implicit evaluation prompts.')
    parser.add_argument('--output-root-name', type=str, default=None, help='Optional output root relative to repo results/.')
    parser.add_argument('--run-id', type=str, default=None, help='Optional explicit run id for resuming into the same output directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.train_regime is not None:
        overrides['train_regime'] = args.train_regime
    if args.feature_name is not None:
        overrides['feature_name'] = args.feature_name
    if args.vector_key is not None:
        overrides['vector_key'] = args.vector_key
    if args.artifact_path is not None:
        overrides['artifact_path'] = args.artifact_path
    if args.metadata_path is not None:
        overrides['metadata_path'] = args.metadata_path
    if args.max_prompts_explicit is not None:
        overrides['max_prompts_explicit'] = int(args.max_prompts_explicit)
    if args.max_prompts_implicit is not None:
        overrides['max_prompts_implicit'] = int(args.max_prompts_implicit)
    if args.output_root_name is not None:
        overrides['output_root_name'] = args.output_root_name
    if args.run_id is not None:
        overrides['run_id'] = args.run_id
    if args.quick_mode:
        overrides['quick_mode'] = True
    run_experiment(overrides)


if __name__ == '__main__':
    main()
