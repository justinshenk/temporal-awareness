#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pairs(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and 'pairs' in data:
        return data.get('metadata', {}), data['pairs']
    return {}, data

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def train_mm_probe(X_train, y_train):
    """Train mean-mass (difference-in-means) probe direction."""
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)
    direction = mu1 - mu0
    return direction


def write_status(status_path: Path, payload: dict):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    job_config = json.loads(config_path.read_text())
    ROOT = Path(job_config.get('stage_root', config_path.parent)).resolve()
    explicit_path = Path(job_config['explicit_path']).resolve()
    implicit_path = Path(job_config['implicit_path']).resolve()
    LOCAL_SAVE_DIR = Path(job_config['results_dir']).resolve()
    LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = LOCAL_SAVE_DIR / 'remote_job_status.json'
    write_status(status_path, {
        'state': 'starting',
        'config_path': str(config_path),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })

    expd_meta, explicit_pairs_expd = load_pairs(explicit_path)
    impd_meta, implicit_pairs_expd = load_pairs(implicit_path)
    print('Expanded explicit dataset:', explicit_path)
    print('Expanded implicit dataset:', implicit_path)
    print('Expanded explicit metadata:', expd_meta)
    print('Expanded implicit metadata:', impd_meta)
    print('Expanded explicit sha256:', sha256(explicit_path))
    print('Expanded implicit sha256:', sha256(implicit_path))

    STEERING_MODEL_NAME = job_config.get('model_name', 'Qwen/Qwen2.5-14B-Instruct')
    STEERING_DATASET_SOURCE = job_config.get('dataset_source', 'expanded')
    QUICK_MODE = bool(job_config.get('quick_mode', False))
    PAIR_SPLIT_RANDOM_STATE = int(job_config.get('pair_split_random_state', 42))
    PATCH_PROMPT_LAST_ONLY = bool(job_config.get('patch_prompt_last_only', True))
    PATCH_GENERATION_TOKENS = bool(job_config.get('patch_generation_tokens', True))
    NORMALIZE_STEERING_VECTORS = bool(job_config.get('normalize_steering_vectors', True))
    STRIP_OPTION_LETTERS_FOR_PROBE_TRAINING = bool(job_config.get('strip_option_letters_for_probe_training', True))
    STEERING_WHITEN_REG = float(job_config.get('steering_whiten_reg', 1e-2))
    STEERING_MAX_NEW_TOKENS = int(job_config.get('max_new_tokens', 32))
    STEERING_BATCH_SIZE = int(job_config.get('steering_batch_size', 1))
    LOCAL_SAVE_DIR = Path(job_config['results_dir'])

    if QUICK_MODE:
        strength_grid_qwen = [1.0, 2.0, 4.0, 8.0, 16.0]
        max_prompts_explicit_qwen = 8
        max_prompts_implicit_qwen = 20
    else:
        strength_grid_qwen = [1.0, 2.0, 4.0, 8.0, 16.0]
        max_prompts_explicit_qwen = None
        max_prompts_implicit_qwen = None

    # Steering sweep: evaluate only the selected layers with the requested strength grid.
    strength_schedule_by_layer_qwen = {
        int(layer): [float(v) for v in values]
        for layer, values in job_config.get('strength_schedule_by_layer', {16: [1.0, 2.0, 4.0, 8.0, 16.0], 20: [1.0, 2.0, 4.0, 8.0, 16.0], 24: [1.0, 2.0, 4.0, 8.0, 16.0], 28: [1.0, 2.0, 4.0, 8.0, 16.0], 32: [1.0, 2.0, 4.0, 8.0, 16.0]}).items()
    }

    print('Steering model:', STEERING_MODEL_NAME)
    print('Dataset source:', STEERING_DATASET_SOURCE)
    print('Quick mode:', QUICK_MODE)
    print('Normalize steering vectors:', NORMALIZE_STEERING_VECTORS)
    print('Patch prompt last token only:', PATCH_PROMPT_LAST_ONLY)
    print('Patch generation tokens:', PATCH_GENERATION_TOKENS)
    print('Strength grid:', strength_grid_qwen)
    print('Strength schedule by layer:', strength_schedule_by_layer_qwen)

    if STEERING_DATASET_SOURCE == 'expanded':
        explicit_pairs_qwen_all = explicit_pairs_expd
        implicit_pairs_qwen_all = implicit_pairs_expd
    else:
        explicit_pairs_qwen_all = explicit_pairs
        implicit_pairs_qwen_all = implicit_pairs

    pair_indices_qwen = np.arange(len(explicit_pairs_qwen_all))
    exp_pair_train_idx_qwen, exp_pair_test_idx_qwen = train_test_split(
        pair_indices_qwen,
        test_size=0.2,
        random_state=PAIR_SPLIT_RANDOM_STATE,
        shuffle=True,
    )

    explicit_train_pairs_qwen = [explicit_pairs_qwen_all[i] for i in exp_pair_train_idx_qwen]
    explicit_test_pairs_qwen_full = [explicit_pairs_qwen_all[i] for i in exp_pair_test_idx_qwen]
    implicit_pairs_qwen_full = list(implicit_pairs_qwen_all)

    explicit_train_eval_pairs_qwen = (
        explicit_train_pairs_qwen[:max_prompts_explicit_qwen]
        if max_prompts_explicit_qwen is not None else explicit_train_pairs_qwen
    )
    explicit_eval_pairs_qwen = (
        explicit_test_pairs_qwen_full[:max_prompts_explicit_qwen]
        if max_prompts_explicit_qwen is not None else explicit_test_pairs_qwen_full
    )
    implicit_eval_pairs_qwen = (
        implicit_pairs_qwen_full[:max_prompts_implicit_qwen]
        if max_prompts_implicit_qwen is not None else implicit_pairs_qwen_full
    )

    print('Qwen steering split:')
    print('  explicit train pairs:', len(explicit_train_pairs_qwen), '| eval subset =', len(explicit_train_eval_pairs_qwen))
    print('  explicit test pairs :', len(explicit_test_pairs_qwen_full), '| eval subset =', len(explicit_eval_pairs_qwen))
    print('  implicit pairs      :', len(implicit_pairs_qwen_full), '| eval subset =', len(implicit_eval_pairs_qwen))


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
        'this', 'that', 'it', 'we', 'what', 'which', 'will', 'would', 'should', 'can', 'our', 'your'
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


    def build_probe_training_examples_qwen(pairs, strip_option_letters=True):
        examples = []
        labels = []
        for pair in pairs:
            option_payload = get_pair_option_payload(pair)
            prompt = format_binary_prompt_qwen(
                pair['question'],
                option_payload['option_a_text'],
                option_payload['option_b_text'],
            )
            immediate_continuation = option_payload['candidate_immediate_text']
            long_term_continuation = option_payload['candidate_long_term_text']
            if not strip_option_letters:
                immediate_continuation = pair['immediate']
                long_term_continuation = pair['long_term']
            examples.append({
                'prompt': prompt,
                'continuation': immediate_continuation,
                'label': 0,
            })
            labels.append(0)
            examples.append({
                'prompt': prompt,
                'continuation': long_term_continuation,
                'label': 1,
            })
            labels.append(1)
        return examples, np.array(labels, dtype=np.int64)


    def normalize_direction(direction):
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            return direction.astype(np.float32), norm
        return (direction / norm).astype(np.float32), norm


    def train_whitened_mm_probe_low_rank(X_train, y_train, reg=1e-2):
        mu0 = X_train[y_train == 0].mean(axis=0)
        mu1 = X_train[y_train == 1].mean(axis=0)
        mm_direction = (mu1 - mu0).astype(np.float64)

        mean_train = X_train.mean(axis=0, keepdims=True).astype(np.float64)
        Xc = X_train.astype(np.float64) - mean_train
        n_samples, d_model_qwen = Xc.shape
        denom = max(n_samples - 1, 1)
        avg_var = float(np.sum(Xc * Xc) / (denom * max(d_model_qwen, 1))) if d_model_qwen else 1.0
        lam = float(reg * avg_var) if avg_var > 0 else float(reg)
        gram = (Xc @ Xc.T) / denom
        system = gram + lam * np.eye(n_samples, dtype=np.float64)
        alpha = np.linalg.solve(system, Xc @ mm_direction)
        effective_direction = (mm_direction / lam) - (Xc.T @ alpha) / (lam * denom)

        try:
            cond = float(np.linalg.cond(system))
        except Exception:
            cond = float('nan')

        return {
            'mean_train': mean_train.reshape(-1).astype(np.float32),
            'mm_direction': mm_direction.astype(np.float32),
            'effective_direction': effective_direction.astype(np.float32),
            'reg_lambda': float(lam),
            'sample_space_condition_number': cond,
        }


    qwen_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Qwen device:', qwen_device)

    qwen_tokenizer = AutoTokenizer.from_pretrained(STEERING_MODEL_NAME, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_model_kwargs = {
        'trust_remote_code': True,
    }
    if qwen_device == 'cuda':
        qwen_model_kwargs['device_map'] = 'auto'
        qwen_model_kwargs['torch_dtype'] = torch.float16
    elif qwen_device == 'mps':
        qwen_model_kwargs['torch_dtype'] = torch.float16
    else:
        qwen_model_kwargs['torch_dtype'] = torch.float32

    qwen_model = AutoModelForCausalLM.from_pretrained(STEERING_MODEL_NAME, **qwen_model_kwargs)
    if qwen_device != 'cuda':
        qwen_model = qwen_model.to(qwen_device)
    qwen_model.eval()
    qwen_n_layers = len(qwen_model.model.layers)
    print('Loaded', STEERING_MODEL_NAME, '| n_layers =', qwen_n_layers, '| hidden_size =', qwen_model.config.hidden_size)


    def get_single_token_ids_for_label_qwen(label):
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
            tok_ids = qwen_tokenizer.encode(variant, add_special_tokens=False)
            if len(tok_ids) == 1:
                ids.add(tok_ids[0])
        return sorted(ids)


    QWEN_A_IDS = get_single_token_ids_for_label_qwen('A')
    QWEN_B_IDS = get_single_token_ids_for_label_qwen('B')
    print('Qwen A token IDs:', QWEN_A_IDS)
    print('Qwen B token IDs:', QWEN_B_IDS)
    if not QWEN_A_IDS or not QWEN_B_IDS:
        raise RuntimeError(f'Could not identify single-token IDs for A/B. A={QWEN_A_IDS}, B={QWEN_B_IDS}')


    def _move_batch_to_qwen_device(batch):
        moved = {}
        model_device = next(qwen_model.parameters()).device
        for key, value in batch.items():
            moved[key] = value.to(model_device)
        return moved


    def extract_mean_answer_token_activations_qwen(prompts, batch_size=1):
        activations = [[] for _ in range(qwen_n_layers)]
        model_device = next(qwen_model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            enc = qwen_tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
            enc = _move_batch_to_qwen_device(enc)
            attention_mask = enc['attention_mask']
            last_positions = attention_mask.sum(dim=1) - 1

            with torch.no_grad():
                with autocast_ctx:
                    outputs = qwen_model(**enc, output_hidden_states=True, use_cache=False)

            hidden_states = outputs.hidden_states[1:]
            for layer, hidden in enumerate(hidden_states):
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                last_hidden = hidden[batch_idx, last_positions, :].detach().float().cpu().numpy()
                activations[layer].append(last_hidden)

        return [np.concatenate(parts, axis=0) for parts in activations]


    def extract_mean_answer_token_activations_from_examples_qwen(examples, batch_size=1):
        activations = [[] for _ in range(qwen_n_layers)]
        model_device = next(qwen_model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()
        pad_id = qwen_tokenizer.pad_token_id if qwen_tokenizer.pad_token_id is not None else qwen_tokenizer.eos_token_id

        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start:start + batch_size]
            prompt_ids_batch = []
            continuation_ids_batch = []
            seq_lengths = []
            answer_spans = []

            for example in batch_examples:
                prompt_ids = qwen_tokenizer(example['prompt'], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
                continuation_ids = qwen_tokenizer(example['continuation'], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
                if continuation_ids.numel() == 0:
                    raise ValueError(f"Empty continuation for training example: {example!r}")
                prompt_ids_batch.append(prompt_ids)
                continuation_ids_batch.append(continuation_ids)
                seq_lengths.append(int(prompt_ids.shape[0] + continuation_ids.shape[0]))

            max_seq_len = max(seq_lengths)
            input_ids = torch.full((len(batch_examples), max_seq_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch_examples), max_seq_len), dtype=torch.long)

            for row_idx, (prompt_ids, continuation_ids) in enumerate(zip(prompt_ids_batch, continuation_ids_batch)):
                seq = torch.cat([prompt_ids, continuation_ids], dim=0)
                seq_len = int(seq.shape[0])
                answer_start = int(prompt_ids.shape[0])
                answer_end = seq_len
                input_ids[row_idx, :seq_len] = seq
                attention_mask[row_idx, :seq_len] = 1
                answer_spans.append((answer_start, answer_end))

            batch = _move_batch_to_qwen_device({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })

            with torch.no_grad():
                with autocast_ctx:
                    outputs = qwen_model(**batch, output_hidden_states=True, use_cache=False)

            hidden_states = outputs.hidden_states[1:]
            for layer, hidden in enumerate(hidden_states):
                pooled_rows = []
                for row_idx, (answer_start, answer_end) in enumerate(answer_spans):
                    pooled = hidden[row_idx, answer_start:answer_end, :].mean(dim=0)
                    pooled_rows.append(pooled.detach().float().cpu().numpy())
                activations[layer].append(np.stack(pooled_rows, axis=0))

        return [np.concatenate(parts, axis=0) for parts in activations]


    def get_pair_option_payload(pair):
        immediate_letter = extract_option_letter(pair['immediate'])
        long_term_letter = extract_option_letter(pair['long_term'])
        if immediate_letter == long_term_letter:
            raise ValueError(f"Immediate and long-term options cannot share the same letter: {pair!r}")

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


    def format_binary_prompt_qwen(question, option_a_text, option_b_text):
        del question
        option_a_text = strip_option_label(option_a_text)
        option_b_text = strip_option_label(option_b_text)
        return (
            'Options:\n'
            f'{option_a_text}\n'
            f'{option_b_text}\n'
            'Answer:\n'
        )


    def build_dataset_packet(dataset_name, pairs):
        packet = []
        for idx, pair in enumerate(pairs):
            option_payload = get_pair_option_payload(pair)
            option_a_text = option_payload['option_a_text']
            option_b_text = option_payload['option_b_text']
            prompt = format_binary_prompt_qwen(pair['question'], option_a_text, option_b_text)

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
                'option_a_text_stripped': strip_option_label(option_a_text),
                'option_b_text_stripped': strip_option_label(option_b_text),
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

    def parse_ab_from_text_qwen(text):
        t = (text or '').strip()
        if not t:
            return None

        strong = re.match(r'^[\s\n]*[\(\[]?\s*([ABab12])\s*[\)\].,:;\- ]?', t)
        if strong:
            ch = strong.group(1).upper()
            return {'1': 'A', '2': 'B'}.get(ch, ch)

        up = t.upper()
        patterns = [
            r'\b(?:ANSWER\s*[:\-]?\s*)([AB12])\b',
            r'\bOPTION\s*([AB12])\b',
            r'\bCHOOSE\s*([AB12])\b',
            r'\(([AB12])\)',
        ]
        for pattern in patterns:
            match = re.search(pattern, up)
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

        parsed_letter = parse_ab_from_text_qwen(continuation)
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


    def maybe_register_qwen_steering_hook(layer, direction, strength, prompt_len, patch_decode_tokens=False):
        if layer is None or direction is None or abs(float(strength)) == 0:
            return None

        model_device = next(qwen_model.parameters()).device
        vector = torch.tensor(direction, device=model_device, dtype=torch.float32)
        target_layer = qwen_model.model.layers[layer]

        def steering_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_mod = hidden.clone()
            delta = (float(strength) * vector).to(hidden_mod.dtype)

            if hidden_mod.shape[1] >= prompt_len:
                if PATCH_PROMPT_LAST_ONLY:
                    hidden_mod[:, prompt_len - 1, :] = hidden_mod[:, prompt_len - 1, :] + delta
                else:
                    hidden_mod[:, :prompt_len, :] = hidden_mod[:, :prompt_len, :] + delta
            elif patch_decode_tokens:
                if PATCH_PROMPT_LAST_ONLY:
                    hidden_mod[:, -1, :] = hidden_mod[:, -1, :] + delta
                else:
                    hidden_mod = hidden_mod + delta

            if isinstance(output, tuple):
                return (hidden_mod,) + output[1:]
            return hidden_mod

        return target_layer.register_forward_hook(steering_hook)


    def score_candidate_logprobs_qwen(packet_item, layer=None, direction=None, strength=0.0):
        prompt_ids = qwen_tokenizer(packet_item['prompt'], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        candidate_texts = [packet_item['candidate_immediate_text'], packet_item['candidate_long_term_text']]
        candidate_ids = [
            qwen_tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            for text in candidate_texts
        ]

        prompt_len = int(prompt_ids.shape[0])
        pad_id = qwen_tokenizer.pad_token_id if qwen_tokenizer.pad_token_id is not None else qwen_tokenizer.eos_token_id
        batch_size = len(candidate_ids)
        seq_lengths = [prompt_len + int(cand.shape[0]) for cand in candidate_ids]
        max_seq_len = max(seq_lengths)

        input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        candidate_token_counts = []
        for row_idx, cand_ids in enumerate(candidate_ids):
            seq = torch.cat([prompt_ids, cand_ids], dim=0)
            seq_len = int(seq.shape[0])
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1
            candidate_token_counts.append(int(cand_ids.shape[0]))

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        batch = _move_batch_to_qwen_device(batch)
        model_device = next(qwen_model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()

        hook = maybe_register_qwen_steering_hook(
            layer=layer,
            direction=direction,
            strength=strength,
            prompt_len=prompt_len,
            patch_decode_tokens=False,
        )
        try:
            with torch.no_grad():
                with autocast_ctx:
                    outputs = qwen_model(**batch, use_cache=False)
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


    def score_ab_logits_qwen(enc, layer=None, direction=None, strength=0.0):
        prompt_len = int(enc['input_ids'].shape[1])
        model_device = next(qwen_model.parameters()).device
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if model_device.type == 'cuda' else nullcontext()
        hook = maybe_register_qwen_steering_hook(
            layer=layer,
            direction=direction,
            strength=strength,
            prompt_len=prompt_len,
            patch_decode_tokens=False,
        )
        try:
            with torch.no_grad():
                with autocast_ctx:
                    logits = qwen_model(**enc, use_cache=False).logits[0, -1, :].float()
        finally:
            if hook is not None:
                hook.remove()
        score_a = float(torch.max(logits[QWEN_A_IDS]).item())
        score_b = float(torch.max(logits[QWEN_B_IDS]).item())
        return score_a, score_b


    def run_qwen_preference(packet_item, layer=None, direction=None, strength=0.0, max_new_tokens=6):
        enc = qwen_tokenizer(packet_item['prompt'], return_tensors='pt')
        enc = _move_batch_to_qwen_device(enc)
        prompt_len = int(enc['input_ids'].shape[1])

        hook = maybe_register_qwen_steering_hook(
            layer=layer,
            direction=direction,
            strength=strength,
            prompt_len=prompt_len,
            patch_decode_tokens=PATCH_GENERATION_TOKENS,
        )
        try:
            with torch.no_grad():
                generated = qwen_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=qwen_tokenizer.pad_token_id,
                    eos_token_id=qwen_tokenizer.eos_token_id,
                )
            new_ids = generated[0, enc['input_ids'].shape[1]:]
            continuation = qwen_tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            if hook is not None:
                hook.remove()

        parsed = parse_preference_completion(packet_item, continuation)

        if parsed['parsed_letter'] is None:
            score_a, score_b = score_ab_logits_qwen(enc, layer=layer, direction=direction, strength=strength)
            parsed_letter = 'A' if score_a >= score_b else 'B'
            parsed = {
                'parsed_letter': parsed_letter,
                'parsed_semantic': semantic_from_letter(packet_item, parsed_letter),
                'parse_method': 'ab_logit_fallback',
                'fallback_used': True,
                'score_a': score_a,
                'score_b': score_b,
            }

        logprob_stats = score_candidate_logprobs_qwen(packet_item, layer=layer, direction=direction, strength=strength)

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


    def evaluate_baseline_packet(dataset_packet, max_new_tokens=6):
        log_rows = []
        for packet_item in dataset_packet:
            pred = run_qwen_preference(packet_item, layer=None, direction=None, strength=0.0, max_new_tokens=max_new_tokens)
            log_rows.append({
                'dataset': packet_item['dataset'],
                'probe_variant': 'baseline',
                'condition': 'baseline',
                'layer': -1,
                'strength': 0.0,
                'signed_strength': 0.0,
                'prompt_idx': packet_item['prompt_idx'],
                'question': packet_item['pair']['question'],
                'immediate': packet_item['pair']['immediate'],
                'long_term': packet_item['pair']['long_term'],
                'prompt': packet_item['prompt'],
                **pred,
            })
        return summarize_packet_predictions(log_rows), pd.DataFrame(log_rows)


    def evaluate_probe_variant_on_packet(dataset_packet, probe_variant, direction_store, baseline_summary, layers_to_test, strength_grid, max_new_tokens=6, partial_summary_path=None, partial_logs_path=None, strength_schedule_by_layer=None):
        summary_rows = []
        log_rows = []

        for layer in layers_to_test:
            raw_vector_norm = float(direction_store[layer]['raw_norm'])
            steering_vector_norm = float(direction_store[layer]['steering_norm'])
            direction = direction_store[layer]['steering_vector']
            active_strengths = strength_schedule_by_layer.get(layer, strength_grid) if strength_schedule_by_layer is not None else strength_grid

            for strength in active_strengths:
                for condition, signed_strength, target_semantic in [
                    ('steer_long_term', float(strength), 'choose_long_term'),
                    ('steer_immediate', -float(strength), 'choose_immediate'),
                ]:
                    pred_rows = []
                    for packet_item in dataset_packet:
                        pred = run_qwen_preference(
                            packet_item,
                            layer=layer,
                            direction=direction,
                            strength=signed_strength,
                            max_new_tokens=max_new_tokens,
                        )
                        full_row = {
                            'dataset': packet_item['dataset'],
                            'probe_variant': probe_variant,
                            'condition': condition,
                            'layer': int(layer),
                            'strength': float(strength),
                            'signed_strength': float(signed_strength),
                            'raw_vector_norm': raw_vector_norm,
                            'steering_vector_norm': steering_vector_norm,
                            'prompt_idx': packet_item['prompt_idx'],
                            'question': packet_item['pair']['question'],
                            'immediate': packet_item['pair']['immediate'],
                            'long_term': packet_item['pair']['long_term'],
                            'prompt': packet_item['prompt'],
                            **pred,
                        }
                        pred_rows.append(full_row)
                        log_rows.append(full_row)

                    summary = summarize_packet_predictions(pred_rows)
                    achieved = summary['prop_choose_long_term'] if target_semantic == 'choose_long_term' else summary['prop_choose_immediate']
                    achieved_no_fallback = summary['prop_choose_long_term_no_fallback'] if target_semantic == 'choose_long_term' else summary['prop_choose_immediate_no_fallback']
                    summary_rows.append({
                        'dataset': dataset_packet[0]['dataset'] if dataset_packet else 'unknown',
                        'probe_variant': probe_variant,
                        'condition': condition,
                        'layer': int(layer),
                        'strength': float(strength),
                        'signed_strength': float(signed_strength),
                        'raw_vector_norm': raw_vector_norm,
                        'steering_vector_norm': steering_vector_norm,
                        'baseline_prop_choose_long_term': baseline_summary['prop_choose_long_term'],
                        'baseline_prop_choose_immediate': baseline_summary['prop_choose_immediate'],
                        'baseline_prop_choose_long_term_no_fallback': baseline_summary.get('prop_choose_long_term_no_fallback', np.nan),
                        'baseline_prop_choose_immediate_no_fallback': baseline_summary.get('prop_choose_immediate_no_fallback', np.nan),
                        'baseline_mean_long_minus_immediate_avg_logprob': baseline_summary.get('mean_long_minus_immediate_avg_logprob', np.nan),
                        'baseline_prop_logprob_prefers_long_term': baseline_summary.get('prop_logprob_prefers_long_term', np.nan),
                        'delta_long_term_vs_baseline': summary['prop_choose_long_term'] - baseline_summary['prop_choose_long_term'],
                        'delta_immediate_vs_baseline': summary['prop_choose_immediate'] - baseline_summary['prop_choose_immediate'],
                        'delta_long_term_vs_baseline_no_fallback': summary['prop_choose_long_term_no_fallback'] - baseline_summary.get('prop_choose_long_term_no_fallback', np.nan),
                        'delta_immediate_vs_baseline_no_fallback': summary['prop_choose_immediate_no_fallback'] - baseline_summary.get('prop_choose_immediate_no_fallback', np.nan),
                        'delta_long_minus_immediate_avg_logprob_vs_baseline': summary['mean_long_minus_immediate_avg_logprob'] - baseline_summary.get('mean_long_minus_immediate_avg_logprob', np.nan),
                        'delta_prop_logprob_prefers_long_term_vs_baseline': summary['prop_logprob_prefers_long_term'] - baseline_summary.get('prop_logprob_prefers_long_term', np.nan),
                        'steering_success': achieved,
                        'steering_success_no_fallback': achieved_no_fallback,
                        **summary,
                    })

                    if partial_summary_path is not None:
                        pd.DataFrame(summary_rows).to_csv(partial_summary_path, index=False)
                    if partial_logs_path is not None:
                        pd.DataFrame(log_rows).to_csv(partial_logs_path, index=False)

                    print(
                        f"[{dataset_packet[0]['dataset'] if dataset_packet else 'unknown'}] {probe_variant} layer={layer:02d} "
                        f"strength={strength:g} {condition}: long={summary['prop_choose_long_term']:.3f} "
                        f"imm={summary['prop_choose_immediate']:.3f} fallback={summary['fallback_rate']:.3f} "
                        f"logprob_margin={summary['mean_long_minus_immediate_avg_logprob']:.4f}"
                    )

        return pd.DataFrame(summary_rows), pd.DataFrame(log_rows)

    train_examples_qwen, y_train_qwen = build_probe_training_examples_qwen(
        explicit_train_pairs_qwen,
        strip_option_letters=STRIP_OPTION_LETTERS_FOR_PROBE_TRAINING,
    )
    print('Qwen probe-training examples:', len(train_examples_qwen), '| class balance:', np.bincount(y_train_qwen))

    qwen_train_acts = extract_mean_answer_token_activations_from_examples_qwen(train_examples_qwen, batch_size=STEERING_BATCH_SIZE)
    print('Qwen training activation shape at layer 0:', qwen_train_acts[0].shape)

    mm_direction_store_qwen = {}
    wmm_direction_store_qwen = {}
    probe_vector_rows_qwen = []

    for layer in range(qwen_n_layers):
        X_train_layer = qwen_train_acts[layer]

        mm_direction = train_mm_probe(X_train_layer, y_train_qwen).astype(np.float32)
        mm_scores = X_train_layer @ mm_direction
        mm_train_acc = float(((mm_scores > 0).astype(np.int64) == y_train_qwen).mean())
        mm_steering_vector, _ = normalize_direction(mm_direction) if NORMALIZE_STEERING_VECTORS else (mm_direction, float(np.linalg.norm(mm_direction)))

        wmm_model = train_whitened_mm_probe_low_rank(X_train_layer, y_train_qwen, reg=STEERING_WHITEN_REG)
        wmm_scores = (X_train_layer - wmm_model['mean_train']) @ wmm_model['effective_direction']
        wmm_train_acc = float(((wmm_scores > 0).astype(np.int64) == y_train_qwen).mean())
        wmm_steering_vector, _ = normalize_direction(wmm_model['effective_direction']) if NORMALIZE_STEERING_VECTORS else (wmm_model['effective_direction'], float(np.linalg.norm(wmm_model['effective_direction'])))

        mm_direction_store_qwen[layer] = {
            'raw_direction': mm_direction,
            'steering_vector': mm_steering_vector,
            'raw_norm': float(np.linalg.norm(mm_direction)),
            'steering_norm': float(np.linalg.norm(mm_steering_vector)),
            'train_acc': mm_train_acc,
        }
        wmm_direction_store_qwen[layer] = {
            'raw_mm_direction': wmm_model['mm_direction'],
            'effective_direction': wmm_model['effective_direction'],
            'mean_train': wmm_model['mean_train'],
            'steering_vector': wmm_steering_vector,
            'raw_norm': float(np.linalg.norm(wmm_model['effective_direction'])),
            'steering_norm': float(np.linalg.norm(wmm_steering_vector)),
            'train_acc': wmm_train_acc,
            'reg_lambda': wmm_model['reg_lambda'],
            'sample_space_condition_number': wmm_model['sample_space_condition_number'],
        }

        probe_vector_rows_qwen.append({
            'layer': layer,
            'mm_train_acc': mm_train_acc,
            'mm_raw_norm': float(np.linalg.norm(mm_direction)),
            'mm_steering_norm': float(np.linalg.norm(mm_steering_vector)),
            'wmm_train_acc': wmm_train_acc,
            'wmm_raw_norm': float(np.linalg.norm(wmm_model['effective_direction'])),
            'wmm_steering_norm': float(np.linalg.norm(wmm_steering_vector)),
            'wmm_reg_lambda': wmm_model['reg_lambda'],
            'wmm_sample_space_condition_number': wmm_model['sample_space_condition_number'],
        })

    probe_vector_df_qwen = pd.DataFrame(probe_vector_rows_qwen).sort_values('layer').reset_index(drop=True)
    print(probe_vector_df_qwen.head())

    layers_to_test_qwen = [layer for layer in sorted(strength_schedule_by_layer_qwen) if layer < qwen_n_layers]

    print('Layers to test:', layers_to_test_qwen)

    save_dir_qwen = LOCAL_SAVE_DIR
    save_dir_qwen.mkdir(parents=True, exist_ok=True)
    run_id_qwen = time.strftime('%Y%m%d-%H%M%S')
    model_slug_qwen = STEERING_MODEL_NAME.split('/')[-1].replace('.', '_').replace('-', '_')
    summary_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_summary_{model_slug_qwen}_{run_id_qwen}.csv'
    logs_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_logs_{model_slug_qwen}_{run_id_qwen}.csv'
    probe_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_probe_vectors_{model_slug_qwen}_{run_id_qwen}.csv'
    probe_artifact_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_probe_artifacts_{model_slug_qwen}_{run_id_qwen}.npz'
    probe_metadata_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_probe_metadata_{model_slug_qwen}_{run_id_qwen}.json'
    gap_path_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_train_test_gap_{model_slug_qwen}_{run_id_qwen}.csv'
    partial_dir_qwen = save_dir_qwen / f'mmraz_probe_steering_options_answer_vast_partial_{model_slug_qwen}_{run_id_qwen}'
    partial_dir_qwen.mkdir(parents=True, exist_ok=True)

    layer_order_qwen = list(range(qwen_n_layers))
    np.savez_compressed(
        probe_artifact_path_qwen,
        layers=np.asarray(layer_order_qwen, dtype=np.int64),
        mm_raw_directions=np.stack([mm_direction_store_qwen[layer]['raw_direction'] for layer in layer_order_qwen], axis=0),
        mm_steering_vectors=np.stack([mm_direction_store_qwen[layer]['steering_vector'] for layer in layer_order_qwen], axis=0),
        wmm_effective_directions=np.stack([wmm_direction_store_qwen[layer]['effective_direction'] for layer in layer_order_qwen], axis=0),
        wmm_steering_vectors=np.stack([wmm_direction_store_qwen[layer]['steering_vector'] for layer in layer_order_qwen], axis=0),
        wmm_mean_train=np.stack([wmm_direction_store_qwen[layer]['mean_train'] for layer in layer_order_qwen], axis=0),
    )
    probe_metadata_payload_qwen = {
        'model_name': STEERING_MODEL_NAME,
        'dataset_source': STEERING_DATASET_SOURCE,
        'probe_format': 'options_answer_stripped_mean_answer_tokens',
        'answer_pooling': 'mean_answer_token_activations',
        'strip_option_letters_for_probe_training': bool(STRIP_OPTION_LETTERS_FOR_PROBE_TRAINING),
        'patch_prompt_last_only': bool(PATCH_PROMPT_LAST_ONLY),
        'patch_generation_tokens': bool(PATCH_GENERATION_TOKENS),
        'normalize_steering_vectors': bool(NORMALIZE_STEERING_VECTORS),
        'pair_split_random_state': int(PAIR_SPLIT_RANDOM_STATE),
        'layers_to_test': layers_to_test_qwen,
        'strength_schedule_by_layer': {str(k): [float(v) for v in vals] for k, vals in strength_schedule_by_layer_qwen.items()},
    }
    probe_metadata_path_qwen.write_text(json.dumps(probe_metadata_payload_qwen, indent=2) + '\n', encoding='utf-8')
    print('Saved probe artifact bundle:', probe_artifact_path_qwen)
    print('Saved probe metadata      :', probe_metadata_path_qwen)

    def checkpoint_results_qwen(summary_parts, log_parts, probe_df=None, gap_df=None, note=''):
        summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
        logs_df = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame()
        if len(summary_df):
            summary_df.to_csv(summary_path_qwen, index=False)
        if len(logs_df):
            logs_df.to_csv(logs_path_qwen, index=False)
        if probe_df is not None and len(probe_df):
            probe_df.to_csv(probe_path_qwen, index=False)
        if gap_df is not None and len(gap_df):
            gap_df.to_csv(gap_path_qwen, index=False)
        note_suffix = f' ({note})' if note else ''
        print(f'Checkpoint saved{note_suffix}:', summary_path_qwen)
        return summary_df, logs_df

    explicit_train_packet_qwen = build_dataset_packet('explicit_train', explicit_train_eval_pairs_qwen)
    explicit_test_packet_qwen = build_dataset_packet('explicit_test', explicit_eval_pairs_qwen)
    implicit_packet_qwen = build_dataset_packet('implicit_full', implicit_eval_pairs_qwen)

    dataset_packets_qwen = {
        'explicit_train': explicit_train_packet_qwen,
        'explicit_test': explicit_test_packet_qwen,
        'implicit_full': implicit_packet_qwen,
    }

    baseline_start = time.time()
    baseline_results_qwen = {}
    for dataset_name, dataset_packet in dataset_packets_qwen.items():
        baseline_summary, baseline_logs = evaluate_baseline_packet(
            dataset_packet,
            max_new_tokens=STEERING_MAX_NEW_TOKENS,
        )
        baseline_results_qwen[dataset_name] = {
            'summary': baseline_summary,
            'logs': baseline_logs,
        }
    print('Baseline generation time (min):', round((time.time() - baseline_start) / 60.0, 2))

    baseline_summary_rows_qwen = []
    for dataset_name, result in baseline_results_qwen.items():
        baseline_summary = result['summary']
        baseline_summary_rows_qwen.append({
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
            **baseline_summary,
        })

    baseline_summary_df_qwen = pd.DataFrame(baseline_summary_rows_qwen)
    baseline_logs_df_qwen = pd.concat(
        [result['logs'] for result in baseline_results_qwen.values()],
        ignore_index=True,
    )
    checkpoint_results_qwen([baseline_summary_df_qwen], [baseline_logs_df_qwen], probe_df=probe_vector_df_qwen, note='after baseline')

    total_prompt_count_qwen = sum(len(packet) for packet in dataset_packets_qwen.values())
    steering_probe_variants_qwen = [('mm', mm_direction_store_qwen)]
    steering_setting_count_qwen = sum(len(strength_schedule_by_layer_qwen.get(layer, strength_grid_qwen)) for layer in layers_to_test_qwen) * 2 * len(steering_probe_variants_qwen)
    total_expected_generations = total_prompt_count_qwen * (1 + steering_setting_count_qwen)
    total_expected_logprob_scoring_passes = total_expected_generations
    print('Estimated total free generations including baseline:', total_expected_generations)
    print('Estimated total candidate-logprob scoring passes including baseline:', total_expected_logprob_scoring_passes)

    run_start = time.time()
    steering_summary_parts_qwen = [baseline_summary_df_qwen]
    steering_logs_parts_qwen = [baseline_logs_df_qwen]
    for probe_variant, direction_store in steering_probe_variants_qwen:
        for dataset_name, dataset_packet in dataset_packets_qwen.items():
            partial_summary_path_qwen = partial_dir_qwen / f'{dataset_name}_{probe_variant}_summary_partial.csv'
            partial_logs_path_qwen = partial_dir_qwen / f'{dataset_name}_{probe_variant}_logs_partial.csv'
            summary_df, logs_df = evaluate_probe_variant_on_packet(
                dataset_packet,
                probe_variant=probe_variant,
                direction_store=direction_store,
                baseline_summary=baseline_results_qwen[dataset_name]['summary'],
                layers_to_test=layers_to_test_qwen,
                strength_grid=strength_grid_qwen,
                max_new_tokens=STEERING_MAX_NEW_TOKENS,
                partial_summary_path=partial_summary_path_qwen,
                partial_logs_path=partial_logs_path_qwen,
                strength_schedule_by_layer=strength_schedule_by_layer_qwen,
            )
            steering_summary_parts_qwen.append(summary_df)
            steering_logs_parts_qwen.append(logs_df)
            checkpoint_results_qwen(steering_summary_parts_qwen, steering_logs_parts_qwen, probe_df=probe_vector_df_qwen, note=f'{probe_variant}-{dataset_name}')
    print('Steering generation time (min):', round((time.time() - run_start) / 60.0, 2))

    steering_summary_df_qwen = pd.concat(steering_summary_parts_qwen, ignore_index=True)
    steering_logs_df_qwen = pd.concat(steering_logs_parts_qwen, ignore_index=True)

    for col in [
        'prop_choose_long_term_no_fallback',
        'prop_choose_immediate_no_fallback',
        'steering_success_no_fallback',
        'baseline_prop_choose_long_term_no_fallback',
        'baseline_prop_choose_immediate_no_fallback',
        'delta_long_term_vs_baseline_no_fallback',
        'delta_immediate_vs_baseline_no_fallback',
        'n_no_fallback_prompts',
        'mean_long_minus_immediate_avg_logprob',
        'mean_long_minus_immediate_sum_logprob',
        'prop_logprob_prefers_long_term',
        'baseline_mean_long_minus_immediate_avg_logprob',
        'baseline_prop_logprob_prefers_long_term',
        'delta_long_minus_immediate_avg_logprob_vs_baseline',
        'delta_prop_logprob_prefers_long_term_vs_baseline',
    ]:
        if col not in steering_summary_df_qwen.columns:
            steering_summary_df_qwen[col] = np.nan

    train_test_gap_df_qwen = (
        steering_summary_df_qwen[
            steering_summary_df_qwen['dataset'].isin(['explicit_train', 'explicit_test'])
            & (steering_summary_df_qwen['layer'] >= 0)
        ]
        .pivot_table(
            index=['probe_variant', 'condition', 'layer', 'strength', 'signed_strength'],
            columns='dataset',
            values=['steering_success', 'steering_success_no_fallback', 'prop_choose_long_term', 'prop_choose_immediate', 'prop_choose_long_term_no_fallback', 'prop_choose_immediate_no_fallback', 'fallback_rate'],
        )
    )
    if len(train_test_gap_df_qwen):
        train_test_gap_df_qwen.columns = [f'{metric}_{dataset}' for metric, dataset in train_test_gap_df_qwen.columns]
        train_test_gap_df_qwen = train_test_gap_df_qwen.reset_index()
        for col in [
            'steering_success_explicit_train',
            'steering_success_explicit_test',
            'steering_success_no_fallback_explicit_train',
            'steering_success_no_fallback_explicit_test',
            'prop_choose_long_term_explicit_train',
            'prop_choose_long_term_explicit_test',
            'prop_choose_long_term_no_fallback_explicit_train',
            'prop_choose_long_term_no_fallback_explicit_test',
            'fallback_rate_explicit_train',
            'fallback_rate_explicit_test',
        ]:
            if col not in train_test_gap_df_qwen.columns:
                train_test_gap_df_qwen[col] = np.nan
        train_test_gap_df_qwen['prop_choose_long_term_gap_train_minus_test'] = (
            train_test_gap_df_qwen['prop_choose_long_term_explicit_train'] - train_test_gap_df_qwen['prop_choose_long_term_explicit_test']
        )
        train_test_gap_df_qwen['prop_choose_long_term_no_fallback_gap_train_minus_test'] = (
            train_test_gap_df_qwen['prop_choose_long_term_no_fallback_explicit_train'] - train_test_gap_df_qwen['prop_choose_long_term_no_fallback_explicit_test']
        )
        train_test_gap_df_qwen['steering_success_gap_train_minus_test'] = (
            train_test_gap_df_qwen['steering_success_explicit_train'] - train_test_gap_df_qwen['steering_success_explicit_test']
        )
        train_test_gap_df_qwen['steering_success_no_fallback_gap_train_minus_test'] = (
            train_test_gap_df_qwen['steering_success_no_fallback_explicit_train'] - train_test_gap_df_qwen['steering_success_no_fallback_explicit_test']
        )
        train_test_gap_df_qwen['fallback_rate_gap_train_minus_test'] = (
            train_test_gap_df_qwen['fallback_rate_explicit_train'] - train_test_gap_df_qwen['fallback_rate_explicit_test']
        )
    else:
        train_test_gap_df_qwen = pd.DataFrame()

    checkpoint_results_qwen(steering_summary_parts_qwen, steering_logs_parts_qwen, probe_df=probe_vector_df_qwen, gap_df=train_test_gap_df_qwen, note='final')

    print(steering_summary_df_qwen.head(20))
    print(steering_logs_df_qwen.head(20))
    if len(train_test_gap_df_qwen):
        print(train_test_gap_df_qwen.head(20))

    write_status(status_path, {
        'state': 'completed',
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'summary_path': str(summary_path_qwen),
        'logs_path': str(logs_path_qwen),
        'probe_stats_path': str(probe_path_qwen),
        'probe_artifact_path': str(probe_artifact_path_qwen),
        'probe_metadata_path': str(probe_metadata_path_qwen),
        'gap_path': str(gap_path_qwen) if len(train_test_gap_df_qwen) else None,
    })


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status_dir = Path.cwd() / 'results' / 'vast_remote_failure'
        status_dir.mkdir(parents=True, exist_ok=True)
        write_status(status_dir / 'remote_job_status.json', {
            'state': 'failed',
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
        raise
