#!/usr/bin/env python3
"""
MECHANISTIC EXTENSIONS — Transforms paper from probing study to mech interp
============================================================================

SECTION 1: LOGIT LENS ACROSS TRAINING (~4h)
  At each layer, project residual stream through W_U to vocabulary space.
  Track probability/rank of return-type tokens (int, str, bool, list, float)
  across training checkpoints. No probe needed — direct readout of model's
  internal predictions. 5 key checkpoints × 2 Pythia sizes (2.8B, 410M).

SECTION 2: ACTIVATION PATCHING ON FUNCTION NAME (~2h)
  Swap function name token activations between signatures with different
  return types. If probe prediction changes → causal evidence that probe
  reads function name. Standard causal mediation analysis.
  Final checkpoint, Pythia-2.8B.

SECTION 3: EXTENDED K DECAY DURING TRAINING (~4h)
  Run Future Lens K decay at all 17 checkpoints (not just 5) for smooth curve.
  Pythia-2.8B only.

SECTION 4: IMPROVED BASELINE SIGNATURES (~3h)
  200 signatures with clearer name→type mappings where N+P baseline achieves ~85%+.
  Rerun training dynamics at 5 key checkpoints with proper baseline.

Resume support. Saves after each section.
Expected: ~13h on L40S.
"""

import json, os, sys, time, re, gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PCA_DIM = 128
OUTDIR = "results/lookahead/final"
os.makedirs(OUTDIR, exist_ok=True)

KEY_STEPS = ["step0", "step512", "step4000", "step32000", "step143000"]
ALL_STEPS = ["step0","step1","step8","step64","step128","step256","step512",
             "step1000","step2000","step4000","step8000","step16000",
             "step32000","step64000","step96000","step128000","step143000"]

# Original signatures (same as before)
SIGS = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'), ('def gcd(a, b):', 'int'), ('def lcm(a, b):', 'int'),
    ('def abs_value(n):', 'int'), ('def sign(n):', 'int'), ('def clamp(val, lo, hi):', 'int'),
    ('def greet(name):', 'str'), ('def farewell(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def to_lower(text):', 'str'), ('def capitalize(text):', 'str'), ('def strip_whitespace(text):', 'str'),
    ('def reverse_string(s):', 'str'), ('def repeat_string(s, n):', 'str'),
    ('def join_words(words):', 'str'), ('def first_word(text):', 'str'), ('def last_word(text):', 'str'),
    ('def remove_spaces(s):', 'str'), ('def replace_char(s, old, new):', 'str'),
    ('def first_name(full_name):', 'str'), ('def last_name(full_name):', 'str'),
    ('def format_date(year, month, day):', 'str'), ('def format_time(hours, minutes):', 'str'),
    ('def to_binary(n):', 'str'), ('def to_hex(n):', 'str'), ('def to_roman(n):', 'str'),
    ('def slug(text):', 'str'), ('def title_case(text):', 'str'), ('def snake_case(text):', 'str'),
    ('def camel_case(text):', 'str'), ('def pad_left(s, width, char):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_odd(n):', 'bool'), ('def is_positive(x):', 'bool'),
    ('def is_negative(x):', 'bool'), ('def is_zero(x):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def is_palindrome(s):', 'bool'), ('def is_empty(s):', 'bool'), ('def is_sorted(items):', 'bool'),
    ('def contains(items, target):', 'bool'), ('def starts_with(text, prefix):', 'bool'),
    ('def ends_with(text, suffix):', 'bool'), ('def is_alpha(text):', 'bool'),
    ('def is_digit(text):', 'bool'), ('def is_upper(text):', 'bool'), ('def is_lower(text):', 'bool'),
    ('def has_duplicates(items):', 'bool'), ('def all_positive(numbers):', 'bool'),
    ('def any_negative(numbers):', 'bool'), ('def is_valid_email(text):', 'bool'),
    ('def is_substring(s, sub):', 'bool'), ('def is_anagram(s1, s2):', 'bool'),
    ('def is_power_of_two(n):', 'bool'), ('def is_leap_year(year):', 'bool'),
    ('def get_evens(numbers):', 'list'), ('def get_odds(numbers):', 'list'),
    ('def filter_positive(numbers):', 'list'), ('def filter_negative(numbers):', 'list'),
    ('def unique(items):', 'list'), ('def flatten(nested):', 'list'),
    ('def sort_ascending(items):', 'list'), ('def sort_descending(items):', 'list'),
    ('def reverse_list(items):', 'list'), ('def split_words(text):', 'list'),
    ('def split_lines(text):', 'list'), ('def split_chars(text):', 'list'),
    ('def zip_lists(a, b):', 'list'), ('def merge_sorted(a, b):', 'list'),
    ('def remove_duplicates(items):', 'list'), ('def take(items, n):', 'list'),
    ('def drop(items, n):', 'list'), ('def chunk(items, size):', 'list'),
    ('def interleave(a, b):', 'list'), ('def get_keys(d):', 'list'),
    ('def get_values(d):', 'list'), ('def range_list(start, stop):', 'list'),
    ('def neighbors(graph, node):', 'list'), ('def find_all(text, pattern):', 'list'),
    ('def average(numbers):', 'float'), ('def median(numbers):', 'float'),
    ('def variance(numbers):', 'float'), ('def std_dev(numbers):', 'float'),
    ('def to_celsius(f):', 'float'), ('def to_fahrenheit(c):', 'float'),
    ('def percentage(part, total):', 'float'), ('def ratio(a, b):', 'float'),
    ('def distance(x1, y1, x2, y2):', 'float'), ('def magnitude(x, y, z):', 'float'),
    ('def dot_product(a, b):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def circle_area(radius):', 'float'), ('def sphere_volume(radius):', 'float'),
    ('def triangle_area(base, height):', 'float'), ('def hypotenuse(a, b):', 'float'),
    ('def sigmoid(x):', 'float'), ('def relu(x):', 'float'), ('def tanh(x):', 'float'),
    ('def log_base(x, base):', 'float'), ('def square_root(x):', 'float'),
    ('def cube_root(x):', 'float'), ('def lerp(a, b, t):', 'float'),
    ('def normalize(value, min_val, max_val):', 'float'), ('def bmi(weight, height):', 'float'),
    ('def compound_interest(principal, rate, years):', 'float'),
]

# Future Lens prompts
FL_PROMPTS = [
    "The process of photosynthesis in plants involves the conversion of",
    "Quantum mechanics fundamentally changed our understanding of physics by",
    "In molecular biology, DNA replication is the process by which",
    "Neurons communicate with each other through electrical and chemical",
    "The periodic table organizes all known chemical elements according to",
    "General relativity predicts that massive objects cause a distortion in",
    "Evolution through natural selection occurs when organisms with favorable",
    "The human immune system protects the body from pathogens by",
    "The Industrial Revolution transformed European society beginning in the",
    "Ancient Rome expanded its territory through military conquest and",
    "Artificial intelligence systems learn from data by identifying patterns",
    "The internet was originally developed as a military communication network",
    "Machine learning algorithms can be broadly classified into supervised and",
    "Cloud computing allows organizations to access computing resources over",
    "Neural networks are computational models inspired by the structure of",
    "Programming languages provide abstractions that allow developers to write",
    "Climate change is primarily driven by the emission of greenhouse",
    "Renewable energy sources include solar wind and hydroelectric power",
    "The largest ocean on Earth is the Pacific Ocean which",
    "Democracy as a form of government gives citizens the power",
    "Education plays a crucial role in the development of modern",
    "The stock market serves as a platform where investors can",
    "The human brain is the most complex organ in the",
    "In Python programming the def keyword is used to define",
    "A function that takes a list of numbers and returns",
    "The algorithm works by first sorting the input array and",
    "To implement a binary search tree you need to define",
    "The main difference between a list and a tuple in",
    "Object oriented programming encapsulates data and behavior within classes",
    "A recursive function must have a base case to prevent",
]


def get_layers(n_layers):
    return sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                       2*n_layers//3, 5*n_layers//6, n_layers-1]))

def save(data, name):
    path = f"{OUTDIR}/mech_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  [SAVED] {path}")

def load(name):
    path = f"{OUTDIR}/mech_{name}.json"
    if os.path.exists(path): return json.load(open(path))
    return None

def load_model(hf_id, tl_id, step):
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    try:
        hf = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=step, torch_dtype=torch.float16, use_safetensors=True)
    except:
        hf = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=step, torch_dtype=torch.float16, use_safetensors=False)
    tl = HookedTransformer.from_pretrained(
        tl_id, hf_model=hf, device="cuda", dtype=torch.float16)
    del hf; torch.cuda.empty_cache()
    tl.eval()
    return tl


# ================================================================
# SECTION 1: LOGIT LENS ACROSS TRAINING
# ================================================================
def section1_logit_lens():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: LOGIT LENS ACROSS TRAINING")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    models_to_run = [
        ("EleutherAI/pythia-2.8b-deduped", "pythia-2.8b-deduped", "Pythia-2.8B"),
        ("EleutherAI/pythia-410m-deduped", "pythia-410m-deduped", "Pythia-410M"),
    ]
    
    targets = sorted(set(r for _, r in SIGS))
    t2i_types = {t: i for i, t in enumerate(targets)}
    
    all_results = {}
    
    for hf_id, tl_id, label in models_to_run:
        logger.info(f"\n  Model: {label}")
        model_results = {"model": hf_id, "label": label, "checkpoints": []}
        
        safe = tl_id.replace('-','_').replace('.','_')
        save_key = f"s1_logit_{safe}"
        existing = load(save_key)
        if existing and "checkpoints" in existing:
            done = {cp["step"] for cp in existing["checkpoints"]}
            model_results = existing
            logger.info(f"  Resuming: {len(done)}/{len(KEY_STEPS)} done")
        else:
            done = set()
        
        for step in KEY_STEPS:
            if step in done:
                logger.info(f"    {step}: cached")
                continue
            
            t0 = time.time()
            try:
                model = load_model(hf_id, tl_id, step)
            except Exception as e:
                logger.warning(f"    {step}: FAILED — {e}")
                continue
            
            layers = get_layers(model.cfg.n_layers)
            n_layers = model.cfg.n_layers
            W_U = model.W_U  # [d_model, d_vocab]
            
            # Find token IDs for return type names
            # Try with and without leading space
            type_token_ids = {}
            for t in targets:
                candidates = [t, f" {t}", f" {t}:", t.capitalize()]
                best_id = None
                for c in candidates:
                    toks = model.to_tokens(c, prepend_bos=False)
                    if toks.shape[1] == 1:
                        best_id = toks[0, 0].item()
                        break
                if best_id is None:
                    # Use first token of the type name
                    toks = model.to_tokens(t, prepend_bos=False)
                    best_id = toks[0, 0].item()
                type_token_ids[t] = best_id
            
            logger.info(f"    Type token IDs: {type_token_ids}")
            
            # For each signature, compute logit lens at each layer
            # Track: correct type rank, correct type probability, top-1 accuracy
            layer_ranks = {l: [] for l in range(n_layers)}
            layer_probs = {l: [] for l in range(n_layers)}
            layer_correct = {l: 0 for l in range(n_layers)}
            
            for si, (sig, ret_type) in enumerate(SIGS):
                prompt = sig + "\n    "
                tokens = model.to_tokens(prompt, prepend_bos=True)
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in range(n_layers)])
                
                correct_tid = type_token_ids[ret_type]
                
                for l in range(n_layers):
                    # Apply final layer norm + unembed
                    h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :]  # last position
                    h_normed = model.ln_final(h.unsqueeze(0)).squeeze(0)
                    logits = h_normed @ W_U  # [d_vocab]
                    
                    # Probability of correct type token
                    probs = F.softmax(logits, dim=-1)
                    correct_prob = probs[correct_tid].item()
                    layer_probs[l].append(correct_prob)
                    
                    # Rank of correct type token
                    sorted_indices = torch.argsort(logits, descending=True)
                    rank = (sorted_indices == correct_tid).nonzero(as_tuple=True)[0].item()
                    layer_ranks[l].append(rank)
                    
                    # Top-1 among type tokens only
                    type_logits = {t: logits[tid].item() for t, tid in type_token_ids.items()}
                    predicted_type = max(type_logits, key=type_logits.get)
                    if predicted_type == ret_type:
                        layer_correct[l] += 1
                
                del cache; torch.cuda.empty_cache()
                if (si + 1) % 25 == 0:
                    logger.info(f"    {step}: {si+1}/{len(SIGS)}")
            
            n_sigs = len(SIGS)
            ckpt_result = {"step": step, "layers": {}}
            
            logger.info(f"    {'Layer':>6} {'MeanRank':>9} {'MeanProb':>9} {'TypeAcc':>8}")
            
            # Report at sampled layers
            for l in list(range(0, n_layers, max(1, n_layers//8))) + [n_layers-1]:
                l = min(l, n_layers-1)
                mean_rank = np.mean(layer_ranks[l])
                mean_prob = np.mean(layer_probs[l])
                type_acc = layer_correct[l] / n_sigs
                
                logger.info(f"    L{l:>4} {mean_rank:>9.1f} {mean_prob:>9.4f} {type_acc:>8.4f}")
                
                ckpt_result["layers"][str(l)] = {
                    "mean_rank": float(mean_rank),
                    "mean_prob": float(mean_prob),
                    "type_accuracy": float(type_acc),
                }
            
            # Store all layers for plotting
            ckpt_result["all_layers"] = {
                str(l): {
                    "mean_rank": float(np.mean(layer_ranks[l])),
                    "mean_prob": float(np.mean(layer_probs[l])),
                    "type_accuracy": float(layer_correct[l] / n_sigs),
                } for l in range(n_layers)
            }
            
            elapsed = time.time() - t0
            logger.info(f"    ({elapsed:.0f}s)")
            
            model_results["checkpoints"].append(ckpt_result)
            save(model_results, save_key)
            del model; torch.cuda.empty_cache(); gc.collect()
        
        all_results[safe] = model_results
    
    return all_results


# ================================================================
# SECTION 2: ACTIVATION PATCHING ON FUNCTION NAME
# ================================================================
def section2_patching():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: ACTIVATION PATCHING ON FUNCTION NAME")
    logger.info("=" * 70)
    
    hf_id = "EleutherAI/pythia-2.8b-deduped"
    tl_id = "pythia-2.8b-deduped"
    step = "step143000"
    
    model = load_model(hf_id, tl_id, step)
    n_layers = model.cfg.n_layers
    layers = get_layers(n_layers)
    
    targets = sorted(set(r for _, r in SIGS))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[r] for _, r in SIGS])
    
    # First, train a probe at the best layer on clean data
    logger.info("  Training reference probe...")
    best_layer = n_layers * 2 // 3  # Use a typical good layer
    
    clean_acts = []
    token_positions = []  # Track where function name tokens are
    
    for si, (sig, _) in enumerate(SIGS):
        prompt = sig + "\n    "
        tokens = model.to_tokens(prompt, prepend_bos=True)
        
        # Find function name token positions
        # Tokens: [BOS, def, space, name_tok1, name_tok2, ..., (, params, ), :, \n, spaces]
        prompt_tokens = tokens[0].tolist()
        # The function name starts after "def " (typically position 2-3)
        # Find position of "(" to know where name ends
        paren_positions = [i for i, t in enumerate(prompt_tokens) 
                          if model.to_string(torch.tensor([t])).strip() == "("]
        if paren_positions:
            name_end = paren_positions[0]
            name_start = 2  # After BOS and "def"
            # Verify by checking the first non-def token
            for pos in range(1, len(prompt_tokens)):
                tok_str = model.to_string(torch.tensor([prompt_tokens[pos]])).strip()
                if tok_str not in ['def', '']:
                    name_start = pos
                    break
        else:
            name_start = 2
            name_end = min(5, len(prompt_tokens) - 1)
        
        token_positions.append((name_start, name_end))
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=[f"blocks.{best_layer}.hook_resid_post"])
        clean_acts.append(cache[f"blocks.{best_layer}.hook_resid_post"][0, -1, :].cpu().numpy())
        del cache; torch.cuda.empty_cache()
    
    X_clean = np.stack(clean_acts)
    X_clean = StandardScaler().fit_transform(X_clean)
    X_clean = PCA(n_components=min(PCA_DIM, X_clean.shape[0]-1), random_state=42).fit_transform(X_clean)
    
    # Train probe
    probe = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    probe.fit(X_clean, labels)
    clean_acc = probe.score(X_clean, labels)
    logger.info(f"  Clean probe accuracy: {clean_acc:.4f}")
    
    # Now do patching: for each pair of different-type signatures,
    # swap the function name embeddings and measure prediction change
    logger.info("\n  Patching function name embeddings...")
    
    # Group signatures by type
    type_groups = {t: [i for i, (_, r) in enumerate(SIGS) if r == t] for t in targets}
    
    # For each target type, patch name from a different-type signature
    patch_results = {"clean_acc": float(clean_acc), "patches": []}
    n_changed = 0
    n_total = 0
    
    # Sample patch pairs: for each signature, patch with a random different-type sig
    np.random.seed(42)
    
    for si, (sig_a, type_a) in enumerate(SIGS):
        # Find a signature with different type
        other_types = [t for t in targets if t != type_a]
        other_type = np.random.choice(other_types)
        other_idx = np.random.choice(type_groups[other_type])
        sig_b, type_b = SIGS[other_idx]
        
        # Get token embeddings for both
        tokens_a = model.to_tokens(sig_a + "\n    ", prepend_bos=True)
        tokens_b = model.to_tokens(sig_b + "\n    ", prepend_bos=True)
        
        name_start_a, name_end_a = token_positions[si]
        name_start_b, name_end_b = token_positions[other_idx]
        
        # Create patched input: signature A with name tokens from B
        # Replace name embedding in the residual stream at layer 0
        with torch.no_grad():
            # Get embeddings
            embed_a = model.W_E[tokens_a[0]]  # [seq_len_a, d_model]
            embed_b = model.W_E[tokens_b[0]]  # [seq_len_b, d_model]
            
            # Patch: replace name positions in A with name from B
            patched_embed = embed_a.clone()
            name_len_a = name_end_a - name_start_a
            name_len_b = name_end_b - name_start_b
            
            # Only patch if name lengths match (simple case)
            if name_len_a == name_len_b and name_len_a > 0:
                patched_embed[name_start_a:name_end_a] = embed_b[name_start_b:name_end_b]
                
                # Run patched through model
                # Use hook to inject patched embeddings
                def patch_embed_hook(value, hook):
                    value[0] = patched_embed
                    return value
                
                patched_logits = model.run_with_hooks(
                    tokens_a,
                    fwd_hooks=[(f"hook_embed", patch_embed_hook)],
                    return_type="logits"
                )
                
                # Get activations at best_layer for the patched input
                _, patched_cache = model.run_with_hooks(
                    tokens_a,
                    fwd_hooks=[(f"hook_embed", patch_embed_hook)],
                    return_type="both",
                    fwd_hook_names_filter=[f"blocks.{best_layer}.hook_resid_post"]
                )
                
                # This approach is getting complex. Simpler: just run with hooks
                # and extract the activation at the last position
                
                # Actually, let's use a simpler approach: 
                # Run the model with the patched embedding directly
                
                # Get residual stream at best_layer
                patched_act = None
                def capture_hook(value, hook):
                    nonlocal patched_act
                    patched_act = value[0, -1, :].cpu().numpy()
                    return value
                
                model.run_with_hooks(
                    tokens_a,
                    fwd_hooks=[
                        ("hook_embed", patch_embed_hook),
                        (f"blocks.{best_layer}.hook_resid_post", capture_hook),
                    ]
                )
                
                if patched_act is not None:
                    # Compare clean vs patched prediction
                    clean_pred = probe.predict(X_clean[si:si+1])[0]
                    
                    # Transform patched activation through same pipeline
                    # (Note: using the fitted scaler/PCA from clean data)
                    # This is approximate but directionally correct
                    patched_pred = t2i.get(type_b, -1)  # What we'd expect if patch works
                    
                    # Simple check: did the name swap change the probe prediction?
                    # We need to re-extract with the same StandardScaler + PCA
                    # For simplicity, check if patched activation is closer to type_b's mean
                    
                    n_total += 1
                    # Record the patch
                    patch_results["patches"].append({
                        "source": sig_a, "source_type": type_a,
                        "donor": sig_b, "donor_type": type_b,
                        "clean_pred": targets[clean_pred],
                        "name_matched": name_len_a == name_len_b,
                    })
        
        if (si + 1) % 25 == 0:
            logger.info(f"    Processed {si+1}/{len(SIGS)}")
    
    # More robust approach: extract ALL patched activations and run probe
    logger.info("\n  Systematic patching analysis...")
    
    # For each type pair, swap ALL name embeddings and measure average accuracy change
    type_pair_results = {}
    
    for src_type in targets:
        for dst_type in targets:
            if src_type == dst_type: continue
            
            src_indices = type_groups[src_type][:5]  # 5 examples per type
            dst_indices = type_groups[dst_type][:5]
            
            changed_count = 0
            tested = 0
            
            for si in src_indices:
                for di in dst_indices:
                    ns_a, ne_a = token_positions[si]
                    ns_b, ne_b = token_positions[di]
                    
                    if (ne_a - ns_a) != (ne_b - ns_b) or (ne_a - ns_a) <= 0:
                        continue
                    
                    sig_a_str, _ = SIGS[si]
                    sig_b_str, _ = SIGS[di]
                    
                    tokens_a = model.to_tokens(sig_a_str + "\n    ", prepend_bos=True)
                    tokens_b = model.to_tokens(sig_b_str + "\n    ", prepend_bos=True)
                    
                    embed_a = model.W_E[tokens_a[0]].clone()
                    embed_b = model.W_E[tokens_b[0]]
                    
                    embed_a[ns_a:ne_a] = embed_b[ns_b:ne_b]
                    
                    patched_act = None
                    def patch_hook(value, hook, patched=embed_a):
                        value[0] = patched
                        return value
                    def capture(value, hook):
                        nonlocal patched_act
                        patched_act = value[0, -1, :].cpu().numpy()
                        return value
                    
                    with torch.no_grad():
                        model.run_with_hooks(tokens_a, fwd_hooks=[
                            ("hook_embed", patch_hook),
                            (f"blocks.{best_layer}.hook_resid_post", capture),
                        ])
                    
                    if patched_act is not None:
                        # Check if probe now predicts donor type
                        # Use cosine similarity to type centroids
                        clean_pred = targets[labels[si]]  # = src_type
                        tested += 1
                        
                        # Compare patched activation to clean activations of each type
                        # Simple: cosine similarity to mean activation of each type
                        type_means = {}
                        for t in targets:
                            t_indices = [i for i, (_, r) in enumerate(SIGS) if r == t]
                            type_means[t] = np.mean(X_clean[t_indices], axis=0)
                        
                        patched_scaled = StandardScaler().fit(X_clean).transform(patched_act.reshape(1, -1))
                        patched_pca = PCA(n_components=min(PCA_DIM, X_clean.shape[0]-1), 
                                         random_state=42).fit(
                            StandardScaler().fit_transform(np.vstack([r.reshape(1,-1) for r in clean_acts]))
                        ).transform(patched_scaled)
                        
                        pred = probe.predict(patched_pca)[0]
                        pred_type = targets[pred]
                        
                        if pred_type != src_type:
                            changed_count += 1
            
            if tested > 0:
                change_rate = changed_count / tested
                logger.info(f"    {src_type}->{dst_type}: {changed_count}/{tested} changed ({change_rate:.2f})")
                type_pair_results[f"{src_type}->{dst_type}"] = {
                    "changed": changed_count, "tested": tested, "rate": float(change_rate)
                }
    
    patch_results["type_pairs"] = type_pair_results
    
    avg_change = np.mean([v["rate"] for v in type_pair_results.values()]) if type_pair_results else 0
    logger.info(f"\n  Average prediction change rate: {avg_change:.4f}")
    logger.info(f"  (Higher = function name causally determines probe prediction)")
    
    patch_results["avg_change_rate"] = float(avg_change)
    
    del model; torch.cuda.empty_cache(); gc.collect()
    return patch_results


# ================================================================
# SECTION 3: K DECAY AT ALL 17 CHECKPOINTS
# ================================================================
def section3_extended_k_decay():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 3: K DECAY AT ALL 17 CHECKPOINTS")
    logger.info("=" * 70)
    
    hf_id = "EleutherAI/pythia-2.8b-deduped"
    tl_id = "pythia-2.8b-deduped"
    K_VALUES = [1, 3, 5]
    N_GEN = 80
    MIN_TARGET = 10
    
    save_key = "s3_kdecay_all"
    existing = load(save_key)
    if existing and "checkpoints" in existing:
        results = existing
        done = {cp["step"] for cp in results["checkpoints"]}
        logger.info(f"  Resuming: {len(done)}/{len(ALL_STEPS)} done")
    else:
        results = {"model": hf_id, "checkpoints": []}
        done = set()
    
    for step in ALL_STEPS:
        if step in done:
            logger.info(f"  {step}: cached")
            continue
        
        t0 = time.time()
        try:
            model = load_model(hf_id, tl_id, step)
        except Exception as e:
            logger.warning(f"  {step}: FAILED — {e}")
            continue
        
        layers = get_layers(model.cfg.n_layers)
        W_E = model.W_E.detach()
        
        # Generate
        all_seqs = []
        for prompt in FL_PROMPTS:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        
        train_seqs = all_seqs[:15]
        test_seqs = all_seqs[15:]
        
        ckpt_result = {"step": step}
        
        for k in K_VALUES:
            test_tgts = []
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                for n in range(pl, len(ids) - k):
                    test_tgts.append(ids[n + k])
            tc = Counter(test_tgts)
            frequent = {t for t, c in tc.items() if c >= MIN_TARGET}
            t2i = {t: i for i, t in enumerate(sorted(frequent))}
            n_cls = len(t2i)
            
            if n_cls < 3:
                ckpt_result[f"k{k}"] = {"skip": True, "n_classes": n_cls}
                continue
            
            activations = {l: [] for l in layers}
            ctx_embs, fl_labels = [], []
            
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    _, cache = model.run_with_cache(inp,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
                for n in range(pl, len(ids) - k):
                    tgt = ids[n + k]
                    if tgt not in t2i: continue
                    fl_labels.append(t2i[tgt])
                    for l in layers:
                        activations[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                    ws = max(0, n - 4)
                    ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                    ctx_embs.append(ctx.mean(axis=0))
                del cache; torch.cuda.empty_cache()
            
            fl_labels = np.array(fl_labels)
            n_ex = len(fl_labels)
            if n_ex < 20:
                ckpt_result[f"k{k}"] = {"skip": True, "n_examples": n_ex}
                continue
            min_c = min(Counter(fl_labels).values())
            n_splits = min(5, min_c)
            if n_splits < 2:
                ckpt_result[f"k{k}"] = {"skip": True}
                continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                StandardScaler().fit_transform(np.stack(ctx_embs)))
            ctx_acc = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_ctx, fl_labels, cv=cv, scoring="accuracy").mean()
            
            best_p, best_l = 0, 0
            for l in layers:
                X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                    StandardScaler().fit_transform(np.stack(activations[l])))
                acc = cross_val_score(
                    LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                    X, fl_labels, cv=cv, scoring="accuracy").mean()
                if acc > best_p: best_p, best_l = acc, l
            
            gap = best_p - ctx_acc
            logger.info(f"  {step} K={k}: {n_ex} ex | gap={gap:+.3f}")
            ckpt_result[f"k{k}"] = {
                "n_examples": n_ex, "n_classes": n_cls,
                "context": float(ctx_acc), "probe": float(best_p),
                "gap": float(gap), "best_layer": int(best_l),
            }
        
        elapsed = time.time() - t0
        logger.info(f"  ({elapsed:.0f}s)")
        results["checkpoints"].append(ckpt_result)
        save(results, save_key)
        del model; torch.cuda.empty_cache(); gc.collect()
    
    step_order = {s: i for i, s in enumerate(ALL_STEPS)}
    results["checkpoints"].sort(key=lambda x: step_order.get(x["step"], 99))
    save(results, save_key)
    return results


# ================================================================
# MAIN
# ================================================================
def main():
    logger.info("=" * 70)
    logger.info("MECHANISTIC EXTENSIONS")
    logger.info("=" * 70)
    t_start = time.time()
    
    all_results = {}
    
    # Section 1: Logit lens
    all_results["logit_lens"] = section1_logit_lens()
    save(all_results, "s1_complete")
    
    # Section 2: Activation patching
    # SKIP: patching needs fix
    all_results["patching"] = {"skipped": True}
    save(all_results, "s2_complete")
    
    # Section 3: Extended K decay
    all_results["k_decay_extended"] = section3_extended_k_decay()
    save(all_results, "s3_complete")
    
    # Save all
    save(all_results, "all_complete")
    
    elapsed = (time.time() - t_start) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"DONE — {elapsed:.1f} hours")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
