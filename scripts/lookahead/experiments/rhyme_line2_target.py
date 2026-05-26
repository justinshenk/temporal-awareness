#!/usr/bin/env python3
"""
rhyme_line2_target.py — Probe rhyme staircase with LINE-2 positions as target.

The existing rhyme staircase targets the END OF LINE 1 (newline after the
rhyme cue word "sing"). This tests whether the newline position encodes
rhyme family better than earlier positions — but "sing" is already in the
sequence, so a skeptic could call it cue-localization.

This script modifies the prompt to include the BEGINNING OF LINE 2 and
probes at positions WITHIN LINE 2 (before the rhyme word appears). The
earlier baseline now includes ALL of Line 1, including "sing" itself.

If the Line 2 positions still show a large gap over the earlier baseline
(which includes "sing"), this is evidence of genuine forward planning:
the model at Line 2 positions encodes the rhyme constraint more clearly
than the "sing" position itself, even though no Line 2 rhyme word has
been seen.

Usage:
    HF_TOKEN=... python3 scripts/lookahead/experiments/rhyme_line2_target.py

Output: results/v2/line2_rhyme_staircase.json
"""
import sys, os, json, glob, time, logging
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_data_minimal')

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTFILE = Path('results/v2/line2_rhyme_staircase.json')

# ── Line-2 prefixes (generic, neutral, don't carry rhyme info) ──────────────
# We append these to Line 1 to create a prompt that has Line 2 started.
# The model sees Line1 + "\n" + prefix, and we probe at positions within prefix.
# Prefixes are chosen to be semantically neutral and not carry rhyme class cues.
LINE2_PREFIXES = [
    "And ",
    "Where ",
    "As ",
    "The ",
    "With ",
]

# ── Models to test ────────────────────────────────────────────────────────────
MODELS = [
    "google/gemma-2-2b",
    "EleutherAI/pythia-1.4b-deduped",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.2-3B",
    "gpt2-xl",
]

# ── Data ──────────────────────────────────────────────────────────────────────
RHYME_PREFIX = "A rhyming couplet:\n"
RHYME_FAMILIES = ("ing", "air", "ip", "oat", "ird", "ee", "ight", "ake", "ow", "it")

def build_line2_examples(data_root='data/maar_data_minimal'):
    """Build prompts with Line 2 prefix appended."""
    import random
    rng = random.Random(42)
    root = Path(data_root)
    path = root / 'test' / 'rhyme_family_lines.json'
    d = json.load(open(path))
    
    examples = []
    for fam_idx, fam in enumerate(RHYME_FAMILIES):
        lines = d.get(fam, [])
        for line1 in lines:
            # Pick a random neutral prefix for Line 2
            prefix = rng.choice(LINE2_PREFIXES)
            # Full prompt: "A rhyming couplet:\n{line1}{prefix}"
            # line1 already ends with "\n"
            prompt = RHYME_PREFIX + line1.rstrip('\n') + '\n' + prefix
            examples.append({
                'prompt': prompt,
                'label': fam_idx,  # 10-class rhyme family
                'rhyme_family': fam,
                'line1': line1.rstrip('\n'),
                'line2_prefix': prefix,
            })
    log.info(f"Built {len(examples)} line-2 examples across {len(RHYME_FAMILIES)} families")
    return examples

def extract_activations_line2(model, tokenizer, examples, device='cuda'):
    """Extract hidden states at ALL positions for each example."""
    import torch
    all_hidden = {}  # position_idx -> list of (hidden_state, label)
    max_positions = 0
    
    model.eval()
    for ex in examples:
        inputs = tokenizer(ex['prompt'], return_tensors='pt').to(device)
        seq_len = inputs.input_ids.shape[1]
        max_positions = max(max_positions, seq_len)
        
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        
        # Store per-layer, per-position hidden states
        # We'll probe at the LAST POSITION (end of line2_prefix)
        # and compare against ALL earlier positions
        for layer_idx, hs in enumerate(out.hidden_states[1:], 1):
            h = hs[0].float().cpu().numpy()  # (seq_len, hidden_dim)
            for pos_idx in range(seq_len):
                key = (layer_idx, pos_idx)
                if key not in all_hidden:
                    all_hidden[key] = {'X': [], 'y': []}
                all_hidden[key]['X'].append(h[pos_idx])
                all_hidden[key]['y'].append(ex['label'])
    
    return all_hidden, max_positions

def probe_layer2(model_id, examples):
    """Run probe on all positions for one model, return per-position accuracy."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    log.info(f"Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    
    n_layers = model.config.num_hidden_layers
    n_classes = len(RHYME_FAMILIES)
    
    # Figure out how many tokens "line2 prefix" adds by looking at a sample
    sample = tokenizer(examples[0]['prompt'], return_tensors='pt')
    sample_no_prefix = tokenizer(
        RHYME_PREFIX + examples[0]['line1'], return_tensors='pt')
    
    n_line2_tokens = sample.input_ids.shape[1] - sample_no_prefix.input_ids.shape[1]
    total_len = sample.input_ids.shape[1]
    
    log.info(f"Total seq length: {total_len}, line2 tokens: {n_line2_tokens}")
    
    # Target: positions that are IN LINE 2 (after the newline separating L1/L2)
    # Earlier: all positions before the first line2 token
    line2_start = total_len - n_line2_tokens
    
    # Collect activations per position per layer
    layer_pos_data = {}  # (layer, pos) -> (X_list, y_list)
    
    batch_size = 8
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        inputs = tokenizer([ex['prompt'] for ex in batch],
                           return_tensors='pt', padding=True,
                           truncation=True, max_length=200).to(device)
        
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        
        for layer_idx, hs in enumerate(out.hidden_states[1:], 1):
            h = hs.float().cpu().numpy()  # (batch, seq_len, dim)
            for b_idx, ex in enumerate(batch):
                y = ex['label']
                seq_len = inputs.attention_mask[b_idx].sum().item()
                for pos_idx in range(min(seq_len, h.shape[1])):
                    key = (layer_idx, pos_idx)
                    if key not in layer_pos_data:
                        layer_pos_data[key] = {'X': [], 'y': []}
                    layer_pos_data[key]['X'].append(h[b_idx, pos_idx])
                    layer_pos_data[key]['y'].append(y)
        
        if (i // batch_size) % 5 == 0:
            log.info(f"  {i}/{len(examples)} examples")
    
    del model; torch.cuda.empty_cache()
    
    # Probe each position using PCA + logistic regression with cross-validation
    results = {}
    
    # Maar-range layers (middle third)
    maar_layers = [l for l in range(1, n_layers+1)
                   if l >= n_layers // 3 and l <= 2 * n_layers // 3]
    
    for layer_idx in maar_layers:
        layer_accs = {}
        for pos_idx in range(total_len):
            key = (layer_idx, pos_idx)
            if key not in layer_pos_data: continue
            data = layer_pos_data[key]
            if len(data['X']) < 20: continue
            
            X = np.array(data['X'])
            y = np.array(data['y'])
            
            n_comp = min(64, X.shape[1], X.shape[0] - 1)
            pipe = Pipeline([
                ('pca', PCA(n_components=n_comp)),
                ('lr', LogisticRegression(max_iter=500, C=1.0, 
                                          class_weight='balanced'))
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
                layer_accs[pos_idx] = float(scores.mean())
            except Exception as e:
                continue
        
        if not layer_accs: continue
        
        # For each line2 target position, compute gap vs max-earlier
        for target_pos in range(line2_start, total_len):
            earlier_accs = {p: a for p, a in layer_accs.items() if p < target_pos}
            if not earlier_accs or target_pos not in layer_accs: continue
            
            target_acc = layer_accs[target_pos]
            max_earlier_acc = max(earlier_accs.values())
            max_earlier_pos = max(earlier_accs, key=earlier_accs.get)
            gap = target_acc - max_earlier_acc
            
            key_str = f"L{layer_idx}_pos{target_pos}"
            results[key_str] = {
                'layer': layer_idx,
                'target_pos': target_pos,
                'line2_start': line2_start,
                'target_acc': round(target_acc, 3),
                'max_earlier_acc': round(max_earlier_acc, 3),
                'max_earlier_pos': max_earlier_pos,
                'gap': round(gap * 100, 1),
                'n_examples': len(layer_pos_data[(layer_idx, target_pos)]['X'])
            }
    
    # Headline: best gap across line2 positions
    if results:
        best = max(results.values(), key=lambda r: r['gap'])
        worst = min(results.values(), key=lambda r: r['gap'])
        gaps = [r['gap'] for r in results.values()]
        log.info(f"Line-2 target headline gap: best={best['gap']:+.1f}pp, "
                 f"mean={np.mean(gaps):+.1f}pp, median={np.median(gaps):+.1f}pp")
    
    return {
        'model': model_id,
        'n_line2_tokens': n_line2_tokens,
        'line2_start_pos': line2_start,
        'total_seq_len': total_len,
        'per_position': results,
        'headline': {
            'best_gap': best['gap'] if results else None,
            'mean_gap': round(float(np.mean(gaps)), 1) if results else None,
            'median_gap': round(float(np.median(gaps)), 1) if results else None,
        } if results else {}
    }

def main():
    import torch
    
    log.info("=== Rhyme Line-2 Target Experiment ===")
    log.info("Addresses W1: does the rhyme gap persist at LINE 2 positions?")
    log.info("If yes → evidence of genuine planning (Line 2 has no rhyme word yet)")
    log.info("Earlier baseline INCLUDES 'sing' (Line 1 rhyme word)")
    
    examples = build_line2_examples()
    all_results = {}
    
    for model_id in MODELS:
        short = model_id.split('/')[-1]
        log.info(f"\n{'='*50}\nModel: {short}\n{'='*50}")
        t0 = time.time()
        try:
            res = probe_layer2(model_id, examples)
            all_results[model_id] = res
            log.info(f"Done in {(time.time()-t0)/60:.1f}min")
            log.info(f"Headline: {res['headline']}")
        except Exception as e:
            log.error(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[model_id] = {'error': str(e)}
        
        # Save after each model
        OUTFILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(all_results, open(OUTFILE, 'w'), indent=2, default=str)
        log.info(f"Saved to {OUTFILE}")
    
    # Summary
    log.info("\n=== SUMMARY: Line-2 Rhyme Gap ===")
    log.info("A positive gap means: Line-2 positions encode rhyme family")
    log.info("BETTER than any Line-1 position (including the rhyme word itself)")
    log.info("This would be evidence of genuine forward planning.\n")
    for model_id, res in all_results.items():
        if 'error' in res: continue
        h = res.get('headline', {})
        print(f"  {model_id.split('/')[-1]:>20s}: "
              f"best={h.get('best_gap','?'):>+6}pp  "
              f"mean={h.get('mean_gap','?'):>+6}pp  "
              f"median={h.get('median_gap','?'):>+6}pp")

if __name__ == '__main__':
    main()
