import json, os, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

sys.path.insert(0, os.getcwd())
from transformer_lens import HookedTransformer

MODEL_NAME = "EleutherAI/gpt-j-6b"
PCA_DIM = 128

logger.info("Loading GPT-J-6B...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.float16)
model.eval()
n_layers = model.cfg.n_layers
layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                     2*n_layers//3, 5*n_layers//6, n_layers-1]))
W_E = model.W_E.detach()
W_U = model.W_U.detach()

results = {}

# ============================================================
# FIX 4 REMAINING: K=5, layers 18,23,27 only
# ============================================================
logger.info("FIX 4 REMAINING: K=5, layers 18,23,27")

PROMPTS = [
    "The process of photosynthesis in plants involves the conversion of",
    "Quantum mechanics fundamentally changed our understanding of physics by",
    "The theory of plate tectonics explains how the earth surface",
    "In molecular biology, DNA replication is the process by which",
    "The second law of thermodynamics states that entropy in an",
    "Neurons communicate with each other through electrical and chemical",
    "The periodic table organizes all known chemical elements according to",
    "General relativity predicts that massive objects cause a distortion in",
    "Evolution through natural selection occurs when organisms with favorable",
    "The human immune system protects the body from pathogens by",
    "The Industrial Revolution transformed European society beginning in the",
    "Ancient Rome expanded its territory through military conquest and",
    "The French Revolution of 1789 was caused by widespread discontent",
    "World War Two ended in 1945 after the Allied powers",
    "The Renaissance was a period of cultural and intellectual rebirth",
    "The fall of the Berlin Wall in 1989 symbolized the",
    "The American Civil War was fought between the northern and",
    "The Age of Exploration led European nations to discover new",
    "The Cold War was a period of geopolitical tension between",
    "Ancient Egyptian civilization developed along the banks of the",
    "Artificial intelligence systems learn from data by identifying patterns",
    "The internet was originally developed as a military communication network",
    "Machine learning algorithms can be broadly classified into supervised and",
    "Cloud computing allows organizations to access computing resources over",
    "Blockchain technology creates a decentralized and distributed digital ledger",
    "The discovery of penicillin by Alexander Fleming revolutionized modern",
    "The Silk Road was an ancient network of trade routes",
    "Volcanic eruptions occur when molten rock from deep within the",
    "The theory of evolution by natural selection was first proposed",
    "Antibiotics are medications designed to fight bacterial infections by",
    "The human genome project was completed in 2003 and mapped",
    "Plate boundaries are the edges where two tectonic plates meet",
    "The Enlightenment was an intellectual movement that emphasized reason and",
    "Nuclear fusion is the process that powers stars including our",
    "The agricultural revolution transformed human society from nomadic hunter",
    "Electromagnetic radiation includes visible light radio waves and gamma",
    "The invention of the printing press by Gutenberg around 1440",
    "Coral reefs are diverse underwater ecosystems held together by calcium",
    "The Pythagorean theorem states that in a right triangle the",
    "Mitochondria are often called the powerhouse of the cell because",
    "The French and Indian War was fought between British and",
    "Superconductivity is a phenomenon where electrical resistance drops to zero",
    "The Doppler effect explains why the sound of an ambulance",
    "Tectonic activity along the Pacific Ring of Fire causes frequent",
    "The development of vaccines has been one of the greatest",
    "Photovoltaic cells convert sunlight directly into electrical energy using",
    "The Ottoman Empire lasted for over six centuries and controlled",
    "Genetic engineering allows scientists to modify the DNA of organisms",
    "The water cycle describes how water evaporates from the surface",
    "Black holes form when massive stars exhaust their nuclear fuel",
    "Neural networks are computational models inspired by the structure of",
    "The development of transistors in the 1950s revolutionized electronics",
    "Programming languages provide abstractions that allow developers to write",
    "Cybersecurity involves protecting computer systems and networks from digital",
    "Quantum computing leverages quantum mechanical phenomena to perform calculations",
    "The largest ocean on Earth is the Pacific Ocean which",
    "Democracy as a form of government gives citizens the power",
    "The global economy is interconnected through international trade and",
    "Climate change is primarily driven by the emission of greenhouse",
    "The United Nations was established to promote international peace and",
    "Education plays a crucial role in the development of modern",
    "The stock market serves as a platform where investors can",
    "Renewable energy sources include solar wind and hydroelectric power",
    "The human brain is the most complex organ in the",
    "Globalization has increased the interconnectedness of economies and cultures",
    "In Python programming the def keyword is used to define",
    "A function that takes a list of numbers and returns",
    "The algorithm works by first sorting the input array and",
    "To implement a binary search tree you need to define",
    "The main difference between a list and a tuple in",
    "Object oriented programming encapsulates data and behavior within classes",
    "A recursive function must have a base case to prevent",
    "The time complexity of quicksort in the average case is",
    "Database normalization reduces data redundancy by organizing tables according",
    "Version control systems like git track changes to source code",
    "The Magna Carta was signed in 1215 and established the",
    "Insulin is a hormone produced by the pancreas that regulates",
    "The speed of sound varies depending on the medium through",
    "Natural selection acts on phenotypic variation within a population",
    "The Rosetta Stone was discovered in 1799 and helped scholars",
    "Entropy is a measure of disorder in a thermodynamic system",
    "The Great Depression of the 1930s was triggered by the",
    "Stem cells are undifferentiated cells that can develop into specialized",
    "The circulatory system transports blood oxygen and nutrients throughout",
    "Fibonacci numbers appear frequently in nature including the arrangement",
    "The Treaty of Versailles was signed in 1919 and imposed",
    "Catalysts are substances that increase the rate of a chemical",
    "The Hubble Space Telescope has provided stunning images of distant",
    "Biodiversity refers to the variety of life forms found in",
    "The human nervous system consists of the central and peripheral",
    "Archimedes principle states that an object submerged in a fluid",
    "The Protestant Reformation began in 1517 when Martin Luther posted",
    "Semiconductors are materials with electrical conductivity between conductors and",
    "The greenhouse effect traps heat in the earth atmosphere causing",
    "Osmosis is the movement of water molecules across a selectively",
    "The Apollo 11 mission in 1969 successfully landed humans on",
    "Cryptography is the practice of securing communication through the use",
    "The circulatory system in humans consists of the heart blood",
    "Magnetism is a fundamental force that arises from the motion",
    "The Industrial Revolution began in Britain due to several factors",
    "Photosynthesis occurs primarily in the chloroplasts of plant cells where",
]

k = 5
remaining_layers = [18, 23, 27]
last_layer = n_layers - 1

logger.info("Generating sequences...")
all_sequences = []
for pi, prompt in enumerate(PROMPTS):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        gen = model.generate(tokens, max_new_tokens=80, temperature=0.0)
    all_sequences.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
    if (pi+1) % 20 == 0: logger.info(f"  {pi+1}/101")

train_seqs = [all_sequences[i] for i in range(50)]
test_seqs = [all_sequences[i] for i in range(50, 101)]

# Collect train/test data for remaining layers
for split_name, seqs in [("train", train_seqs), ("test", test_seqs)]:
    data = {l: {"source": [], "target_hidden": [], "target_token": []} for l in remaining_layers}
    for seq in seqs:
        ids = seq["full_ids"]; pl = seq["prompt_len"]
        inp = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            _, cache = model.run_with_cache(inp,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in remaining_layers + [last_layer]])
        for n in range(pl, len(ids) - k):
            tgt_tok = ids[n+k]
            tgt_h = cache[f"blocks.{last_layer}.hook_resid_post"][0, n+k, :].cpu().numpy()
            for l in remaining_layers:
                data[l]["source"].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                data[l]["target_hidden"].append(tgt_h)
                data[l]["target_token"].append(tgt_tok)
        del cache; torch.cuda.empty_cache()
    if split_name == "train": train_data = data
    else: test_data = data

logger.info(f"\nFix 4 K=5 remaining layers:")
logger.info(f"  {'Layer':>6} {'FL_acc':>8} {'Probe_acc':>10}")
fix4_results = {}
for l in remaining_layers:
    X_tr = np.stack(train_data[l]["source"])
    Y_tr = np.stack(train_data[l]["target_hidden"])
    X_te = np.stack(test_data[l]["source"])
    test_tokens = np.array(test_data[l]["target_token"])
    
    pca_s = PCA(n_components=min(PCA_DIM, X_tr.shape[0]-1), random_state=42)
    X_tr_p = pca_s.fit_transform(StandardScaler().fit_transform(X_tr))
    X_te_p = pca_s.transform(StandardScaler().fit_transform(X_te))
    pca_t = PCA(n_components=min(PCA_DIM, Y_tr.shape[0]-1), random_state=42)
    Y_tr_p = pca_t.fit_transform(StandardScaler().fit_transform(Y_tr))
    
    ridge = Ridge(alpha=1.0); ridge.fit(X_tr_p, Y_tr_p)
    Y_pred = pca_t.inverse_transform(ridge.predict(X_te_p))
    pred_logits = Y_pred @ W_U.cpu().numpy()
    fl_acc = float(np.mean(pred_logits.argmax(axis=1) == test_tokens))
    
    tc = Counter(test_tokens)
    freq = {t for t,c in tc.items() if c >= 5}
    mask = np.array([t in freq for t in test_tokens])
    if mask.sum() > 50:
        t2i = {t:i for i,t in enumerate(sorted(freq))}
        y_f = np.array([t2i[t] for t in test_tokens[mask]])
        mc = min(Counter(y_f).values())
        if mc >= 2:
            cv = StratifiedKFold(n_splits=min(5,mc), shuffle=True, random_state=42)
            probe_acc = cross_val_score(LogisticRegression(C=1.0,max_iter=5000,solver="lbfgs"),
                X_te_p[mask], y_f, cv=cv, scoring="accuracy").mean()
        else: probe_acc = -1
    else: probe_acc = -1
    
    logger.info(f"  L{l:>4} {fl_acc:>8.4f} {probe_acc:>10.4f}")
    fix4_results[str(l)] = {"fl_acc": float(fl_acc), "probe_acc": float(probe_acc)}

results["fix4_k5_remaining"] = fix4_results

# ============================================================
# FIX 5: MULTIPLE FIXED POSITIONS
# ============================================================
logger.info("\nFIX 5: MULTIPLE FIXED POSITIONS")

SIGS = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'),
    ('def greet(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def get_evens(numbers):', 'list'), ('def flatten(nested):', 'list'),
    ('def average(numbers):', 'float'), ('def distance(x1, y1, x2, y2):', 'float'),
    ('def multiply(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def reverse_string(s):', 'str'), ('def join_words(words):', 'str'),
    ('def is_sorted(items):', 'bool'), ('def contains(items, target):', 'bool'),
    ('def sort_ascending(items):', 'list'), ('def unique(items):', 'list'),
    ('def median(numbers):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'),
    ('def capitalize(text):', 'str'), ('def slug(text):', 'str'),
    ('def is_empty(s):', 'bool'), ('def is_palindrome(s):', 'bool'),
    ('def reverse_list(items):', 'list'), ('def zip_lists(a, b):', 'list'),
    ('def variance(numbers):', 'float'), ('def sigmoid(x):', 'float'),
]

targets = sorted(set(r for _,r in SIGS))
t2i = {t:i for i,t in enumerate(targets)}
labels = np.array([t2i[r] for _,r in SIGS])

FIXED_POSITIONS = [0, 1, 2, 3, 5]
GEN_STEPS = [0, 5, 10, 15]
test_layer = layers[len(layers)//2]

logger.info(f"  {len(SIGS)} sigs, layer {test_layer}, positions {FIXED_POSITIONS}, steps {GEN_STEPS}")

fix5_results = {}
for gs in GEN_STEPS:
    logger.info(f"\n  Gen step {gs}:")
    pos_results = {}
    for fp in FIXED_POSITIONS:
        acts = []
        for sig, ret in SIGS:
            prompt = sig + "\n    "
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                for s in range(gs):
                    logits = model(tokens)
                    nt = logits[0,-1,:].argmax().item()
                    tokens = torch.cat([tokens, torch.tensor([[nt]], device="cuda")], dim=1)
                _, cache = model.run_with_cache(tokens,
                    names_filter=[f"blocks.{test_layer}.hook_resid_post"])
            if fp < tokens.shape[1]:
                act = cache[f"blocks.{test_layer}.hook_resid_post"][0, fp, :].cpu().numpy()
            else:
                act = np.zeros(model.cfg.d_model)
            acts.append(act)
            del cache; torch.cuda.empty_cache()
        
        X = np.stack(acts)
        X_s = StandardScaler().fit_transform(X)
        if X_s.shape[1] > PCA_DIM:
            X_s = PCA(n_components=min(PCA_DIM, X_s.shape[0]-1), random_state=42).fit_transform(X_s)
        cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())), shuffle=True, random_state=42)
        acc = cross_val_score(LogisticRegression(C=1.0,max_iter=5000,solver="lbfgs"),
            X_s, labels, cv=cv, scoring="accuracy").mean()
        logger.info(f"    pos={fp}: {acc:.3f}")
        pos_results[str(fp)] = float(acc)
    fix5_results[f"step_{gs}"] = pos_results

results["fix5_fixed_positions"] = fix5_results

# Summary
logger.info(f"\n  === FIXED POSITION SUMMARY (Layer {test_layer}) ===")
logger.info(f"  {'Step':>6}" + "".join(f"  pos={p:>2}" for p in FIXED_POSITIONS))
for gs in GEN_STEPS:
    vals = fix5_results[f"step_{gs}"]
    line = f"  {gs:>6}"
    for p in FIXED_POSITIONS:
        line += f"  {vals[str(p)]:>6.3f}"
    logger.info(line)

# Save
outfile = "results/lookahead/final/reviewer_fixes.json"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    json.dump(results, f, indent=2, default=str)
logger.info(f"\nSaved to {outfile}")
logger.info("DONE — REMAINING FIXES")
