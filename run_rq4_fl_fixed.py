#!/usr/bin/env python3
"""
FUTURE LENS — FIXED PAL ET AL. METHOD
=======================================
Fix: PCA only on source side. Target stays full 4096 dims.
Ridge: 128 → 4096, then decode through W_U.

Also try NO PCA at all: Ridge 4096 → 4096 with strong regularization.
"""

import json, os, time
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

MODEL_NAME = "EleutherAI/gpt-j-6b"
K_VALUES = [1, 3, 5]

PROMPTS = [
    # TRAIN (0-49)
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
    # TEST (50-99)
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


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS — FIXED PAL ET AL. METHOD")
    logger.info("=" * 70)

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.float16)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    last_layer = n_layers - 1

    W_U = model.W_U.detach().cpu().numpy()  # [d_model, vocab]
    logger.info(f"  {n_layers} layers, d_model={d_model}, W_U shape={W_U.shape}")

    # Generate
    logger.info(f"\n  Generating 80 tokens for {len(PROMPTS)} prompts...")
    all_sequences = []
    for pi, prompt in enumerate(PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=80, temperature=0.0)
        all_sequences.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        if (pi + 1) % 20 == 0:
            logger.info(f"    {pi+1}/{len(PROMPTS)}")

    train_seqs = [all_sequences[i] for i in range(50)]
    test_seqs = [all_sequences[i] for i in range(50, len(PROMPTS))]

    all_results = {"model": MODEL_NAME}

    for k in K_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"K={k}")
        logger.info(f"{'='*60}")

        # Collect train and test data
        for split_name, seqs in [("train", train_seqs), ("test", test_seqs)]:
            data = {l: {"source": [], "target_hidden": [], "target_token": []} for l in layers}
            for seq in seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    _, cache = model.run_with_cache(inp,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers + [last_layer]])
                for n in range(pl, len(ids) - k):
                    tgt_h = cache[f"blocks.{last_layer}.hook_resid_post"][0, n+k, :].cpu().numpy()
                    tgt_tok = ids[n+k]
                    for l in layers:
                        data[l]["source"].append(
                            cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                        data[l]["target_hidden"].append(tgt_h)
                        data[l]["target_token"].append(tgt_tok)
                del cache; torch.cuda.empty_cache()
            if split_name == "train":
                train_data = data
            else:
                test_data = data

        n_train = len(train_data[layers[0]]["source"])
        n_test = len(test_data[layers[0]]["source"])
        logger.info(f"  Train: {n_train}, Test: {n_test}")

        logger.info(f"\n  {'Layer':>6} {'PCA_src':>10} {'Full_dim':>10} {'Our_probe':>10}")

        k_results = {}
        for l in layers:
            X_train = np.stack(train_data[l]["source"])      # [n_train, 4096]
            Y_train = np.stack(train_data[l]["target_hidden"]) # [n_train, 4096]
            X_test = np.stack(test_data[l]["source"])          # [n_test, 4096]
            test_tokens = np.array(test_data[l]["target_token"])

            # Standardize source
            scaler_src = StandardScaler()
            X_train_s = scaler_src.fit_transform(X_train)
            X_test_s = scaler_src.transform(X_test)

            # Standardize target
            scaler_tgt = StandardScaler()
            Y_train_s = scaler_tgt.fit_transform(Y_train)

            # === METHOD A: PCA source only, full-dim target ===
            pca_src = PCA(n_components=128, random_state=42)
            X_train_pca = pca_src.fit_transform(X_train_s)    # [n_train, 128]
            X_test_pca = pca_src.transform(X_test_s)          # [n_test, 128]

            # Ridge: 128 → 4096
            ridge_a = Ridge(alpha=10.0)
            ridge_a.fit(X_train_pca, Y_train_s)
            Y_pred_s = ridge_a.predict(X_test_pca)            # [n_test, 4096]
            Y_pred = scaler_tgt.inverse_transform(Y_pred_s)   # back to original scale

            # Decode through W_U
            logits_a = Y_pred @ W_U                           # [n_test, vocab]
            preds_a = logits_a.argmax(axis=1)
            acc_a = float(np.mean(preds_a == test_tokens))

            # === METHOD B: Full dim, no PCA ===
            ridge_b = Ridge(alpha=100.0)  # stronger reg for 4096→4096
            ridge_b.fit(X_train_s, Y_train_s)
            Y_pred_b_s = ridge_b.predict(X_test_s)
            Y_pred_b = scaler_tgt.inverse_transform(Y_pred_b_s)
            logits_b = Y_pred_b @ W_U
            preds_b = logits_b.argmax(axis=1)
            acc_b = float(np.mean(preds_b == test_tokens))

            # === Our probe (classification) for comparison ===
            # Quick: top-1 accuracy on frequent tokens using logistic regression
            tc = Counter(test_tokens)
            freq = {t for t, c in tc.items() if c >= 10}
            if len(freq) >= 5:
                t2i = {t: i for i, t in enumerate(sorted(freq))}
                mask = np.array([t in freq for t in test_tokens])
                y_f = np.array([t2i[t] for t in test_tokens[mask]])
                X_f = X_test_pca[mask]
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                from sklearn.linear_model import LogisticRegression
                mc = min(Counter(y_f).values())
                if mc >= 2:
                    cv = StratifiedKFold(n_splits=min(5, mc), shuffle=True, random_state=42)
                    probe_acc = cross_val_score(
                        LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                        X_f, y_f, cv=cv, scoring="accuracy").mean()
                else:
                    probe_acc = -1
            else:
                probe_acc = -1

            logger.info(f"  L{l:>4} {acc_a:>10.4f} {acc_b:>10.4f} {probe_acc:>10.4f}")

            k_results[str(l)] = {
                "pca_source_only": float(acc_a),
                "full_dim": float(acc_b),
                "our_probe": float(probe_acc),
            }

        all_results[f"k{k}"] = k_results

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'K':>3} {'Layer':>6} {'PCA_src':>10} {'Full_dim':>10} {'Our_probe':>10}")
    for k in K_VALUES:
        key = f"k{k}"
        if key not in all_results:
            continue
        for l_str, r in all_results[key].items():
            logger.info(f"{k:>3} L{l_str:>4} {r['pca_source_only']:>10.4f} "
                        f"{r['full_dim']:>10.4f} {r['our_probe']:>10.4f}")

    outfile = "results/lookahead/final/future_lens_fixed_method.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    logger.info("\nDONE — FIXED FUTURE LENS METHOD")


if __name__ == "__main__":
    main()
