"""
Phase 1: Collect activations from Gemma 2 9B IT at key layers.
Phase 2: Train SAEs on collected activations.

Collects residual stream and MLP output activations at layers 9, 20, 31
by running diverse text through the model. Trains 16k-width SAEs using
a simple sparse autoencoder implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, ast, random, gc, argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--n-tokens", type=int, default=200000,
                   help="Target number of activation vectors to collect")
    p.add_argument("--sae-width", type=int, default=16384)
    p.add_argument("--layers", default="9,20,31")
    p.add_argument("--train-epochs", type=int, default=5)
    p.add_argument("--train-lr", type=float, default=3e-4)
    p.add_argument("--l1-coeff", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--out-dir", default="results/custom_it_sae")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ── Simple SAE ──────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.b_dec
        return nn.functional.relu(self.encoder(x_centered))

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_dec
        return x_hat, z


def train_sae(activations, d_sae, n_epochs, lr, l1_coeff, batch_size, device="cpu"):
    """Train a sparse autoencoder on collected activations."""
    n, d_in = activations.shape
    print(f"    Training SAE: {n} samples, d_in={d_in}, d_sae={d_sae}")

    sae = SparseAutoencoder(d_in, d_sae).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    activations = activations.to(device)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0
        total_recon = 0
        total_l1 = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch = activations[perm[i:i + batch_size]]
            x_hat, z = sae(batch)

            recon_loss = (batch - x_hat).pow(2).mean()
            l1_loss = z.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder columns
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_l1 += l1_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches

        # Sparsity stats
        with torch.no_grad():
            sample = activations[:1000]
            z = sae.encode(sample)
            avg_active = (z > 0).float().sum(1).mean().item()
            frac_dead = ((z > 0).float().sum(0) == 0).float().mean().item()

        print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
              f"recon={avg_recon:.4f} l1={avg_l1:.4f} "
              f"active={avg_active:.0f}/{d_sae} dead={frac_dead:.1%}")

    return sae.cpu()


# ── Activation collection ───────────────────────────────────────────

class MultiLayerCapture:
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        self.res_acts = {li: [] for li in target_layers}
        self.mlp_acts = {li: [] for li in target_layers}
        self.hooks = []
        self.enabled = False

        for li in target_layers:
            # Residual: after full layer
            self.hooks.append(
                model.model.layers[li].register_forward_hook(self._res_hook(li)))
            # MLP: after MLP sublayer
            self.hooks.append(
                model.model.layers[li].mlp.register_forward_hook(self._mlp_hook(li)))

    def _res_hook(self, li):
        def hook(mod, inp, out):
            if not self.enabled: return
            hs = out[0] if isinstance(out, tuple) else out
            # Save ALL token positions (not just last)
            self.res_acts[li].append(hs[0].detach().float().cpu())
        return hook

    def _mlp_hook(self, li):
        def hook(mod, inp, out):
            if not self.enabled: return
            o = out[0] if isinstance(out, tuple) else out
            self.mlp_acts[li].append(o[0].detach().float().cpu())
        return hook

    def remove(self):
        for h in self.hooks: h.remove()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.layers.split(",")]

    # ── Phase 1: Collect activations ────────────────────────────────
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    capture = MultiLayerCapture(model, target_layers)

    # Use diverse data sources for training
    print("Collecting activations from diverse text...")
    texts = []

    # DDXPlus cases
    with open("release_evidences.json") as f:
        raw_ev = json.load(f)
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    for i in range(min(200, len(ds))):
        row = ds[i]
        evs = ast.literal_eval(row["EVIDENCES"])
        ev_text = ", ".join(evs[:10])
        texts.append(f"Patient: {row['AGE']}-year-old. Pathology: {row['PATHOLOGY']}. Evidence: {ev_text}")

    # MMLU questions
    for subj in ["high_school_psychology", "college_biology", "nutrition"]:
        mmlu = load_dataset("cais/mmlu", subj, split="test")
        for row in mmlu:
            texts.append(f"Question: {row['question']}\nA) {row['choices'][0]}\nB) {row['choices'][1]}")

    # NarrativeQA stories
    try:
        nqa = load_dataset("deepmind/narrativeqa", split="test")
        for i in range(min(100, len(nqa))):
            texts.append(nqa[i]["document"]["summary"]["text"][:500])
    except Exception:
        pass

    random.Random(42).shuffle(texts)
    print(f"  {len(texts)} text samples")

    # Run through model, collecting activations at all token positions
    total_tokens = 0
    batch_texts = []

    for i, text in enumerate(texts):
        if total_tokens >= args.n_tokens:
            break

        # Gemma: no system role
        conv = [{"role": "user", "content": text}]
        encoded = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(encoded, return_tensors="pt", truncation=True,
                       max_length=512).input_ids.to(args.device)

        capture.enabled = True
        with torch.no_grad():
            _ = model(ids, use_cache=False)
        capture.enabled = False
        del _

        total_tokens += ids.shape[1]

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(texts)}, ~{total_tokens} tokens")

        torch.cuda.empty_cache()

    print(f"  Total tokens collected: {total_tokens}")

    # Concatenate and save activations
    act_data = {}
    for li in target_layers:
        res = torch.cat(capture.res_acts[li], dim=0)
        mlp = torch.cat(capture.mlp_acts[li], dim=0)
        act_data[f"res_{li}"] = res
        act_data[f"mlp_{li}"] = mlp
        print(f"  Layer {li}: res={res.shape}, mlp={mlp.shape}")

    capture.remove()

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # Save raw activations
    torch.save(act_data, out_dir / "activations.pt")
    print(f"Activations saved to {out_dir}/activations.pt")

    # ── Phase 2: Train SAEs ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Phase 2: Training SAEs")
    print(f"{'='*70}")

    # Train on GPU if available, otherwise CPU
    train_device = args.device

    sae_models = {}
    for li in target_layers:
        for sublayer in ["res", "mlp"]:
            key = f"{sublayer}_{li}"
            acts = act_data[key]

            # Subsample if too many
            if acts.shape[0] > args.n_tokens:
                perm = torch.randperm(acts.shape[0])[:args.n_tokens]
                acts = acts[perm]

            print(f"\n  Training SAE for {key} ({acts.shape})...")
            sae = train_sae(
                acts, args.sae_width, args.train_epochs,
                args.train_lr, args.l1_coeff, args.batch_size,
                device=train_device)
            sae_models[key] = sae

            # Save individual SAE
            torch.save({
                "state_dict": sae.state_dict(),
                "d_in": acts.shape[1],
                "d_sae": args.sae_width,
                "layer": li,
                "sublayer": sublayer,
            }, out_dir / f"sae_{key}.pt")
            print(f"  Saved {key}")

            torch.cuda.empty_cache()

    print(f"\nAll SAEs saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
