# LoReFT commonsense reproduction (subset-first, Llama-2-7B) — drafted here, run on cloud GPU

## 1. Problem statement
The register-vs-procedure claim rests on one point per side: the conditional linear map works on
refusal (0.62 @ benign 0.02) and fails on GSM8K procedure transfer (ridge/MLP/DAgger ≈ 0; DAS = 0).
LoReFT's published commonsense gains are the predicted-success case for the *same map family*
(`h ← h + (Wh + b − hR)Rᵀ` is an input-conditional affine edit in a learned subspace) on
single-token disposition-shaped tasks. Reproduce those gains on Llama-2-7B with a subset of
commonsense-170k to fill the middle row of the matrix:

| | conditional linear map | fixed vector / subspace |
|---|---|---|
| refusal (register) | works (ridge 0.62) | fails (CAA/Arditi 0.00) |
| commonsense (single token) | **this experiment** | — |
| GSM8K (procedure) | fails (≈0) | fails (PCA<512, DAS=0) |

## 2. Approach (decisions locked)
- **Model:** NousResearch/Llama-2-7b-hf (same ungated mirror as the GSM8K/refusal work), frozen, bf16.
- **Intervention:** native LoReFT reusing `OrthoSubspace` from `das_subspace.py` (pyreft is stale vs
  the pinned transformers>=5.0.0). Per layer: R (4096×8, orthonormal via QR) + Linear(4096→8).
  Forward: `h + (source(h) − hR)Rᵀ`, computed f32, cast back to model dtype.
- **Paper recipe (pyreft examples/loreft README, commonsense task):** ALL 32 layers, rank 8,
  positions f7+l7 (first 7 + last 7 *prompt* tokens), shared weights across positions (one
  intervention per layer), lr 9e-4, 6 epochs, batch 16 × grad-accum 2, dropout 0, warmup 0.1,
  greedy decoding, max_new 32.
- **Data (LLM-Adapters, format verified by direct fetch):** items are
  `{"instruction", "input"(empty), "output": "the correct answer is X", "answer": "X"}`.
  Prompt template is literally `"%s\n" % instruction` (NO alpaca wrapper — verified in
  pyreft task_config.py). Trigger for answer extraction: `"the correct answer is "`.
  Answers: boolq true/false; piqa solution1/solution2; ARC-Challenge answer1..answer5.
- **Subset-first:** train on 20k of commonsense-170k; eval BoolQ / PIQA / ARC-Challenge test sets,
  base zero-shot vs LoReFT-steered. Scale to full 170k + 8 benchmarks only if direction confirms.
- **Padding:** LEFT for both train and eval; intervention locations offset by per-sample pad length.
  Prefill-only guard in the hook (skip when seq_len ≤ max location) so decode steps pass through.

## 3. Concrete traces
- BoolQ eval sample 0 ("does ethanol take more energy make that produces?"): prompt =
  instruction + "\n"; model generates; extractor takes the word after "the correct answer is" →
  compare to gold "false" → 1/0.
- Train sample: input_ids = prompt_ids + target_ids(+eos); labels = ids with prompt+pad = −100;
  intervention edits positions {pad+0..pad+6} ∪ {pad+plen−7..pad+plen−1} during the forward.
- Param count at d=4096, r=8 per layer: 4096·8 (R raw) + 8·4096 + 8 (linear) = 65,544;
  ×32 layers ≈ 2.1M trainable (asserted in tests).

## 4. Files
- `src/probes/attribution/loreft_intervention.py` — LoReFTIntervention, PositionEditHook,
  intervention_locations, response_labels
- `src/probes/attribution/commonsense_data.py` — load/format/extract/score/subset (torch-free)
- `configs/attribution/loreft_commonsense_llama2.yaml`
- `scripts/attribution/download_commonsense_data.py` → data/commonsense/
- `scripts/attribution/train_loreft_commonsense.py`, `scripts/attribution/eval_commonsense_suite.py`
- `tests/test_loreft_intervention.py`, `tests/test_commonsense_data.py` (CPU, no network)

## 5. Non-goals
No LoRA baseline this phase; no CoT commonsense; no arithmetic; no full 8-benchmark suite; no
pyreft dependency; no changes to lockstep/PCA/DAS code beyond imports; no ridge-map-to-δ fit yet.

## 6. Acceptance criteria
- All new tests pass locally on CPU.
- Trainable param count matches formula exactly (test-asserted).
- On the cloud GPU: subset-trained LoReFT beats base zero-shot clearly on ≥2 of 3 benchmarks,
  direction consistent with the paper's Llama-2-7B commonsense row (subset-scaled margins).
- Writeup `results/attribution/2026-06-XX-loreft-commonsense.md` in the existing format.

## 7. Cloud GPU run sequence (user runs; nothing executed locally)
```bash
uv run python -m scripts.attribution.download_commonsense_data            # → data/commonsense/
uv run python -m scripts.attribution.train_loreft_commonsense \
    --config configs/attribution/loreft_commonsense_llama2.yaml           # subset train (20k)
uv run python -m scripts.attribution.eval_commonsense_suite \
    --config configs/attribution/loreft_commonsense_llama2.yaml --interventions none   # base
uv run python -m scripts.attribution.eval_commonsense_suite \
    --config configs/attribution/loreft_commonsense_llama2.yaml \
    --interventions results/attribution/loreft_commonsense/loreft_interventions.pt     # steered
```
