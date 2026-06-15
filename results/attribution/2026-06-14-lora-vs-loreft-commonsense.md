# Do LoRA and LoReFT make the same commonsense edit? (activation similarity)

**Question.** Both LoRA and LoReFT solve commonsense — the register/disposition row of the
register-vs-procedure matrix. Do the two adaptation methods make the *same* change to activations, or
reach the same behavior by different representational routes? Compare each method's induced edit
δ = h_steered − h_base on commonsense prompts.

**Apparatus.** Matched arms on Llama-2-7B: the existing paper-faithful LoReFT (all 32 layers, f7+l7,
rank 8; `loreft_interventions.pt`) and a **newly trained LoRA** (`train_lora_commonsense.py`) on the
*same* 20k commonsense-170k subset, prompt template, "the correct answer is X" target, and
response-only CE labels (it reuses the LoReFT collate, so the supervised signal is byte-identical;
the only difference is the method — LLM-Adapters LoRA r=32/α32 on {q,k,v,up,down}_proj). Both are
*working* edits (base = 0.00 on all three; 200-item check below). `compare_lora_loreft_commonsense.py`
captures base / LoRA / LoReFT residuals at LoReFT's f7+l7 positions on n=60 prompts (20 each from
BoolQ/PIQA/ARC-Challenge), over probe layers {4,8,12,16,20,24,28}, and computes per-layer:
mean per-token cosine(δ), linear CKA(δ) (rotation/scale-invariant), top-8 PCA-subspace overlap(δ),
and — as a control — linear CKA on the steered reps h. New code: `activation_similarity.py`
(`linear_cka`, `subspace_overlap`; 7 tests), driver + plot, `tests/test_lora_commonsense.py` (2),
`tests/test_activation_similarity.py` (7).

**Accuracy (both work; 200 items/dataset, vs LoReFT full-split).**

| | BoolQ | PIQA | ARC-C | base |
|---|--:|--:|--:|--:|
| LoRA | 0.680 | 0.790 | 0.655 | 0.00 |
| LoReFT | 0.667 | 0.798 | 0.602 | 0.00 |

## Result

| layer | cosine(δ) | CKA(δ) | subspace ovl(δ) | CKA(h) | ‖δ‖_LoRA | ‖δ‖_LoReFT |
|--:|--:|--:|--:|--:|--:|--:|
| 4  | 0.106 | **0.959** | 0.185 | 1.000 | 3.16  | 5.08 |
| 8  | 0.183 | **0.929** | 0.170 | 1.000 | 8.22  | 12.83 |
| 12 | 0.261 | 0.757 | 0.157 | 1.000 | 13.78 | 23.70 |
| 16 | 0.308 | 0.468 | 0.171 | 1.000 | 23.64 | 38.83 |
| 20 | 0.377 | 0.270 | 0.208 | 1.000 | 34.64 | 61.52 |
| 24 | 0.390 | 0.177 | 0.232 | 1.000 | 50.79 | 93.53 |
| 28 | 0.390 | 0.129 | 0.249 | 0.999 | 72.47 | 133.68 |

Figure: `results/figures/lora_vs_loreft_similarity.png`.

## Reading — same behavior, different representational routes (with a shared early-layer skeleton)

**CKA(h) ≡ 1.0 is a confounded control, not signal.** Both steered reps are base + a small edit
(‖δ‖ ≪ ‖h‖, and both run the same prompts through near-identical networks), so h_LoRA and h_LoReFT
are near-identical *because both ≈ base* — CKA(h) stays 1.000 even at L28 where the edits are most
divergent (CKA(δ)=0.13). It measures base dominance, not edit agreement (cf. the E1 emitted-token
confound in the L20 work). The **δ** metrics carry the answer.

**The edits are NOT the same; they share only an early-layer skeleton.** Reading the δ columns:
- **CKA(δ) collapses with depth: 0.96 (L4) → 0.13 (L28).** Early layers, the two edits share token-
  relational structure; from mid-network on, they don't.
- **Per-token cosine is low throughout (0.11–0.39)** and **top-8 subspace overlap is low throughout
  (0.16–0.25).** The methods push the same token in different directions, and their dominant edit
  directions barely overlap, at *every* depth.

The early-layer combination — **high CKA(δ) with low cosine and low subspace overlap** — is the
signature of an approximate **rotation**: if δ_LoReFT ≈ δ_LoRA·Q for an orthogonal Q, the token gram
matrix (hence CKA) is preserved while per-token direction and feature-space principal axes are not.
So in early layers the two methods make a *structurally equivalent edit in a rotated basis* (CKA
0.93–0.96 at L4–8) — plausibly the shared "register/format" nudge that installs the disposition to
emit "the correct answer is X". By mid-to-late layers even that equivalence dissolves (CKA(δ)→0.13):
the task-specific commonsense content is written through genuinely different activation changes.
Throughout, LoReFT's edit is ~1.6–1.8× larger in magnitude than LoRA's.

## Verdict
**LoRA and LoReFT reach matched commonsense accuracy through different representational edits.** There
is no canonical "commonsense edit direction" both methods find: per-token directions are weakly
aligned and dominant edit subspaces barely overlap at any layer. The only shared structure is an
**early-layer, rotation-equivalent skeleton** (CKA(δ) 0.93–0.96 at L4–8) that decays monotonically
with depth to near-zero by L28. Consistent with the register-vs-procedure thesis — a register/
disposition task is solvable by many methods, but they implement it via divergent deeper-layer edits
sharing only a shallow structural component, not a common direction. Caveats: n=60 prompts, 7 probe
layers, f7+l7 positions (LoReFT's sites; LoRA also edits other positions); CKA(h) confounded as
above; early-layer activations are lower-rank/more shared, which may inflate the early CKA(δ).

## Reproduce
```
uv run python -m scripts.attribution.train_lora_commonsense \
    --config configs/attribution/loreft_commonsense_llama2.yaml
uv run python -m scripts.attribution.eval_commonsense_suite \
    --config configs/attribution/loreft_commonsense_llama2.yaml --interventions none \
    --lora results/attribution/loreft_commonsense/lora_commonsense --limit 200 --tag lora_lim200
uv run python -m scripts.attribution.compare_lora_loreft_commonsense \
    --config configs/attribution/loreft_commonsense_llama2.yaml \
    --lora results/attribution/loreft_commonsense/lora_commonsense \
    --loreft results/attribution/loreft_commonsense/loreft_interventions.pt \
    --n-eval 60 --layers 4,8,12,16,20,24,28 --k 8
uv run python -m scripts.attribution.plot_activation_similarity \
    --json results/attribution/loreft_commonsense/lora_vs_loreft_similarity.json
```
JSON: `loreft_commonsense/lora_vs_loreft_similarity.json`. Brief: `tasks/lora_vs_loreft_commonsense.md`.
