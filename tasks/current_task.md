# Current task — Multi-hop generality test of the register-vs-procedure thesis (EXECUTION)

**Decision locked (was the open §0 in `multihop_generality.md`):** dataset = **MuSiQue**
(`dgslibisey/MuSiQue` on HF; schema verified: `paragraphs[{title,paragraph_text,is_supporting}]`,
`question_decomposition[{question,answer}]` with `#k` refs, `answer`, `answer_aliases`), framing =
**open-book** (gold *supporting* passages placed in the instruction; LoRA learns multi-hop
*composition* over given facts — the analogue of GSM8K's in-problem numbers — not parametric recall).
This diverges from the brief's earlier "closed-book 2WikiMultiHopQA" default; the code
(`multihop_prompts.py`) already committed to MuSiQue open-book and the user confirmed it. Brief
updated to match.

**Environment:** one RTX PRO 6000 Blackwell (96 GB), CUDA on, datasets 4.4.2 / torch 2.9 / peft 0.19
/ transformers 5.3. Base `NousResearch/Llama-2-7b-hf` (ungated). Heavy phases run here on GPU.

## Goal
Re-run the GSM8K *procedure* apparatus on a second, non-arithmetic multi-step procedure (MuSiQue) and
adjudicate **H_general** (procedure thesis generalizes: full-δ oracle recovers; pointwise-map ladder
≈0; temporal density sharp) vs **H_arith** (it was arithmetic-specific). Either is publishable.

## Reuse map (verified by reading the code)
- `attribution_common.py` centralizes the GSM8K seam: `prompt_token_ids → metamath_prompt(question)`,
  `gsm8k_problems` (HF "gsm8k"), `gsm8k_accuracy → numeric_match`. Parameterize via a **task registry**
  (default GSM8K, unchanged) — add `--task {gsm8k,multihop}` to the 5 drivers.
- Trainer helpers reused from `train_loreft_commonsense.py`: `load_frozen_base`, `collate_left_padded`,
  `linear_warmup_decay`. `encode_examples` is commonsense-specific (calls `commonsense_data.format_*`)
  → write a multihop encode. Prompt↔target join = prompt + `"\n"` + solution (per
  `metamath_fewshot_prompt`). LoRA recipe (LLM-Adapters): r32/α64/dropout .05, {q,k,v,up,down}_proj.
- Seams already done + tested (9 CPU tests): `multihop_prompts.py` (prompt, `resolve_decomposition`,
  `format_multihop_solution`, `extract_pred_answer`, `normalize_answer`, `answer_match`,
  `answer_span_gate`). KEEP `multihop_prompt(q, passages)` signature (test-pinned); refactor it to
  delegate to a single-`{instruction}` template so a one-arg driver prompt exists too.

## Phases (each gates the next; recovery = (acc−base)/(lora−base))
- **P0 — LoRA + gap (go/no-go).** New: `src/probes/attribution/multihop_data.py`,
  `configs/attribution/multihop_llama2.yaml`, `scripts/attribution/train_lora_multihop.py`. Train r32
  LoRA on ~20k answerable MuSiQue (open-book supporting passages). GATE: donor exact-match ≫ base
  closed-book on a ≤500 scan → need ≥~80 base-fail/donor-solve contrast problems. If gap too small,
  STOP and report (itself a finding).
- **P1 — oracle + L\*.** Task-parameterize drivers. all-layers oracle positive control (≈donor);
  single-layer sweep {0,4,…,28,31} → `L*`.
- **P2 — pointwise ladder @L\*.** ridge / MLP / on-policy DAgger recovery vs full-δ oracle.
- **P3 — temporal density @L\*.** periodic(k) knee + structural `answer_only`/`reasoning_only` split.
- **P4 (optional) — plan-vs-execute** (E1b analogue), only if P1–3 land H_general.

## Acceptance / outputs
Per-axis verdict vs the GSM8K numbers, honest n=2-procedures caveat. Writeup
`results/attribution/2026-06-16-multihop-generality.md` + JSONs/figures; update
`results/activation_weight_investigation.md` if the thesis generalizes. CPU tests for every new seam
pass with no network. All seeded (42); contrast set cached `multihop_contrast_set.json`.

## Progress
- [x] Recon: GPU, dataset (MuSiQue verified), trainer/driver contract mapped, decisions locked.
- [ ] P0 code (data module, config, trainer) + CPU test + data-pipeline smoke.
- [ ] P0 run: train LoRA, base-vs-donor gap, build contrast set.
- [ ] P1 driver task-registry refactor + oracle + L*.
- [ ] P2 ladder. [ ] P3 temporal. [ ] writeup + commit.
