# DDXPlus function vector (FV) — extraction and FV-vs-LoRA comparison

`google/gemma-2-9b-it` | 4-shot ICL | 24 instances | causal mediation over 42x16 heads | top-10 FV heads | restricted A-E readout.

## Task signal (sanity)

- Clean 4-shot ICL accuracy: **0.71**; corrupted (shuffled-label) accuracy: **0.71**; zero-shot (no demos): **0.67**.
- Label-dependence (clean − corrupted) = **+0.00**: ≈0 ⇒ the demo labels are inert (no in-context task signal to recover — see Reading).

## Top FV heads (by average indirect effect)

| rank | layer | head | AIE |
|-----:|------:|-----:|----:|
| 1 | 28 | 8 | +0.0072 |
| 2 | 23 | 13 | +0.0057 |
| 3 | 22 | 13 | +0.0049 |
| 4 | 24 | 4 | +0.0048 |
| 5 | 25 | 7 | +0.0027 |
| 6 | 23 | 12 | +0.0024 |
| 7 | 26 | 5 | +0.0022 |
| 8 | 39 | 0 | +0.0019 |
| 9 | 30 | 8 | +0.0017 |
| 10 | 41 | 11 | +0.0017 |

## FV validation — zero-shot accuracy with the FV added

| insert layer | zero-shot acc + FV |
|-------------:|-------------------:|
| 10 | 0.67 |
| 14 | 0.67 |
| 20 | 0.67 |

Baseline zero-shot acc (no FV): **0.67**; best insert layer L10 -> **0.67**.

## FV vs LoRA vs ICL task vector (cosine @ L10)

| pair | cosine |
|------|-------:|
| FV · LoRA-shift | +0.001 |
| FV · ICL-task-vector | +0.018 |
| LoRA-shift · ICL-task-vector | +0.025 |

## Reading

- **NULL — and a code-independent sanity check says why.** Clean 4-shot accuracy (0.71) ≈ corrupted shuffled-label accuracy (0.71), both ≈ zero-shot (0.67). Shuffling the demonstration labels does not hurt ⇒ the model ignores the in-context labels and answers DDXPlus from its medical knowledge. There is **no in-context task signal** for causal mediation to recover, so the AIE values are ~0 and the FV is noise.
- **The FV is inert, as expected:** adding it to a zero-shot prompt does not move accuracy (0.67→0.67 at every insert layer).
- **The FV·LoRA cosine (+0.001) is UNINFORMATIVE, not evidence of orthogonality.** You cannot conclude the in-context and in-weights task vectors point different ways when the FV extraction had no signal to extract. (The LoRA·ICL cosine here, +0.025, is measured at L10 — an early layer where the subspace study already showed convergence ≈ 0; the 0.81 convergence was at L35.)
- **What this qualifies:** DDXPlus is a *knowledge* task this model already solves near-zero-shot, **not** an *in-context-learning* task. Our LoRA-vs-ICL subspace convergence is real activation geometry, but it is **not** a Todd-style causal function vector — there is no extractable in-context task algorithm here; the convergence reflects context/format/calibration adaptation. To test FV extraction (and "same task vector, two routes"), use a task where ICL carries the signal (zero-shot fails, demos define the mapping — e.g. antonyms or a relabeled/symbolic task).
- **Scope:** one model/task, 24 instances, top-10 heads, restricted-letter readout; FV insertion at the last token only.
