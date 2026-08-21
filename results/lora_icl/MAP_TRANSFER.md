# Cross-model LoRA capability transmission via linear maps — NULL, maximally attributed

**Verdict: the capability does not survive the linear change of basis, and every control needed
to localize that failure came back positive.** Within-model mean-shift steering works on both
models (donor recovers 71% of its floor→ceiling gap from one vector; recipient 68% from its
own). The 7B→1.5B ridge maps capture 0.72–0.82 of held-out state variance at every layer. Yet
the mapped donor shift transfers zero capability at any layer or dose — because the map sends
the donor's capability direction essentially **orthogonal to the recipient's own**
(cos = −0.19 to +0.18 at L7/14/18/21), while still landing in behaviorally potent subspace
(at 2× dose the mapped vectors collapse generation, parse 0.00–0.09, where norm-matched random
vectors at identical dose are harmless, parse 0.97–1.00). Linear correspondence of *states* is
not linear correspondence of *function*.

Run 2026-08-21 · donor Qwen2.5-7B-Instruct, recipient Qwen2.5-1.5B-Instruct (both 28 layers;
maps 3584→1536, identity layer pairing) · eval panel 100 seeded test-split DDXPlus cases shared
by every arm · map corpus 400 disjoint cases · adapters retrained from the committed configs
(r16 α32, seed 42; 7B at micro-batch 2 / grad-accum 8 / gradient checkpointing — 32 GB box) ·
artifacts `results/lora_icl/map_transfer/` · driver `run_lora_map_transfer.py` (5 phases) ·
brief `tasks/lora_map_transfer_execution.md`.

## Calibration (all arms, same 100 items, same parser)

| | floor | ceiling (own adapter) | shuffled adapter |
|---|---:|---:|---:|
| donor 7B | 0.130 | **0.970** | 0.140 |
| recipient 1.5B | 0.140 | **0.900** | — |

Floors sit below chance because the bare "You are a doctor." prompt yields prose the 6-token
parse window misses; the parser is identical in every arm, so the comparison is internally
consistent. Shuffled-label adapter ≈ floor: task-specificity clean.

## Within-model self-steering (the premise holds at home, twice)

Mean shift δ = mean(resid(+LoRA) − resid(base)) at the answer position, added decode-time at
one layer, no adapter:

| layer | donor self-steer (α=1) | recipient self-steer (α=1 / α=2) |
|---|---:|---:|
| L7 | 0.140 | 0.140 / 0.140 |
| L14 | 0.480 | 0.640 / 0.660 |
| L18 | **0.730** | **0.670** / 0.420 |
| L21 | **0.730** | 0.340 / 0.190 |

One vector at one mid-depth layer carries 66–71% of the adapter's whole effect in *both*
models. (Recipient deep layers overdose at α=2 — the familiar dose sensitivity.)

## The maps are good…

Held-out R² of the per-layer ridge maps (λ chosen per layer on the holdout): 0.72–0.82 across
all 28 layers (L14 0.773, L18 0.745, L21 0.715; profile in `map_r2_profile.json`). A transfer
null cannot be blamed on the map's fit.

## …and the transfer is zero anyway

Recipient steered with M_L·δ_donor (decode-time), full grid L ∈ {7,14,18,21} × α ∈ {1,2}:
every arm at or below the 0.140 floor (best 0.140; L18/L21 at α=2: accuracy 0.00–0.04 with
generation destroyed). Norm-matched random controls: 0.120–0.140, parse 0.87–1.00 —
harmless. Mapped shuffled-adapter shift: equally destructive at deep layers, equally
non-transferring.

**The closing geometry.** cos(M_L·δ_donor, δ_recipient_own) = −0.185 (L7), +0.007 (L14),
−0.115 (L18), +0.176 (L21). The mapped donor direction is orthogonal to the direction that
demonstrably drives the recipient (0.640 at L14). Ridge shrinkage also halves the mapped
norms (e.g. 23.9 vs the recipient's own 31.7 at L14), but at cos ≈ 0 no rescaling could help —
the failure is direction, not dose.

## Reading

The user's hypothesis — fine-tuned chat models share enough activation-space structure to
transmit a LoRA's capability through a linear map — is **refuted in this rung's strongest
form**: the shared structure is real (R² ≈ 0.75 linear correspondence of states, and mapped
vectors are direction-specifically potent, so the map does carry *behaviorally meaningful*
geometry), but the task-functional direction is not part of what corresponds. Two models that
each encode "answer this MCQ with the trained letter" as a steerable mid-depth direction encode
it in directions a state-fitted linear map does not connect.

Contrast that makes it sharp: the same mean-shift machinery that transfers **nothing across
models here** installed a format mode **within** a model in Paper B's E6 (0.000 vs 0.525
against matched control). Direction-based behavior transmission is real; its currency is
model-internal.

## Caveats and next rungs

- Mean shift is a rank-1 summary of a rank-16 adapter; subspace transfer (top-k principal
  directions of the shift set, mapped and added jointly) is untested.
- Identity layer pairing (i→i) across a 4.7× parameter gap; a cross-layer pairing search
  (fit maps for all 28×28 pairs on the cheap corpus, steer at the best-R² pairing) is one
  loop away.
- Maps were fit on task-prompt states only; a broader corpus, or orthogonal-Procrustes maps
  (norm-preserving, no shrinkage), might align function better than ridge.
- One task, one direction of transfer (big→small), same family. The 7B→7B-across-family rung
  (Qwen→Gemma at matched hidden width) would separate "scale gap" from "family gap".

## Void checks

None fired: donor self-steer positive, recipient self-steer positive, map R² high, random
controls harmless at every dose, parse rates intact outside the deliberate overdose arms,
identical eval panel everywhere (`eval_gold.json`).
