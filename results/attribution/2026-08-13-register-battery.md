# The register battery — measuring the other half of the boundary (S2)

**Date:** 2026-08-13 · **Brief:** `tasks/s2_execution_brief.md` · **Spec:**
`docs/superpowers/specs/2026-08-07-workshop-papers-design.md` §3 · **Status:** in progress

## Why this exists

The register-vs-procedure claim is two-sided — a *register* is low-rank, on-manifold and roughly
pointwise; a *procedure* is high-rank, off-manifold and time-dense — but **only the procedure side
had ever been measured**. GSM8K and MuSiQue each have an oracle layer sweep, a temporal-density
knee, a PCA-band cliff and a five-rung null ladder. No register task had any of them: the register
half rested on the single observation that a ridge map installs refusal tone. At a venue whose
stated topics are measurement validity and falsifiability, that is the first objection, not a
nitpick.

S2 points the **unmodified** procedure drivers at a register donor and reports what comes out.

## Apparatus

| | |
|---|---|
| donor | LoRA r32/α64, dropout .05, {q,k,v,up,down}\_proj, lr 3e-4, 3 epochs, 20k subset of commonsense-170k, seed 42, final CE 0.0074 |
| base | `NousResearch/Llama-2-7b-hf` (the ungated mirror used by every prior strand) |
| eval | ARC-Challenge test, n=500 scan, `max_new=32` |
| prompt | pyreft commonsense template, literally `"%s\n"` — no alpaca wrapper |
| target | `"the correct answer is X"` (~7 tokens) |
| scoring | the word after `"the correct answer is"`, exact match against gold |

The donor is the *same recipe and same 20k subset* as the LoReFT arm of
`2026-06-14-lora-vs-loreft-commonsense.md`, so that comparison stays addressable.

**Why ARC-Challenge and not boolq.** Recovery is read as accuracy on a base-fails/donor-solves
contrast set. On a **binary** task (boolq true/false, piqa solution1/solution2) an intervention that
merely garbles decoding still scores ~50% by coin flip, so a partial recovery cannot be
distinguished from a destroyed one. ARC-Challenge is 4-way. Measured floors on the n=500 scan:
**chance 0.25**, and — more demanding — **majority-class 0.288**, since the gold spread is
answer2 144 / answer3 137 / answer1 117 / answer4 102 and a degenerate always-`answer2` policy would
be perfectly format-compliant. Conditional accuracy is read against **0.288**.

**What this task is, structurally.** The supervised target is ~7 tokens and contains **no
intermediate work**. Contrast MuSiQue, whose target is a chain in which hop 1's answer is written
down and then *consumed* by hop 2 (`#1 >> spouse`). Nothing in the commonsense target refers to
anything the model produced earlier. That absent dependency chain is the whole distinction under
test. It also has two methodological consequences:

1. **The temporal-density axis is degenerate here and was not run.** On a ~7-step trajectory
   `periodic:2` and `periodic:4` are nearly the same intervention, so a density number would look
   like a result while measuring nothing. Per the spec that axis belongs to **refusal**, whose
   generations are long.
2. **`base = 0.000` does not mean the same thing as MuSiQue's `base = 0.000`.** Llama-2 base will
   not emit `"the correct answer is X"` unprompted, so its zero is substantially *format
   non-compliance*, not incapacity — it plainly knows some of these answers. The donor's δ is
   therefore partly a **format register**. S2c below measures that split rather than assuming it.

## Results

### Gap gate + AC1 — PASSED

`.run_logs/s2_gate.log`, ARC-Challenge scan n=500, `max_new=32`:

| | |
|---|---|
| base accuracy | **0.000** |
| donor accuracy | **0.676** (the spec predicted 0.68) |
| contrast problems | **338** base-fail/donor-solve (floor was 80) |
| AC1 all-layers lockstep | **PASS** — reproduces the donor per-problem, 3/3 |

Cached to `commonsense_contrast_set.json`. **Every later commonsense run must pass `--n-eval 500`**:
the cache stores indices into the scan, so the driver's default of 60 would silently misindex all
338 — the trap MuSiQue's 317 set earlier in this work.

### Oracle layer sweep — the register separates from both procedures

`lockstep_commonsense_single.json`, n-contrast 100, `max_new=32`:

| layer | commonsense | MuSiQue | GSM8K |
|---|--:|--:|--:|
| 0 / 4 / 8 | 0.000 | 0.000 | — |
| 12 | 0.050 | 0.020 | — |
| 16 | **0.830** | 0.020 | — |
| 20 | **0.990** | 0.760 | 0.750 |
| 24 | 0.990 | 0.780 | — |
| 28 | 1.000 † | 0.890 † | — |
| 31 | 1.000 † | 1.000 † | — |

† Degenerate tail, excluded from `L*` by the same rule both procedures used: the hook overwrites the
*output* of `model.model.layers[L]`, so L31 hands `lm_head` the donor's final hidden state verbatim
(the all-layers control in disguise), and L28 sits close enough to readout to inherit most of that.

**`L* = 20`** — the same layer as both procedures, by the same earliest-plateau rule. But the *curve*
is not theirs. Onset is earlier and far sharper (0.830 at L16, where MuSiQue is still at 0.020), and
the plateau is essentially total (0.990 against 0.75 / 0.76).

This is the **first quantitative register-vs-procedure separation on the oracle axis itself**. Every
prior comparison between the two sides ran through the pointwise ladder; the oracle was the positive
control the two sides were assumed to share. Here the control itself distinguishes them.

> **Caveat pending the floors.** A 4-way answer space means a large perturbation that merely pushes
> the model somewhere fluent could score well by accident, which is not a risk on GSM8K (a garbled
> injection cannot emit the right integer). The 0.990 is not interpretable until the floors below
> land. This section will be finalized against them.

### Floors at L20 — IN FLIGHT

Two controls, both keeping δ's magnitude and destroying only its content
(`lockstep_oracle.control_injection`, seeded):

- **`mean_delta`** — every position gets the trajectory-average shift: the best fixed vector **at
  the donor's own scale**, chosen per problem with oracle knowledge of its trajectory.
- **`shuffle_positions`** — the true per-token shifts applied in permuted order. Same multiset, same
  norms; only the alignment between shift and token is destroyed.

| intervention at L20 | recovery (n=100) |
|---|--:|
| true oracle, per-token δ | **0.990** |
| `mean_delta` | **0.000** |
| `shuffle_positions` | **0.000** |

**What this settles.** The empirical floor at L20 is **0.000, not 0.25**. A perturbation of the same
magnitude as the true shift does not score at chance on this 4-way task — it scores nothing. So the
0.990 above is *not* an artifact of a small answer space, which was the one outcome that would have
invalidated the oracle result. The register's oracle is real, and it needs the **per-token** shift:
neither averaging it nor misaligning it survives.

**What this does NOT settle, and an overclaim corrected.** An earlier draft of this section said
`mean_delta` "upper-bounds the entire fixed-vector class". That is wrong: the class includes
*scaled* vectors α·δ̄, and this run applies the average at full magnitude (α=1.0) at **every
position, prompt tokens included**. Landing *below chance* is the signature of an off-manifold,
destructive injection rather than of a merely uninformative one — the same α-resonance this work
already measured on MuSiQue's ridge leak (≈0.26 at α=1.0, ≈0 at α=1.5). The supported claim is
therefore: **the best fixed vector at the donor's own scale fails completely**, not that no fixed
vector can install a register.

Deciding which of those it is needs one cheap measurement, and it is exactly what the
`commonsense_format` registry entry exists for: re-score the *same* floor generations asking only
whether the response format was adopted. Format ≈ 0 ⇒ the injection destroyed generation and the
controls bound the floor but do not test the pointwise hypothesis; format high with answers at ≈0.25
⇒ the format installed and only answer selection was lost, which *would* be a real test.
**Running now** (`.run_logs/s2_fmt_{mean,shuffle}.log`.)

### PCA band (δ-rank / off-manifold) — pending

Against GSM8K's cliff (top-64 = 55% of energy, 0% recovery).

### S2c — the ridge map on base, and the format/answer split — pending

`collect_cot_residuals` → `fit_ridge_sweep` → `steer_gsm8k`, all `--task commonsense`, then the same
steered generations re-scored under `commonsense_format`, whose `score` asks only whether the
donor's response format was adopted. Greedy decoding is deterministic and both specs share
`problems` and `prompt` by identity, so the pair is an **exact** decomposition of one eval into
*format installation* vs *answer selection* — a split no procedure task can offer, and the
measurement that replaces caveat (2) above.

## Provenance

Every number above names its artifact; see `papers/register_vs_procedure/numbers.md`.
