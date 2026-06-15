# Does the L20 oracle install a capability, or just transport the answer? (E1 logit-lens + E3 lesion)

**Question.** The lockstep oracle recovers GSM8K by overwriting base's L20 residual with the LoRA's
every decode step (full δ = 0.75, top-512 = 0.45 on the base-fails/LoRA-solves contrast set;
`2026-06-10-pca-band-complement.md`). A successful patch cannot tell apart:
- **H_compute** — base's layers 21–31 do the genuine downstream reasoning once given a corrected L20
  state; the capability is latent in base and L20 was the missing input.
- **H_communicate** — the LoRA's L20 already encodes the answer; base's 21–31 only transcribe it.

L20-of-32 is suspicious: patching deeper is *more* communication by construction. Two experiments look
at *where the answer appears* (E1) and *whether base's downstream is load-bearing* (E3).

**Apparatus.** Same lockstep oracle and the base-fails/LoRA-solves contrast set as the PCA/DAS work
(`scripts/attribution/lockstep_patch_gsm8k.py` builds/caches it; here scanned test[0:200] → 113
base-fail/LoRA-solve, first 20 used — byte-identical to the prior 20). Positive control: all-layers
lockstep recovers 0.938 ≈ LoRA. New code: `src/probes/attribution/logit_lens.py` (`LogitLens`),
`layer_ablation.py` (`IdentityAblationHook`); drivers `logit_lens_patch_gsm8k.py`,
`downstream_lesion_gsm8k.py`; tests `tests/test_logit_lens.py` (4), `tests/test_layer_ablation.py` (5).

## E3 — downstream-lesion necessity

Apply the full-δ L20 patch, then identity-ablate base layers 21→31 cumulatively from the top (a
decoder block made identity returns its input), measuring recovery vs #layers ablated. Control: the
LoRA-natural model on the same problems under the identical ablation (base solves ≈nothing, so a
base-solvable control is empty; LoRA-natural is the purest "does the model that *has* the capability
still need these layers?").

| k | ablated layers | recovery_patch | recovery_lora (control) |
|--:|---|--:|--:|
| 0 | — | **0.750** | 1.000 |
| 1 | [31] | 0.600 | 0.650 |
| 2 | [30,31] | **0.000** | 0.100 |
| 3 | [29,30,31] | 0.000 | 0.050 |
| 4–11 | …[21..31] | 0.000 | 0.000 |

k=0 reproduces the prior full-δ oracle exactly (0.750), so the wiring is sound. Figure:
`results/figures/downstream_lesion_L20.png`.

**Reading — H_compute; H_communicate refuted.** The two curves collapse together: identity-ablating
just **two** downstream layers (30,31) craters not only the patched base (0.75→0.00) but the
**natively-capable LoRA** (1.00→0.10). If L20 already carried the answer, the LoRA would not need
30–31 to express it — it does. So L20 is a genuine *intermediate*, not a finished answer awaiting
readout, and base's 21–31 do the same load-bearing computation the LoRA does (they collapse in
lockstep). The matched control also rules out "ablation merely breaks base's fluency" — it breaks the
LoRA's intact capability identically.

## E1 — where the answer token crystallizes

Three closed-loop trajectories (patch / lora / base); at each decode step, logit-lens the residual at
{20,22,…,31} for the emitted token, headline restricted to answer-bearing tokens (decoded token
contains a digit). Caveat: logit-lens top-1 at L20 is *sufficient but not necessary* for communication
(base's 21–31 are a nonlinear decoder), so this is a conservative lower bound; E3 is the causal
complement.

Fraction of answer-bearing tokens already logit-lens top-1, per layer (Figure
`results/figures/logit_lens_patch_L20.png`):

| layer | patch (15/20 solved) | lora (20/20) | base (0/20) |
|--:|--:|--:|--:|
| L20 | 66% (464/698) | 63% (381/603) | 70% (390/554) |
| L24 | 72% | 71% | 76% |
| L28 | 95% | 94% | 92% |
| L31 | 100% | 100% | 100% |

**Reading — the emitted-token metric is confounded; treat E1 as a negative/methodological result.**
The crystallization profile is *identical across all three modes, including base, which solves 0/20*.
Base's wrong answer tokens become top-1 across its stack exactly as the LoRA's right tokens do, so the
metric measures **autoregressive self-consistency** (each rollout progressively commits to its own next
token), not correctness or the presence of the answer. Lensing the emitted token therefore cannot
discriminate compute-vs-communicate — the discriminating fact (base wrong, LoRA right) is invisible to
it. This confirms the documented caveat in its strong form and rules out the lazy "where does the
emitted token crystallize" approach. The real format-vs-compute test is the **gold-token** version:
teacher-force base on the correct CoT and lens the *gold* answer token (does base, given the working,
make the right answer decodable?). Not yet run.

## E1b — gold-token, teacher-forced: can base finish given the working?

The fix for E1's confound: teacher-force base on the LoRA's correct CoT (context held to the correct
chain) and lens the *gold* next token, classified by whether it is a genuinely **computed** result (a
digit in the space/digit run after ``=``) vs a **copied** digit (problem number / the ``The answer is:
N`` restatement). `scripts/attribution/gold_token_lens_gsm8k.py`; n=20 problems (LoRA-TF sanity 0.997).

| class | n | TF-acc | final rank | lens rank L20→L31 |
|---|--:|--:|--:|---|
| all | 2784 | 0.835 | 0 | 1 1 0 0 0 0 0 |
| digit | 598 | 0.906 | 0 | 0 0 0 0 0 0 0 |
| **computed (result of `=`)** | 95 | **0.968** | 0 | **18 → 7 → 0** (L20/22/24) |
| copied digit (not computed) | 503 | 0.895 | 0 | 0 0 0 0 0 0 0 |

**Reading — base can compute the steps; its deficit is trajectory generation, not arithmetic.**
Given the correct working, base predicts genuine computed-result tokens at **96.8%** — *higher* than
copied digits (89.5%) or overall (83.5%). So base's 0/20 free-generation failure is **not** a per-step
compute deficit; when on the correct rails base executes the arithmetic almost perfectly. The lens
column shows these are genuinely computed, not looked up: copied digits are rank-0 already at L20,
while computed results crystallize with depth (rank 18→7→0 over L20–24) — the signature of real
layer-wise computation, and unconfounded because the context is fixed to the correct chain. The
deficit is **multi-step trajectory control**: base can execute each step but cannot lay down and
maintain a correct chain autonomously. Caveat: teacher-forcing on the correct prefix isolates
*execution* from *planning* — this shows base executes given a correct prefix, not that base could
plan the chain; n=95 computed tokens.

## Verdict
**The L20 oracle installs a distributed intermediate, not a transported answer (H_compute); and base's
procedure deficit is trajectory control, not per-step computation.** E3 is
decisive and causal: the answer is not present at L20 — even the natively-capable LoRA needs layers
21–31 (ablating 30–31 alone: 1.00→0.10), and base+patch tracks that collapse, so base's downstream
performs the same load-bearing computation. E1's emitted-token logit-lens is confounded by rollout
self-consistency (base ≈ LoRA ≈ patch) and does not speak to the question; the gold-token teacher-forced
lens is the outstanding test of whether base's *own* stack could express the answer given the working.
Consistent with the register-vs-procedure thesis and the steering nulls: a procedure edit is genuine
distributed computation, recoverable only by transplanting the full per-step trajectory state, not a
low-rank/pointwise function of the recipient activation.

## Reproduce
```
uv run python -m scripts.attribution.lockstep_patch_gsm8k --config configs/attribution/metamath_llama2_gsm8k.yaml --mode control --n-eval 200
uv run python -m scripts.attribution.downstream_lesion_gsm8k --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
uv run python -m scripts.attribution.logit_lens_patch_gsm8k --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
uv run python -m scripts.attribution.gold_token_lens_gsm8k --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
uv run python -m scripts.attribution.plot_downstream_lesion --json results/attribution/downstream_lesion_L20.json
uv run python -m scripts.attribution.plot_logit_lens --json results/attribution/logit_lens_patch_L20.json
```
JSON: `downstream_lesion_L20.json`, `logit_lens_patch_L20.json`, `gold_token_lens_L20.json`. Brief:
`tasks/compute_vs_communicate_L20.md`.
