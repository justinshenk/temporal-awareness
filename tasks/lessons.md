# Lessons

Patterns worth not re-learning. One entry per mistake, with the rule that prevents it.

## Never diagnose a running job from `tail`

**What happened (P4, 2026-08-06):** read `tail` of the GSM8K contrast-rebuild log, saw 15
consecutive `lora_ok=False`, and told the user the run was broken. The full log showed 6 of those
first 15 were `KEEP` — a ~47% keep rate, exactly the original 113/200. Nothing was wrong.

**Rule:** before claiming a run has failed, `grep -c` the success marker over the *whole* log.
A window of a log is a sample, not a summary, and consecutive failures are the expected shape of
any filter with a <100% pass rate.

## Verify behaviour-preservation before spending GPU, not after

**What worked (P4):** the `computed_flags` → `chain_token_roles` refactor touched a function whose
output feeds committed, published numbers. Ran a parity check over 20,000 random token sequences
against the legacy implementation *while both still existed*, on CPU, before any model was loaded.
It passed, so the later GPU regression arm was a confirmation rather than a gamble — and it
reproduced the committed JSON exactly.

**Rule:** when refactoring anything whose output is already committed to a results file, write the
old-vs-new equality check as a scratchpad script and run it before deleting the old code. Do not
carry the parity check as a permanent test (it would break on deletion); permanent tests must
carry explicit expected values instead.

## Pin the semantics you are preserving, especially where they look like a bug

**What happened (P4):** wrote a test asserting a newline closes GSM8K's `=`-result span. That is
`temporal_gate.in_result_span`'s behaviour; `computed_flags` keeps the span open across *all*
whitespace including newlines. Had I "fixed" the code to match my test, the published E1b numbers
would have moved.

**Rule:** when porting logic, the test asserts what the original *does*, not what it should do.
If the two differ, pin the actual behaviour and write the divergence into the docstring with the
reason. Two functions that look like they should agree may deliberately not.

## Cluster the resampling unit to the independent case

**What happened (P4):** the multihop plan-vs-execute verdict rested on a 0.055 gap over 19,970
tokens — but those tokens came from only 317 problems, and tokens within one chain are strongly
dependent. A token-level interval would have been ~3x too narrow and the claim overstated.

**Rule:** when a metric is computed per-token/per-step but the experiment's independent unit is a
problem/case, bootstrap over the *cases* (`src/common/bootstrap_stats.clustered_rate_gap`). State
the cluster count, not the observation count, as the n.

## Pass the object the callee needs, not the one with a similar name

**What happened (P4):** the driver unpacked `pos, gtok, ... = gold_ranks(...)` and then called
`classify(..., gtok)` where `classify` needs the problem's `gold` **dict** (it reads
`gold["decomposition"]`). `gtok` is a tensor of gold *token ids*. Failed 40+ minutes in, after
model load, with `IndexError: too many indices for tensor of dimension 1`.

**Rule:** when a name means two different things in one scope (`gold` the label dict vs `gold` the
token ids), rename the one you do not need to `_gtok` at the unpack site. And for any driver whose
first useful output is minutes away, run one problem end-to-end before launching the full set.

## Read the generations before arguing from the number

**What happened (S2, 2026-08-13):** two floor controls at L20 both scored 0.000, and I built an
argument on it — "a perturbation the size of the true shift yields nothing, so the oracle's 0.990 is
not chance-inflated." Then I decoded the actual text. The control generations were
**character-for-character identical to unpatched base**. The controls were **no-ops**: `mean_delta`
averaged δ over *all* positions, and with ~150 prompt tokens against ~7–32 generated ones the mean
was dominated by near-zero prompt shifts. Worse, the contrast set is *defined* as
base-fails/donor-solves, so **base scores exactly 0.000 on it by construction** — meaning any no-op
scores 0.000 automatically and my "floor" was a tautology about an intervention that never
intervened. Three claims were committed before six lines of decoded text refuted them. In the same
session I had also read 0% format compliance as "generation destroyed", when base is 0% compliant
too — a metric whose floor and whose failure mode are the same number cannot tell them apart.

**Rule:** before an aggregate becomes an argument, **decode a handful of generations and look at
them** — for the intervention, for the unpatched baseline, and for the positive control side by
side. Specifically:

1. **Ask what a no-op would score.** If a do-nothing intervention produces the same number as the
   result you are attributing to a mechanism, the metric cannot support the claim. On a
   base-fails/donor-solves contrast set that number is always 0.000.
2. **Verify an intervention intervened at all.** Diff its output against the unpatched baseline. An
   ablation that changes nothing is a bug in the ablation, not a finding about the model.
3. **Never infer *why* from a scalar.** "Destroyed", "ignored", "degraded" and "unchanged" can all
   read 0.00. Only the text distinguishes them, and it takes one short script.
4. Prefer controls whose *magnitude is matched by construction* (matched-norm random direction at
   the same positions) over ones assumed to be comparable.

The generations also carry free information the metric discards: base here was not incoherent but
fluently re-listing the answer options, which named the failure mode (format non-compliance) that
the accuracy number could only report as a zero.

## Fit the map on the positions you will apply it to

**What happened (S2c, 2026-08-14):** the commonsense ridge arm read **0.000 at every layer** and was
one write-up away from entering the paper as "a register does not transport through a pointwise
map" — a result the local pushback had already flagged as forcing a §1 rewrite. The generations said
otherwise: at α=1.0 the output was `\n\end​​​​…`, degenerate zero-width-space repetition, and at
α=0.5 it was **byte-identical to base**. It was a *destroyed* model, not a null.

The cause was a window mismatch nobody had reason to notice for three strands.
`collect_cot_residuals` fits on `cot_token_slice` — **generated positions only** — while
`LinearPrimalSteerHook` applies the map at **every** position. On GSM8K the chain is ~250 of ~400
positions, so the map mostly saw what it would be applied to. On commonsense the target is ~6 tokens
against a ~97-token prompt, so **~94% of the positions it was applied to lay off its fit
distribution**, and it extrapolated there to about double the correct magnitude: `‖Wa‖/‖a‖ = 0.551`
where the true δ ratio is ~0.3–0.45. Refitting with `--fit-positions all` brought the ratio to
**0.224**, and the map then installed the donor's format cleanly.

**Rule:** before fitting any map, write down the set of positions it will be *applied* at and the
set it is *fit* on, and make them the same set — or justify the gap explicitly. A design inherited
from one task is not neutral on another; the thing that changed here was the prompt-to-target ratio,
which no line of code mentions. Two cheap diagnostics catch it before a sweep runs: compare
`‖Wa‖/‖a‖` against the measured per-token `‖δ‖/‖a‖`, and check whether the same fit set is a
sensible sample of the application set.

## Do not read R² without its constant baseline

**What happened (S2c, 2026-08-14):** the commonsense map's `R²_te = 0.89` was about to be compared
against GSM8K's 0.61 as "the register's shift is far more linearly predictable". But `r2_te` divides
by the **uncentred** `Σ‖δ‖²` (`gram_accumulator.py`), so it credits a map for merely reproducing δ's
constant component — and *how constant δ is* happens to be the exact property the register/procedure
contrast is about. The comparison was confounded by the thing under study. I also mis-estimated the
baseline at ~0.48 by computing it over generated positions when the fit spans all of them; measured,
it is **0.106**.

**Rule:** whenever an R² is used to compare *tasks*, report the best-fixed-vector baseline beside
it. `constant_r2 = ‖Σδ‖²/(n·Σ‖δ‖²)` needs only a first moment, and
`R²_centred = (R² − R²_const)/(1 − R²_const)` is an identity, so no refit is required. And state the
**window** any "the shift is one direction" claim holds on: commonsense δ is near-constant over its
~6 generated positions and strongly conditional over the full sequence, so the same task supports
opposite-sounding sentences depending on which positions were summarized.

## A watcher whose pattern matches its own command line never fires

**What happened (S2, 2026-08-13):** armed `until ! pgrep -f "train_lora_commonsense"; do sleep 30;
done` to wait for a training job. `pgrep -f` matches against full command lines — including the
watcher's own, which contains that exact string. The watcher therefore always found "a match",
never exited, and sent no notification. Two of them span for an hour while the job they were
watching finished *and failed to save*. The failure was found by accident, from an unrelated
`Errno 122` on a small file write.

**Rule:** never wait on a process by grepping for a string that appears in the waiting command.
Capture the PID at launch (`cmd & PID=$!`) and poll `kill -0 "$PID"`. If only a pattern is
available, break the literal with a character class (`pgrep -f "[t]rain_foo"`). And a watcher that
has produced no output for longer than the job's expected runtime is itself suspect — check it,
rather than reading its silence as "still running".

## Check free space before a long job, not after

**What happened (S2, 2026-08-13):** a 55-minute LoRA training run completed all three epochs, then
was truncated mid-`save_pretrained` by a disk quota: `adapter_model.safetensors` stopped at exactly
192 MiB against 224,395,264 B expected, and `adapter_config.json` was never written. `results/` had
been sitting at 53 G, of which 50 G was two Gram-accumulator trees that nothing in the current plan
reads. The GPU time was spent twice for want of a one-second check.

**Rule:** before launching any job that ends in a large write, verify free space against the size of
what it will write (`du -sh` the output's siblings). A round file size — exactly 192 MiB, exactly
2 GB — is the signature of a truncated write, not a coincidence; verify the *expected* byte count
from parameter counts, and confirm every sidecar file (config/tokenizer/index) exists before
treating a save as done. Training completing is not the same as the artifact existing.

## A config's capacity assumption is a box assumption — recheck it when the box changes

**What happened (2026-08-17):** the MuSiQue all-positions collect ran 50 minutes to completion of
its train split, then OOM'd allocating the held-out split's accumulators — on the replacement
32 GB RTX 5090, where every prior collect had run on 80–96 GB cards. The config said so out loud:
`accum_device: cuda  # 64 x 4096^2 f64 ~ 8.6 GB; fits 80GB`. Two splits of float64 Gram matrices
(~26 GB) beside the bf16 models cannot fit in 32 GB, and the write-at-end design meant the
completed train split saved nothing.

**Rule:** when the hardware changes, grep the configs for capacity assumptions (`accum_device`,
`device_map`, dtype comments, "fits NN GB") before the first long run, and budget peak memory for
the *whole* pipeline — the crash came at the second allocation site, not the first. Accumulation
belongs on CPU unless the card demonstrably fits both splits plus the models.

**Same day, same run, the disk half (a lesson already written, then half-followed):** the rerun
completed both splits and died **mid-save at 49 of 65 files** — `results/` hit the same ~53 G
quota that truncated the Aug 13 donor save, with **447 T "free" in `df`** (quotas are invisible to
`df`) and **no traceback** (the traceback write went to a log on the same quota'd volume and
failed too — a silent exit with a truncated file is the quota's signature). My preflight had been
`df` plus a 2 G write test because the FS lacks `fallocate`; the S2d brief's preflight is
full-size *for this reason*. **A disk preflight is a write of the size you will write, on the
filesystem you will write to** — `dd` the full byte count, then delete it. Anything smaller tests
the wrong thing.

## A comparison cell needs an artifact, not a recollection

**What happened (2026-08-06):** the multihop writeup's ridge-divergence claim cited "GSM8K ≈0.05,
≈0 at every layer" as the baseline. Tracing it found **no artifact**: the only committed GSM8K
per-layer ridge steering covers L0/L1/L14/L16/L31 (smoke, n=12/50, all 0.00); L20 and L24 — the
exact layers where multihop leaks — were never probed. The "0.05" appears to have been absorbed
from the PCA-band oracle and lesion-control tables, different experiments entirely. The claim had
already propagated into two results docs and two commit messages.

**Rule:** every cell in a cross-run comparison table names the JSON it came from. Before asserting
"X diverges from Y", open Y's artifact and confirm it was measured *at the same setting* — same
layer, same α, same injection mode. If it wasn't, the cell reads "not measured", not a number. A
number you cannot open a file for is a memory, not a result.

## A 429 from the Hub arrives disguised as a corrupted tokenizer

**What happened (2026-08-19):** the E3 preflight died on
`AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")` with
`OSError: Unable to load vocabulary from file. Please check that the provided vocabulary is
accessible and not corrupted.` The cache was complete and untouched — `tokenizer.json`, 7.1 MB,
same blob the E1/E2 runs had used the day before. The real cause was five retried
**HTTP 429s on `merges.txt`**: `tokenizer_config.json` declares `tokenizer_class: GPT2Tokenizer`,
so transformers goes looking for the slow tokenizer's `vocab.json`/`merges.txt`, gets rate-limited
rather than 404'd, and reports the *outcome* (no vocabulary) instead of the *cause* (throttled).
`use_fast=True` does not help; it takes the same path. `HF_HUB_OFFLINE=1` loads the cached
`tokenizer.json` instantly.

**Rule:** when a load fails on a cache that worked yesterday, read the *whole* stderr for HTTP
status lines before believing the exception's own diagnosis — the message names what broke, not
why. On a warm box, run drivers under `HF_HUB_OFFLINE=1` by default: it is faster, it is
reproducible, and it converts "the Hub is throttling us" from a corrupted-data red herring into an
honest cache miss.

## Report absolute changes first; a ratio on a small base is not a finding

**Correction (2026-08-19, Paper B head analysis).** I ranked attention heads by *fractional*
drain and led with head 28 at "99.3%". Head 28 goes 0.0198 → 0.0001: the largest percentage and
nearly the smallest absolute movement in the set. Head 19 loses 0.0747 of attention mass — 40×
more — and my ranking put it seventh. The percentage was measuring the size of the denominator,
not the size of the effect.

**Rule.** Lead with the absolute change in the unit the quantity is measured in. A ratio,
percentage, or "N×" may follow it, never replace it, and never set an ordering on its own. Before
quoting a ratio, state both numbers it came from and ask whether the difference would matter to
anyone at that scale.

**Where this already bit:** "competition heads move 9.6× the net change" is 0.0026 against 0.0003,
both small next to a 0.041 evidence share; "50× larger share change needed" is the same
arithmetic. State the shares, then the ratio.

## Graders fail on real outputs, not imagined ones — sample generations before trusting any metric

**Correction (2026-08-20, E6 mmlu arm).** Fourth grader bug of one experiment, all the same
shape: a checker written against an imagined output format scored a real, common reply shape as a
violation. Run 1: `ANSWER: Epiglottitis` scored as no-answer and wrong. Run 2: bulleted
SUPPORTING lists scored as zero symptoms; 128-token truncation cut the component under test.
Today: `"B\n<prose>"` — a correct answer in the exact style the filler demonstrates — fell
through every fallback and was scored unparsed *and* wrong, fabricating a below-chance accuracy
collapse (0.075) at exactly the depth where the finding lived. The corrected number is 0.500.

**Rule.** Before reading any per-condition metric, pull 3-5 raw generations *from the condition
where the effect appears* and hand-check them against the grader. A metric that moves exactly
where the hypothesis predicts is the one most likely to be a grader artifact, because the
treatment changes the output format, and the grader was written before seeing treated outputs.
When a run's metric collapses, the first suspect is the parser, not the model — especially when
parse rate moves with the metric (parse 0.35 at the "collapsed" depth was the tell).

## A control must match the arm it controls for — dose by dose

**Correction (2026-08-20, mode-vector steering).** Round 2 ran the real vector at α=1 and α=3
but the random control only at α=3, where any vector that large breaks generation outright. The
α=1 result — the interesting one — was left uncontrolled, and a third run existed only to add
the missing arm. Round 1 had the complementary failure: the intervention itself was so strong
that real and random were indistinguishable, which is a statement about the instrument, not the
hypothesis.

**Rule.** Every treatment arm gets a control matched on every nuisance parameter — norm, dose,
application schedule, layer — at that arm's own setting. If controls are expensive, drop a
treatment dose rather than sharing one control across doses. And when real and random produce
the same large effect, the finding is "instrument too blunt", never "effect confirmed".

## Thousands of tiny fits want one thread

**Correction (2026-08-20, probe analysis).** Default multi-thread BLAS on ~20k small
scaler→PCA→LDA fits burned 929 CPU-minutes without finishing; the same job single-threaded
(`OMP_NUM_THREADS=1` etc.) completed in ~15 wall-minutes. Thread spin-up and synchronization
dominate when each matrix op is milliseconds; CPU% looks impressively busy while doing almost
no work — full cores are not progress.

**Rule.** For many small fits, set the BLAS thread env vars to 1 (and parallelize across fits
if needed). Reserve multi-thread BLAS for few large decompositions. Also: run long analyses
unbuffered (`python -u`) with output tee'd to a file, so progress is observable and a kill
doesn't destroy the evidence of where it was.

## A null on one span's mass does not license "not attention-mediated"

**Correction (2026-08-21, E3c — user-caught).** E3 measured the *evidence's* attention share
unchanged under competition and the paper concluded competition "does not act through attention
mass." The user pointed out the untested route: the filler's arm-constant mass lands on
instances of the probe's own answer candidates in near_dup. Closing those spans recovered 59%
of the penalty with a clean size-matched control — competition is attention-mediated,
competitor-side, and the title claim built on the null was wrong.

**Rule.** A mediation null is a claim about one span, not about attention. Before writing "not
attention-mediated," enumerate the spans whose reading could carry the effect and close the
treatment-loaded ones with a size-matched control. The aggregate that stays constant is exactly
the one that can hide a redistribution of *what* is being read. Corollary: the closure clamp is
cheap — when a mechanism claim rests on a negative, run the closure before the claim ships,
not after a reader finds the hole.

## Report numbers are read from artifacts, never asserted from intent

**What happened (E7, 2026-08-24):** the Stage-1 report claimed "0 probes skipped in either
arm" and "fill ≈0.6" for the code arm. The OLMo box's verification pass found 6 overflow
skips (n=34, not 40) and fill 0.943 — the claims were written from the design's intent, not
from `turns.csv`. The ΔΔ estimands were all exact; only the unread metadata was wrong.

**Rule:** every n, fill, skip count, and parse rate in a report is computed from the
artifact in the same session that writes the report — `groupby` the turns file and paste,
never transcribe from memory or the plan. If a number was not computed, it does not go in.

## A cross-family difference is a difference-in-differences, never a CI-vs-CI eyeball

**What happened (Q4, 2026-08-24):** the Qwen E3 report claimed Qwen's contrast CI
"excludes OLMo's entire interval." The intervals overlap on [+0.030, +0.074]; the DiD is
+0.055 [−0.015, +0.125], not significant. The verification pass downgraded the claim: the
penalty is *not detected* on Qwen, and only the attention inversion (CIs disjoint by an
order of magnitude) is an established family difference.

**Rule:** any "differs across models/runs" claim gets its own bootstrap on the difference
of the two effects. Comparing one CI against another CI's endpoints — or against the other
run's point estimate — is not a test. The repo's E2b rule applies: overlapping intervals
are non-robustness, not refutation.
