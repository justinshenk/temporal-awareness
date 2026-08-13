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
