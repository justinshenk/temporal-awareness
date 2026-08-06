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
