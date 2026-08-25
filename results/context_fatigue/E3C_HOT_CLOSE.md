# E3c′ — measured hot-set closure: the residual is not instrument slack

**Verdict: closing the context tokens the final position actually reads most does NOT beat
the verbatim competitor closure — it recovers nothing. The competition penalty is carried by
*what* the closed tokens say (option names, at 0.0076 attention mass), not by *how much* mass
they receive (the hottest content tokens carry 0.087, 11× more, and closing them is a null).
The E3c residual therefore stays open as prefill-borne interference or non-verbatim channels,
and "instrument slack on high-mass tokens" is eliminated. Secondary finding: 0.42 of
context-body final-position mass sits on chat-template glue and turn boundaries, and closing
*that* is catastrophic (−0.18), consistent with E7's template-glue localization.**

Run 2026-08-25 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · driver
`run_competition_sweep.py --close-arms --measured-close 1.0 --store-rows` · artifacts
`results/context_fatigue/e3c_hot_close/` (turns.csv, summary.json, rows/probe_*.npz) ·
brief `tasks/per_token_capture_brief.md` Stage 1.

## Panel

Identical construction to the committed E3c: n = 365 probes with all arms (15 starved,
4 overflow skips, 0 gold leaks), 127.9 closed tokens per arm (size-matched throughout),
mean fill 0.738/0.751. The committed anchors reproduce: near_dup 0.425, random 0.512,
penalty **+0.0877 [+0.0329, +0.1425]** (committed: +0.085 [+0.030, +0.140]).

## Arms

One capture forward per probe (all-layer/head-mean final-position attention row, stored
under `rows/`) ranks context-body tokens (token region `intro_end`→`evid_start`) by received
mass. Budget = 1.0× the verbatim closure's per-probe token count, so every closure arm
closes the same number of tokens.

| arm | closed set | mass closed | n | accuracy | parse |
|---|---|---|---|---|---|
| `near_dup` | — | 0 | 365 | 0.425 | 0.932 |
| `near_dup_comp_close` | verbatim option-name mentions | 0.0076 | 365 | 0.474 | 0.934 |
| `near_dup_rand_close` | size-matched random | ~0 | 365 | 0.406 | 0.901 |
| `near_dup_hot_close` | top-mass tokens, as measured | 0.416 | 365 | 0.236 | 0.852 |
| `near_dup_hot_rand_close` | size-matched random | ~0 | 365 | 0.411 | 0.890 |
| `near_dup_hotc_close` | top-mass tokens, content-restricted | 0.087 | 365 | 0.389 | 0.962 |
| `near_dup_hotc_rand_close` | size-matched random | ~0 | 365 | 0.386 | 0.871 |
| `random` | — | 0 | 365 | 0.512 | 0.940 |

The as-measured hot set barely intersects the verbatim mentions (0.23 tokens of 127.9): the
mass ranking and the effect-carrying spans are nearly disjoint sets. `hotc` restricts
candidates to the context cases' own content (vignettes, questions, demonstrated answer
letters) by excluding inter-turn template regions; its mass, 0.087, is still 11× the
verbatim mentions'.

## Paired gaps (10,000 draws over probes)

| contrast | estimate | 95% CI | |
|---|---|---|---|
| verbatim: comp_close − near_dup | +0.049 | [−0.003, +0.099] | ns |
| verbatim: control − near_dup | −0.019 | [−0.052, +0.011] | ns |
| **verbatim net (comp − rand ctrl)** | **+0.069** | **[+0.016, +0.121]** | SIG |
| verbatim residual: random − comp_close | +0.038 | [−0.011, +0.088] | ns |
| hot: net (hot_close − hot ctrl) | **−0.175** | [−0.230, −0.121] | SIG |
| hot residual: random − hot_close | +0.277 | [+0.216, +0.337] | SIG |
| hotc: rescue (hotc_close − near_dup) | −0.036 | [−0.082, +0.011] | ns |
| hotc: control − near_dup | −0.038 | [−0.074, −0.003] | SIG |
| **hotc net (hotc_close − hotc ctrl)** | **+0.003** | **[−0.049, +0.055]** | ns |
| head-to-head: hot_close − comp_close | −0.238 | [−0.293, −0.181] | SIG |
| head-to-head: hotc_close − comp_close | −0.085 | [−0.140, −0.027] | SIG |

All four key contrasts survive parsed-only re-analysis on probes where both arms parsed
(verbatim net +0.067 [+0.010, +0.125] SIG at n=313; hotc net −0.038 ns at n=314; hot net
−0.159 SIG at n=284; penalty +0.077 SIG at n=325).

The committed verbatim result replicates within-session (net +0.069 vs committed +0.060
[+0.008, +0.112]), so the harness did not drift and the new arms are read against a live
anchor, not a memory.

## Reading

1. **Mass does not rank causal relevance for competition.** The closure that works closes
   0.0076 of attention mass; closures of 0.087 and 0.416 at the same token budget recover
   nothing and −0.18 respectively. Received attention mass, the quantity the whole
   displacement program (E1c/E1f) shows is causal for *evidence*, does not identify which
   competitor-side tokens carry the competition penalty — the penalty follows token
   *content* (the probe's option names).
2. **The residual is not instrument slack of the "hot paraphrase" kind.** The brief's
   Stage-1 hypothesis — that the verbatim closure misses high-mass paraphrases or shared
   symptom phrases — predicts hotc_close ≥ comp_close. Measured: hotc_close is 0.085
   *worse* than comp_close, with a clean size-matched control at its own null. Remaining
   candidates for the ~40% residual: prefill-borne interference (E6′/Stage 3 bears on the
   general prefill-installation question) and non-verbatim channels that are not
   high-mass either.
3. **Template glue is high-mass and load-bearing.** 0.42 of context-body final-position
   mass sits on inter-turn template/turn-boundary tokens. Closing them costs −0.175 net,
   drops parse to 0.852, and the replies switch from the demonstrated bare-letter style to
   verbose prose ("Based on the provided symptoms and history, …"), sometimes truncating
   before any letter — i.e. the glue carries the transcript's task/format structure, not
   competition. This converges with E7's Qwen bisection (role and recency position patches
   carry ~nothing of dd_full = −2.488; glue tokens are excluded from both role subsets) and
   with E6's precedent mechanism. It also explains why `hot_rand_close` (−0.014 ns) is an
   inadequate control for `hot_close` in mass terms — no size-matched random set can carry
   0.42 mass — so the hot arm's net is read as "closing this *set*, whatever it carries,"
   not as a mass-matched contrast.

## Caveats

- Parse rates are below 1.0 in every arm here (0.85–0.96) where the committed Qwen E3c run
  reported 1.000; OLMo's committed E3c parse rate is not recoverable (artifacts gone). The
  parsed-only sensitivity above covers the inference; unparsed replies score as wrong in
  the headline numbers, as everywhere in the program.
- `hotc_rand_close` is mildly harmful on its own (−0.038 [−0.074, −0.003]): randomly
  closing 128 content tokens clips real case text. The hotc net is computed against it.
- The hot-set ranking uses the all-layer/head-mean row; a per-head or per-layer ranking
  could in principle target differently (Stage 2's instrument would support it).
