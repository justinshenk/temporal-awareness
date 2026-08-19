# E5 — does compliance need the system prompt's attention mass?

**Verdict: YES, causally — and the reason it looked otherwise for most of this experiment is that
in-context demonstration masks it completely. With nothing in context to copy, clamping the system
span's attention share from 0.165 to 0.050 collapses instruction compliance from 0.99 to 0.03
(prefix canary; the suffix canary flips on 120 of 120 items) while accuracy moves 0.525 to 0.467,
an interval that includes zero. With a single compliant example in context, compliance is 3/3 at
every clamp level down to 0.021 and nothing moves. With a single *non*-compliant example, it is
0/2 at every level, including unclamped. The model reproduces the last assistant turn's format
exactly — every one of 720 replies was 8 characters in one arm and 1 character in the other — and
the system prompt only governs behaviour when the transcript is silent.**

Run 2026-08-19 · `allenai/OLMo-2-1124-7B-Instruct` · seed 42 · eager attention · share read
**pooled over all 32 layers** · artifacts `e5_system_clamp/`, `e5_neutral/`, `e5_profile` ·
driver `scripts/context_fatigue/run_system_clamp.py` · brief `tasks/e5_system_clamp_brief.md`.

## Why

The paper's instruction-adherence null reports a canary obeyed at ceiling regardless of context
fill. The all-layer attention sweep then showed the substrate for that behaviour thinning sharply:
pooled over 1,024 heads, attention on the system span falls 0.2351 -> 0.0375 between cold start
and 90% fill. Per token the span stays over-attended (enrichment 5.6x -> 4.4x), so the model is
not *deciding* the instruction matters less — but E1c/E1f established that on a fixed-size span it
is **absolute** mass that drives behaviour, and the system prompt is fixed-size.

## The ladder came from this setup, not another one

`--profile` measures the system span's natural share as prior cases accumulate here:

| prior cases | 0 | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| system share | 0.1661 | 0.0808 | 0.0599 | 0.0372 | 0.0279 | 0.0210 |

**Accumulation alone drives the share down 8x in eight turns.** The first ladder imported 0.0375
from the fill sweep, which used a shorter system prompt and different context; natural here was
0.061, so two arms sat *above* natural and were correctly skipped rather than silently clamping
upward. Every level below is one accumulation actually reaches.

## 1. The causal result (neutral context: no prior turn)

120 items, paired, no assistant turn anywhere — so nothing in context exhibits or contradicts the
canaries and the system prompt is the only source of the instruction.

| | natural | clamped | paired delta 95% CI |
|---|---:|---:|---|
| system share | 0.1652 | 0.0500 (-2.57 nats) | |
| prefix canary | 0.992 | **0.025** | **+0.967 [+0.933, +0.992]** |
| suffix canary | 1.000 | **0.000** | **+1.000 [+1.000, +1.000]** |
| canaries obeyed (of 3) | 2.992 | 1.025 | +1.967 [+1.933, +1.992] |
| accuracy | 0.525 | 0.467 | +0.058 [-0.017, **+0.133**] |
| parse rate | 1.000 | 0.967 | |

**Compliance is destroyed and task competence is not.** The accuracy interval includes zero and
parsing stays at 0.967, so the brief's *void* condition — clamp breaking the model rather than the
instruction — does not fire. At -2.57 nats this is well clear of the -4.7 to -6.1 near-ablation
regime E2a excluded. The suffix canary flipped on **every** item; this is not a shifted
distribution, it is a switch.

Note the level that does it: **0.050**, which accumulation passes on its own between two and four
prior cases.

## 2. One contrary example overrides the instruction entirely

The main run gave every item a single prior case, in two arms differing only in whether that
turn's answer exhibited the canaries.

| arm | prior turn | canaries at natural | at 0.021 (-2.1 nats) | reply length |
|---|---|---:|---:|---:|
| `demonstrated` | `◆ B\n⟦OK⟧` | 3.00 / 3 | 3.00 / 3 | 8 chars, always |
| `undemonstrated` | `B` | **1.00 / 3** | 1.00 / 3 | 1 char, always |

720 generations per arm, zero variance, at six clamp levels. Response length has min = max in both
arms: the model reproduces the previous assistant turn's format exactly. (`forbid_diagnosis` is
satisfied trivially by any short answer, so there are two informative canaries, not three.)

Against §1, where compliance with no prior turn is 0.99: **a single bare-letter example drops it to
zero.** The instruction is not weak — it is outcompeted.

## 3. Why the clamp does nothing in either arm

Both arms are pinned. `demonstrated` is at ceiling because the format is available from the
transcript, so removing the system prompt's mass costs nothing. `undemonstrated` is at floor
because the contrary example already won. The clamp was working correctly throughout — targets hit
to three decimals at sane biases — and had no room to move anything. This is the floor/ceiling trap
the adherence report itself warned about ("the canaries are so easy that... a test needs an
instruction with an intermediate baseline violation rate"), reappearing one level up.

**Mid-experiment this was recorded as a falsification of E5's hypothesis. That verdict was wrong**,
and only the neutral condition of §1 exposed it.

## What this means for the paper's adherence null

The null says a canary is obeyed at ceiling as context fills. Its `forced` arm *rewrites history to
always exhibit the canary*, and its `baseline` arm accumulates the model's own compliant outputs.
Both therefore feed the copy route measured in §2. The null is true as stated and does not mean
what it appears to: it shows compliance survives accumulation of **compliant** history, not that
compliance survives accumulation.

Meanwhile the mechanism underneath is genuinely eroding — accumulation drives the share past the
level where §1 shows compliance breaks. **In-context learning holds compliance up while the
substrate goes away**, which is the paper's accuracy thesis in a second domain.

## Two design faults this run exposed

1. **The `undemonstrated` arm is a counter-demonstration, not an absence.** Its prior turn answers
   with a bare letter, which is itself an example — of not complying. Only a context with no
   assistant turn at all isolates the system prompt.
2. **A share measured in one setup does not transfer to another.** Importing 0.0375 put two arms
   above natural. Clamp ladders must be derived from the setup they are applied to (`--profile`).

## Open

- The share is read pooled over all 32 layers here, unlike E1c/E1f/E2a which index at layer 24.
- Whether *unlearnable* accumulated context erodes compliance the way this clamp does is E6; the
  code-filler arm suggests not, with system-prompt enrichment *rising* under accumulation.
- Recovery is untested: clamping the span back up, and restating the policy in the latest user
  turn, are the two interventions queued in `tasks/tomorrow_paper_b.md`.
