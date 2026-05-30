# Testing the RLHF Story Behind Context Fatigue

The context-fatigue writeup (`scripts/context_fatigue/WRITEUP.md`, §1.1 and §7.5)
makes a strong **causal** claim:

> *Entropy collapse and fatigue-amplified sycophancy are instruction-tuning
> artifacts. RLHF teaches confidence and conciseness; accumulated context
> amplifies this training signal.*

Its entire evidentiary basis is **one base-vs-IT pair (Gemma 2 9B)** with thin
samples, and it conflates SFT with preference optimization ("RLHF/DPO"). These
experiments stress-test that claim:

1. **OLMo-2 post-training dose-response** — replays an identical fatigue probe
   across `base → SFT → DPO → Instruct(RLVR)` of one model, separating SFT from
   preference optimization and turning a correlational pair into a graded curve.
2. **F90871 causal steering** — intervenes on the boundary-detector feature the
   writeup calls "a key regulator," testing whether *restoring* it rescues a
   fatigued model.
3. **Base-model sycophancy at scale** — enlarges the n≈15 base control.

Scripts: `scripts/context_fatigue/run_olmo_gradient.py`,
`run_f90871_steering.py`. Shared helpers in `_cf_common.py`.

---

## Experiment 1 — OLMo-2 7B post-training dose-response  ✅ HEADLINE RESULT

The same DDXPlus accumulation probe (identical cases and option orders replayed
to every checkpoint, so differences are attributable only to post-training) run
across the four public OLMo-2-1124-7B checkpoints. `entropy_early` = mean
next-token entropy over the first third of accumulated cases (baseline
confidence); `entropy_late` = last third.

| stage | early entropy | late entropy | within-ctx ratio | syc flip (fat / clean) |
|-------|--------------:|-------------:|-----------------:|------------------------|
| base       | **1.058** | 0.645 | 1.64× | 0% / 100%* |
| + SFT      | **0.582** | 0.462 | 1.26× | 30% / 25%  (+5pp) |
| + DPO      | **0.327** | 0.404 | 0.81× | 60% / 50%  (+10pp) |
| + Instruct (RLVR) | **0.201** | 0.426 | 0.47× | 10% / 42%  (−32pp)* |

\* base/instruct sycophancy denominators are tiny (see caveats). Data:
`results/olmo_gradient/{gradient.csv,ddxplus_turns.csv,sycophancy.csv}`.

**Finding 1 (robust): baseline confidence collapses monotonically with each
post-training stage.** Early-context entropy drops 1.06 → 0.58 → 0.33 → 0.20 —
a clean ~5× contraction, monotone across all four stages. Crucially, **the
largest single drop is base → SFT (1.06 → 0.58)**, with DPO and RLVR each adding
a further halving. So the confidence shift is *not* purely an RLHF/preference
phenomenon as the writeup implies — **supervised fine-tuning on assistant
transcripts already installs most of it**, and preference optimization
compounds it. This directly addresses the writeup's SFT-vs-RLHF conflation.

**Finding 2 (refinement): the "entropy collapses *as context fills*" framing is
partly a base-model ICL effect, confounded by a floor.** The within-context
ratio (early/late) *falls* across stages: base collapses with context (1.64×,
classic in-context learning sharpening a still-uncertain model), but the aligned
models start already near the confidence floor (early entropy 0.20–0.33) and
have nowhere left to collapse — entropy even ticks up slightly later. Within
OLMo's 4096-token window the RLHF effect reads as a **downward level shift in
baseline entropy, not a steeper collapse slope.** (The writeup's steeper IT
collapse was measured over 32k-token Qwen/Llama contexts and hundreds of cases;
the level-shift and slope effects are not mutually exclusive — they operate at
different context scales.)

**Finding 3 (directional, noisy): sycophancy amplification tracks
post-training, but n is too small here to be conclusive.** SFT→DPO shows the
predicted rise in fatigued-minus-clean flip rate (+5 → +10pp) and the absolute
fatigued flip rate climbs 30% → 60%. But base (can't follow the `ANSWER: X`
format → ~1 baseline-correct case) and Instruct (−32pp) are dominated by
single-digit denominators. Re-run at larger N: `results/olmo_gradient_n50/`.

---

## Experiment 2 — F90871 causal steering  ⚠️ NULL / COMPLICATES THE STORY

The writeup's §5.3 elevates F90871 (a BOS/document-boundary detector in the
`gemma-scope-9b-it-res` layer-20 width-131k SAE) to "a key regulator of whether
the model treats new input as independent," suppressed in *every* experiment.
We test it causally on Gemma 2 9B IT: build a fatigued DDXPlus context (83%
fill), then on held-out cases compare **clean**, **fatigued**,
**fatigued + F90871 clamped up to its clean level**, and **fatigued + a random
control feature clamped** (specificity). Data: `results/f90871_steering/`.

| condition | entropy | DDX acc | sycophancy flip |
|-----------|--------:|--------:|----------------:|
| clean              | 0.293 | 0.50 | 31% |
| fatigued           | 0.442 | 0.45 | 25% |
| fatigued + F90871  | 0.465 | 0.45 | 27% |
| fatigued + random  | 0.524 | 0.50 | —   |

**The intervention does not rescue the fatigued model.** Clamping F90871 back up
moves entropy the *wrong* way (0.442 → 0.465, away from clean's 0.293), leaves
accuracy flat (0.45), and barely moves sycophancy (25% → 27%). A random feature
of similar magnitude perturbs entropy at least as much.

**The suppression premise is also not robust in this setup.** At the *decision
token* on DDXPlus chat cases, F90871 is **−33% (i.e. 33% higher), not
suppressed**, under fatigue — the opposite sign to the writeup's −37%. (A weakly-
fatigued 25%-fill smoke test showed −43% suppression, so the sign is
config-dependent.) F90871 fires hugely at the literal BOS token (~3137) and is
the single top feature at the clean decision token (~44–60), so it is a real,
strong feature — but its decision-token level is not a stable monotone function
of fatigue.

**Interpretation.** As operationalized on DDXPlus, F90871 suppression is a
context-dependent *correlate*, and single-feature restoration is *not*
sufficient to undo fatigue. This tempers the writeup's "key regulator" framing.
Caveat: the writeup measured F90871 in different setups (single-turn sycophancy:
35.6→6.7; NarrativeQA), so this is evidence against a *robust causal* role on
DDXPlus, not a refutation of the original correlational observations. A faithful
causal test in the exact single-turn sycophancy setup (where the largest
suppression was reported) remains future work.

---

## Experiment 3 — Base-model sycophancy at scale  ⏳ RUNNING

Enlarging the n≈15 Gemma 2 9B base control (fatigued vs clean flip rate) to
n=80 test questions, alongside the IT model, via `run_sycophancy_final.py`.
Results: `results/sycophancy_scaleup/`. *(section to be completed)*

---

## Bottom line

- **Strengthened:** the core RLHF claim survives and generalizes beyond the
  single Gemma pair. Baseline confidence collapses monotonically across an
  independent model family's full post-training chain (1.06 → 0.20), and — a new
  result — **SFT, not just preference optimization, installs most of the
  overconfidence.**
- **Refined:** within a fixed context window the effect is a *level shift* in
  baseline entropy; the "steeper collapse slope" is confounded by an ICL effect
  in the base model and a confidence floor in the aligned models.
- **Complicated:** the F90871 *mechanism* does not hold up to a causal test on
  DDXPlus — suppression is sign-unstable and restoration does not rescue.
  The behavioral RLHF story is solid; the specific single-feature mechanistic
  story is not.
