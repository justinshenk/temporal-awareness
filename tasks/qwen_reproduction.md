# Qwen reproduction — headline experiments only (user-scoped 2026-08-24)

**Goal:** reproduce the paper's headline results on a second model family,
`Qwen/Qwen2.5-7B-Instruct` (cached). Explicitly out of scope (user): post-training
dose-response (needs a public SFT/DPO chain), F90871 SAE clamp (Gemma-Scope-specific),
WildChat signatures (already run on Qwen-2.5 — free), non-headline appendix analyses
(per-head structure, E1d/E1e/E2a) unless a headline result disagrees and needs them.

**Conventions:** all-layer pooled share readout (`--reference-layer 0..27`; Qwen2.5-7B has 28
layers) — no per-family reference-layer choice, consistent with the OLMo all-layer
re-denomination (2026-08-24). `max_ctx` 4096 (same budget as OLMo, so fill is comparable).
Seed 42, same panel constructions, paired bootstrap over cases. HF_HUB_OFFLINE=1. Preflight
every run; validate every grader against real Qwen generations before trusting a ladder
(this program's five voided runs were all format-drift artifacts).

## Queue (strict order; each ~step gated on the previous)

- [x] **Q0 gate — DISSOLVED 2026-08-24:** the Qwen program runs on its own box (A100), so it
      no longer waits on the OLMo all-layer re-runs; those continue on the original box
      (e1f_alllayer running, e2a_alllayer queued there). Start at Q1 directly. Coordination:
      commit each experiment's report (`git add -f`) and pull before starting a stage; the
      OLMo box does the same.
- [ ] **Q1 E1 distance sweep** + `--measure-attention`: headline ladder + fill-β null.
      `run_distance_sweep.py --model Qwen/Qwen2.5-7B-Instruct --reference-layer 0..27`.
      Preflight must include answer-extraction check on real replies.
- [ ] **Q2 E1c mass removal** (sufficiency): donor back_20, clamp local. Records Qwen's
      natural all-layer shares → calibrates Q3's ladder.
- [ ] **Q3 E1f dose-response**: 6 levels from Q2's natural down through its back_20 share.
- [ ] **Q4 E3 competition** (paired n=365 panel) + all-layer attention addendum + **E3c
      closure** (eager attention).
- [ ] **Q5 E5 system clamp**: share profile, then the neutral-context clamp arm
      (compliance collapse at graded dose).
- [ ] **Q6 E6 format erosion**: mmlu/gsm8k/code ladders, then recovery arms at the deepest
      depth of whichever arm erodes (do not assume mmlu erodes on Qwen — check first).
- [ ] **Q7 accumulation null**: random-subject MMLU stream, bounded null + adherence canaries.

## Runtime budget (32 GB RTX PRO 4500, bf16)

Q1 ~2h · Q2 ~1h · Q3 ~2h · Q4 ~4–6h · Q5 ~2–3h · Q6 ~3–4h · Q7 ~3–5h → **~17–23 GPU-hours**,
sequential. Wall-clock ~2 days with preflights and grader iteration.

## Family adaptation notes

- Capture already validated on Qwen2 GQA exactly 0.0 (`attention_capture.py` tests).
- Clamp biases the additive mask — family-generic; sdpa needs the padded-token trick
  (already in drivers); closure arms need eager attention.
- Qwen chat template injects a default system prompt when none is supplied — E5/E6 pass
  explicit system prompts, so verify the rendered transcript in preflight.
- Recheck `max_new` truncation per experiment: Qwen replies are longer-winded than OLMo's.
- Artifacts under `results/context_fatigue/qwen_*/`; one report per experiment quoting
  artifact filenames, n per cell, skip counts, verdict (brief §9 conventions).

## Box bootstrap (5090 or any fresh box on the /workspace mount)

```
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.12
ln -sf ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 /usr/local/bin/python
echo 'export HF_HOME=/workspace/.cache/huggingface' >> /root/.bashrc
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q  # expect all green
```

Qwen2.5-7B-Instruct and the datasets are in the HF cache on the mount; everything runs
`HF_HUB_OFFLINE=1`. If the mount is fresh instead, drop the offline flag once to download.

## Exact command queue (run in order; PREFLIGHT FIRST, always)

Add `--preflight` and swap the out-dir for `<name>_preflight` before each full run; read the
preflight transcript and replies before trusting the grader (Qwen's reply style is not OLMo's).
`QWEN=Qwen/Qwen2.5-7B-Instruct`, layers `$(seq -s' ' 0 27)`.

```
# Q1 — E1 distance sweep + attention
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_distance_sweep.py \
  --model $QWEN --reference-layer 24 --measure-attention \
  --out-dir results/context_fatigue/qwen_e1_distance_sweep
#   NOTE: --measure-attention records at --reference-layer (int); record the all-layer
#   pooled read in Q2/Q3 via the clamp drivers, which take the list.

# Q2 — E1c mass removal (also yields Qwen's natural all-layer shares)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_evidence_clamp.py \
  --model $QWEN --clamp-arm local --donor-arm back_20 --reference-layer $(seq -s' ' 0 27) \
  --out-dir results/context_fatigue/qwen_e1c_evidence_clamp

# Q3 — E1f dose-response; SET LEVELS from Q2's measured naturals:
#   6 levels from ~0.86x natural down through Q2's back_20 share, one below
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_evidence_clamp.py \
  --model $QWEN --clamp-arm local --levels <FROM_Q2> --reference-layer $(seq -s' ' 0 27) \
  --out-dir results/context_fatigue/qwen_e1f_share_sweep

# Q4 — E3 competition; then attention addendum; then E3c closure
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_competition_sweep.py \
  --model $QWEN --out-dir results/context_fatigue/qwen_e3_competition
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_competition_sweep.py \
  --model $QWEN --attention-only --head-layers $(seq -s' ' 0 27) \
  --out-dir results/context_fatigue/qwen_e3_attention
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_competition_sweep.py \
  --model $QWEN --close-arms --out-dir results/context_fatigue/qwen_e3c_competitor_close

# Q5 — E5 system clamp (profile, then neutral-context ladder; check driver flags on the day —
#   the ladder MUST be derived from the Qwen-measured profile, not OLMo's shares)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_system_clamp.py --model $QWEN ...

# Q6 — E6 ladders (recovery only after seeing which arm erodes on Qwen)
for F in mmlu gsm8k code; do
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_format_erosion.py \
  --model $QWEN --filler $F --n-probes 40 --max-new 256 \
  --out-dir results/context_fatigue/qwen_e6_$F ; done
#   depths per filler as in the committed OLMo runs (mmlu 0 3 7 14 21 28 35 42; others 0 2 4 6 9 12 15)

# Q7 — accumulation null, random-subject stream
HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_random_context.py \
  --model $QWEN --out-dir results/context_fatigue/qwen_random_context
```

## OLMo re-run state on the old box (restart if the swap kills them)

- `e1c_alllayer/` DONE, analyzed, addendum committed (36ccbc5).
- `e1f_alllayer/` — if turns.csv has fewer than 1345 rows, it died mid-run; relaunch:
  `run_evidence_clamp.py --clamp-arm local --levels 0.065 0.055 0.046 0.038 0.030 0.024
   --reference-layer $(seq -s' ' 0 31) --out-dir results/context_fatigue/e1f_alllayer`
- `e2a_alllayer` NOT STARTED: preflight `run_mass_clamp.py --reference-layer $(seq -s' ' 0 31)`
  first to read the all-layer natural query share, then set `--levels` at the committed run's
  fractions of natural (incl. one level above).
