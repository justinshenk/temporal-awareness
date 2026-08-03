# Current Task: Local MacBook runs — Experiments A, B, C (2026-08-03)

DURABLE LOCATIONS (a machine OOM crash at 15:39:55 wiped /tmp — never put scripts there):
- session scripts: `scripts/scratch/` (gitignored)
- run logs: `out/logs/`
- run outputs: `out/experiments/`, `out/geo/`, `out/steering_ci/`

## Memory rule (crash root cause)
Crash: swap 58.4/59.4 GB, a foreign `ollama` process at 20.8 GB RSS beside a ~16 GB
Llama. Before EVERY model load run `bash scripts/scratch/mem_gate.sh`; if it exits
nonzero, WAIT and report. Never kill another person's process. One model resident
at a time, always.

## Experiment A — break model-by-domain confound (INVESTMENT domain)
- Config `data/intertemporal/investment/investment_local.json` -> 1,512 samples.
- Llama-3.1-8B-Instruct + Mistral-7B-Instruct-v0.3, TransformerLens on MPS,
  `TA_TL_NO_PROCESS=1` (fp32 weight processing OOM-kills 7-8B loads here),
  n_pairs 24, all layers, resid_post+attn_out+mlp_out.
- Smoke gate PASSED (L16-17): 24/24 pairs, 0 skips, sanity recovery/disruption
  1.000 on every pair, clean baseline logit diff 2.39-6.20.
- Llama full run RESTARTED CLEAN after the crash — `--cache` cannot resume
  (`_use_cached_pairs` forces n_select = cached pair count, so it would have run
  10 pairs, not 24). Pair selection verified deterministic, so the rerun rebuilds
  the identical pairs. Crashed run's 10 pairs preserved at
  `out/experiments/loc_llama_investment_20260803_154638`.
- Verify each with `scripts/verify_experiment_output.py --patching <dir>`.

## Experiment B — specificity null on Qwen3-4B-Instruct-2507 (RISK contrast)
- Configs `data/intertemporal/risk/{risk_local,risk_geometry}.json` survived the
  crash. risk_local -> 576 samples, verified: same SITUATION/TASK/OBJECTIVE/
  CONSTRAINT/ACTION/FORMAT markers as investment, certain (a, "1 hour") vs
  50%-gamble (b, "2 hours"), horizon constant "1 year", 24 unique reward strings
  with no substring collisions.
- Coarse sweep -> `out/experiments/loc_qwen_risk`; then turn-token PCA via
  generate_geometry_samples.py --turn-only --components resid_post.
- Deliverable: risk attention peak/band vs the temporal band 0.58-0.67, and a
  per-layer silhouette verdict at the assistant token.

## Experiment C — steering per-prompt scores + 10k bootstrap CIs
- `scripts/intertemporal/steer_turn_preference.py` already patched: score_items
  returns per-prompt diffs (means unchanged).
- Scripts recreated durably: `scripts/scratch/steer_ci_rescore.py`,
  `scripts/scratch/steer_ci_bootstrap.py`. Outputs -> `out/steering_ci/`.
- Inputs verified: 8 vector .npz mirrors, unit norms, best cells Qwen L18 a20,
  Llama L18 a35, Gemma L21 a50, Mistral L19 a20; all stored runs used bfloat16.
- Gate: recomputed S / S_ctrl / baseline must reproduce the stored values before
  any CI is quoted.

## Queue order (one model resident at a time)
1. Llama investment (running) 2. Mistral investment 3. Qwen risk + PCA 4. Steering CIs.
