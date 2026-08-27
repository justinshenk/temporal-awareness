# Paper B repo split — survey and plan (deferred, 2026-08-27)

Goal: extract Paper B (context fatigue) into its own repo under Ronen's GitHub account.
Current origin is `justinshenk/temporal-awareness`, so the new repo also moves ownership.
Survey done 2026-08-27. Split itself deferred until Ronen says go.

## Coupling is thin: 4 cross-paper import lines total

Paper B → outside:
- `src.common.base_schema` (4 uses), `src.common.bootstrap_stats` (2), `src.common.auto_export` (1).
- `scripts/context_fatigue/run_format_steering.py:42` imports `AdditionSteeringHook, DirectionProjectionHook`
  from `src.probes.safety.steering_hook`. That file depends only on torch. Copy it wholesale.
- `scripts/context_fatigue/run_instruction_adherence.py:49` imports `build_cases, select_valid_indices`
  from `src.probes.lora_icl.ddxplus_cases`. That chain adds `src/probes/ddxplus.py` (stdlib-only).
  Note: Paper B already has its own `src/probes/context_fatigue/ddxplus_cases.py` (13 imports).
  Reconcile the near-duplicate instead of copying both.
- Zero imports from `src/probes/attribution`.

Paper A → Paper B (fix in the old repo after the split):
- `scripts/safety/analyze_route_sweep.py:18` imports `pearson` (one-function statistic, inline it).
- `scripts/safety/run_attention_base_vs_lora.py:26` imports `SelectiveAttentionCapture` (vendor the class).

## What moves

| Path | Files | Size |
|---|---|---|
| `results/context_fatigue` | 677 | 428M (546 `.npz` in `e3c_hot_close/` + `e1_rows/`) |
| `data/context_fatigue` | 40 | 4.9M |
| `scripts/context_fatigue` | 38 | 548K |
| `context_fatigue_paper` | 20 | 1.1M |
| `src/probes/context_fatigue` | 13 | 164K |
| `tests/probes/context_fatigue` | 13 | 148K |
| stray tracked results dirs | 16 | small |

Stray Paper B results outside `results/context_fatigue` (do not miss): `results/f90871_steering`,
`results/random_context`, `results/olmo_gradient`, `results/olmo_gradient_n35`,
`results/olmo_attention_sft`.

Shared code to duplicate (not worth a shared package): `src/common` minus `null_intervals.py`
(Paper A only), i.e. `base_schema.py`, `bootstrap_stats.py`, `auto_export.py`, `file_io.py`,
`__init__.py`. The `src/common/{analysis,math,choice,logging,profiler}` subdirs have no tracked
files. Ignore them.

Also needed:
- `data/adversarial/narrativeqa/narrativeqa_modifications` (check whether the whole dir or a subset).
- Tests infra: `tests/conftest.py` (repo root on sys.path, `slow` marker, `--skip-slow`),
  `tests/__init__.py`, `tests/probes/__init__.py`, and `tests/common/test_bootstrap_stats.py`.
- `pyproject.toml` trimmed to ~15 deps (torch, transformers, datasets, numpy, pandas, scipy,
  scikit-learn, matplotlib, seaborn, tqdm, sae-lens, huggingface_hub, accelerate, anthropic).
  Drop dash/fastapi/uvicorn/plotly/pacmap/umap/nnsight/pyvene/transformer_lens/peft and the
  `latents` git dep. Keep `[tool.pytest.ini_options] pythonpath=["."]` and the ruff config.
  Regenerate `uv.lock`, do not copy it.
- `Makefile`: keep `install`, `test`, `paper-b`. Drop `paper-a`.
- `.github/workflows/ci.yml`: prune the nonexistent `experiments/` lint path and stale `--ignore`s.
- `CLAUDE.md` forked with the Paper A entry deleted. `tasks/` comes along (CLAUDE.md references it).
  Paper-B tasks: `context_fatigue_dilution_localization.md`, `context_fatigue_worries.md`,
  `e3c_competitor_close_brief.md`, `format_patch_brief.md`, `qwen_reproduction.md`,
  `per_token_capture_brief.md`, `compute_vs_communicate_L20.md`, plus `todo.md`, `lessons.md`,
  `current_task.md`.
- `.env.example`, `.gitignore` (see decision 3), `README.md` rewritten, `docs/` checked for
  Paper B content (11 files).

Paper B scripts are pure argparse. Nothing in `configs/` moves. Prompt templates are inline in
Python. Datasets (ddxplus, WildChat, mmlu, narrativeqa, gsm8k) download from HuggingFace at runtime.

Layout constraint: no `sys.path` hacks and no `__init__.py` under `scripts/`. Drivers sibling-import
(`from _cf_common import ...` × 16, `from run_distance_sweep import ...` etc.). The new repo must
keep "run from repo root with root on PYTHONPATH".

## Decisions to make before executing

1. **The 428 MB of `.npz`.** They are pattern-ignored yet force-added. Options: (a) history-preserving
   `git filter-repo` then strip `.npz` blobs and re-add only the ~130 small files (`.csv`, `.json`,
   `.md`), publishing `.npz` as a GitHub release artifact or keeping them on the A100 box; (b) carry
   them (history preservation drags all 428 MB of blob history regardless of later deletion).
2. **History vs fresh start.** `git filter-repo --path` over the ~6 Paper B paths preserves provenance
   (paper numbers trace to commits) but needs `git-filter-repo` installed (not on this machine yet)
   and carries the blob weight. Fresh `git init` + copy is clean and small but severs the
   commit-level provenance chain the audit relies on.
3. **`.gitignore` honesty.** The old repo ignores `results/`, `data/`, `tasks/` yet tracks 995, 65,
   19 files there via force-adds. The new repo should whitelist explicitly
   (e.g. ignore `results/**/*.npz`, track the rest) so intent is visible.
4. **`ddxplus_cases.py` duplication** (see above).

## Post-split cleanup in the old repo

Inline `pearson` and vendor `SelectiveAttentionCapture` in `scripts/safety/`, then delete the six
Paper B trees. Not urgent, can lag the split.
