# Verification Log

## 2026-07-06 — Analysis-only zip of investment_geometry + geoapp compatibility

1. **Output**: `out/zipped/investment_geometry_analysis.zip` (1.07 GB uncompressed content, 39,936 entries)
   - **How verified**: `unzip -l` entry count; extracted the full archive to scratchpad and listed the tree: contains `analysis/` (pca, embeddings, linear_probe, trajectories, relpos_counts.json), `config.json`, `summary.json`, `data/metadata.json`, `data/prompt_dataset.json`, and 4,588 sample dirs each with exactly `choice.json`, `prompt_sample.json`, `preference_sample.json`, `position_mapping.json`. Zero `.npy` raw-activation files under `data/` (18,354 JSONs added = 4,588×4 + 2).
   - **Result**: VERIFIED

2. **Output**: Original `out/geo/investment_geometry/data/` untouched
   - **How verified**: zip/find operations were read-only; after all work, re-listed `data/samples` (4,588 dirs) and `data/samples/sample_0/L0/*.npy` still present.
   - **Result**: VERIFIED

3. **Output**: `src/intertemporal/geoapp/data_loader.py` — analysis-only-bundle support (layers fall back to `summary.json`; `get_valid_sample_indices` falls back to position-mapping validity when no activations shipped)
   - **How verified**:
     - Mapping-derived validity vs npy-based validity on FULL real data: 126/126 targets identical (all 38 positions × 3 layers × 2 components + per-rel_pos variants) — `scratchpad/check_mapping_validity.py`.
     - Patched loader on zip-extracted data vs real data: 126/126 targets identical indices, identical layer discovery — `scratchpad/check_zip_equivalence.py`.
     - Real-data code path unchanged (same npy `any()` check; summary load reordered before `_discover_targets`, used only in fallback).
   - **Result**: VERIFIED

4. **Output**: geoapp server works on zip-extracted data
   - **How verified**: ran two servers (zip data :8765, real data :8766); diffed 16 endpoints (config, 5 embeddings incl. previously-broken `time_horizon` and per-rel_pos, 2 metadata colorings, 2 samples, metrics, heatmap, trajectory, tokens, scree, alignment): 15/16 byte-identical; `/config` semantically identical (all 13 keys compare equal, byte diff is JSON ordering only).
   - **Result**: VERIFIED

5. **Output**: geoapp UI works on zip-extracted data
   - **How verified**: started backend :8000 (zip data) + Vite dev server :3000; rendered `http://localhost:3000/investment_geometry` in headless Chrome and viewed the screenshot with image tokens: 3D PCA point cloud renders, "4,588 / 4,588 visible", 12 layers · 17 positions header, token/position panel populated, Time Horizon legend active (`scratchpad/ui_dev_zip.png`).
   - Production static mount (`/` on backend) serves index.html but has no SPA fallback for `/dataset` sub-paths — identical behavior on real data (pre-existing, not zip-related).
   - **Result**: VERIFIED

6. **Output**: branch working-tree deletions undone (118 files, `src/common/` etc.)
   - **How verified**: `git ls-files --deleted` → 0; `git status --short` shows only the intentional `data_loader.py` modification and pre-existing untracked dirs; `src/common/auto_export.py` present; geoapp imports resolve (servers ran from main repo).
   - **Result**: VERIFIED

## 2026-07-06 — geo bundle published to GitHub release `geo-bundles`

7. **Output**: Release `geo-bundles` with `investment_geometry_analysis.zip` split into 20×25MB parts + `.manifest` (21 assets)
   - **How verified**: local network corrupted large TLS uploads ("bad record MAC"; sandbox made it worse but 488MB failed even unsandboxed), so the bundle ships as parts. All 21 assets listed via API and matched against local files by name+size; then the full set was downloaded back from the public release URLs, reassembled, and SHA-256 compared: downloaded == local == manifest hash (8de28908…162b). Manifest flow of `run_geoapp_bundle.sh` tested end-to-end locally (assemble, checksum OK, 4,588 samples extracted, idempotent re-run, no-overwrite guard).
   - **Result**: VERIFIED

## 2026-07-31 — `paper/submission.zip` unpacked into `paper/`

8. **Output**: `paper/` updated with the 214 files from `paper/submission.zip`, zip then deleted
   - **How verified**: pre-flight `git status` showed the working tree clean apart from the untracked zip, and all 183 existing overwrite targets were confirmed git-tracked (`git ls-files --error-unmatch` on each) before any write. Extracted the zip to a scratchpad copy first and diffed it against the folder: 31 new, 116 changed, 67 identical. Applied with `rsync -a` (no `--delete`), then byte-compared every one of the 214 files against its repo copy with `cmp`: 214 identical, 0 mismatched. `git status` shows exactly 117 modified + 30 new files, and `comm` against the zip file list confirms every modified path came from the zip and nothing else was touched. Non-zip files (`references.bib`, `build.sh`, `package.sh`, `blackbox.tex`, `README.md`, `acl.sty`, `acl_natbib.bst`, `sections/cross_study_convergence.tex`, `figures/prompting_comparison.tex`) confirmed still present. Zip removed and its absence confirmed.
   - **Not verified**: the paper was NOT rebuilt. `bash build.sh` deploys to the public site path, so the PDF is untested against the new sources.
   - **Result**: VERIFIED (file sync only)

9. **Output**: paper rebuilt after the zip sync (`review.pdf`, `preprint.pdf`, `camera_ready.pdf`)
   - **The first build FAILED, and `exit 0` hid it.** `bash build.sh | tail` reported exit 0 while `latexmk` died: `./paperdefs.tex:72: LaTeX Error: Option clash for package lineno` / `Fatal error occurred, no output PDF file produced!`. The zip's `paperdefs.tex` loads `lineno` unconditionally (a COLM-build addition), but `neurips_2026.sty:405` already `\RequirePackage{lineno}` in its `main` mode, which the review and camera_ready variants use. Fixed by guarding the block with `\@ifpackageloaded{lineno}`. Byte-level file verification could not have caught this; only building could.
   - **Second defect, caught by the build**: `Citation 'herel2025timeawarenesslargelanguage' on page 25 undefined` in all three variants. The zip's `appendix/groundwork/extended_literature.tex` cites a key that does not exist in `references.bib` (which the zip did not ship); the same paper (arXiv 2409.13338) is there under `herel2024timeawarenesslargelanguage` from the verified-bib commits. Repointed the citation to the existing entry.
   - **How verified**: final `bash build.sh` run with output redirected (not piped) to capture the real exit status: `REAL_EXIT=0`. All three logs de-wrapped with `tr -d '\n'` first, because pdflatex hard-wraps at 79 chars and a naive grep silently misses split citation keys: 0 fatal errors, 0 undefined citations, 0 undefined references, 0 multiply-defined across all three. All PDFs (16:12-16:14) are newer than the newest source edit (16:11:17). `pdfinfo`: 150 pages each, letter. `out/light.pdf` is 19.5MB, far above the 2KB Ghostscript-stub failure mode. Rendered pages to PNG with `pdftoppm` and LOOKED at them: `review` p1 shows "Anonymous Author(s)", the "Submitted to ... Do not distribute" footer, and line numbers 1-34 in the margin (confirming the lineno guard preserved the style file's numbering rather than suppressing it); `preprint` p1 shows the 7-author block, affiliations, "Preprint." footer, no line numbers; `camera_ready` p1 shows authors with the final "40th Conference..." footer; p25 shows the formerly-broken citation resolved as numerals with no `[?]`; p78 shows new zip figures M.8/M.9 rendering with plots, captions, and resolved cross-references. All 6 sampled new images are referenced by a `.tex` file; 303 images embedded.
   - Deployed `preprint.pdf` confirmed byte-identical to `~/work/unrulyabstractions.github.io/pdfs/temporalpref.pdf`. No stray build artifacts in the repo root.
   - **Not verified**: only 5 of 150 pages were viewed (1, 25, 78 plus two title pages). The other 145 pages are UNVERIFIED by eye.
   - **Result**: VERIFIED (builds clean, all three variants correct at the pages inspected)

## 2026-07-31 — Llama-3.1-8B-Instruct load/hook feasibility investigation (read-only)

10. **Output**: feasibility findings for loading and hooking `meta-llama/Llama-3.1-8B-Instruct` through `src/inference`
    - **How verified**: ran the installed interpreter directly rather than trusting docs or memory. `OFFICIAL_MODEL_NAMES` enumerated in-process (the Instruct id IS present); `get_official_model_name` resolved `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.1-8B` and raised `ValueError` on `meta-llama/Meta-Llama-3.1-8B`; `get_pretrained_model_config` returned n_layers=32, d_model=4096, n_heads=32, n_key_value_heads=8, n_ctx=2048. Installed version read from package metadata as **3.0.0b3** (not the 3.0.0b1 stated in the task), with transformers 5.3.0 / torch 2.9.1. Backend auto-selection executed in-process: `get_recommended_backend_inference()` → MLX, `get_recommended_backend_interventions()` → TRANSFORMERLENS. Gated-repo access proven with `HfApi.model_info` under the resolved token (both repos `gated="manual"`, access OK; `whoami` = unrulyabstractions). Hub cache inspected: only tokenizer/config for the 8B are present (8.7 MB), weights NOT downloaded. End-to-end plumbing proven by actually loading `gpt2` and then the gated `meta-llama/Llama-3.2-1B-Instruct` through `ModelRunner(..., backend=TRANSFORMERLENS)` and caching all six components — `blocks.{L}.hook_{resid_pre,resid_mid,resid_post,attn_out,mlp_out}` and `blocks.{L}.attn.hook_z` — 16 keys each, correct shapes, GQA expanded to 32 heads in `hook_z`.
    - **Result**: VERIFIED (registry, auth, config and hook plumbing). The 8B model itself was NOT loaded — its ~16 GB of weights are not cached locally, so the 8B forward pass and its true peak memory are UNVERIFIED and all VRAM figures below are ESTIMATES, not measurements.
    - **Defects found (not fixed)**: `scripts/activation_extraction/activation_extractor.py:175` uses the non-existent id `meta-llama/Meta-Llama-3.1-8B`; the `scripts/experiments/submit_*.sh` family sets `HF_TOKEN=$(cat ~/.cache/huggingface/token || echo "")` and no such token file exists here, so those scripts clobber a working env token with the empty string; the Llama-3.1 chat template injects a `Today Date:` system line into every prompt, which is a live confound for a temporal-awareness study.


## 2026-07-31 — NeurIPS 2026 rebuttal materials prepared (submission 25332)

10. **Output**: `paper/rebuttal/` with four OpenReview-ready posts plus two working documents
   - **How verified**: Read all 57 pages of `paper/rebuttal/resources/` directly (`neurips.pdf` 15pp, `temp_review.pdf` 21pp, `colm.pdf` 4pp, `co_collab_ideas.pdf` 6pp, `some_new_exp.pptx.pdf` 11pp), not summaries. `temp_review.pdf` is the same OpenReview thread as `neurips.pdf` at an earlier timestamp, and its last 10 pages are duplicate copies of the PAT feedback; no unique content.
   - Every number in the rebuttal was checked against the slide-deck tables in `some_new_exp.pptx.pdf`, which render cleanly, rather than against the garbled text extraction in `co_collab_ideas.pdf`: both localization rows (layers, pair counts, resid_post windows, top attn_out and mlp_out layers with fractional depths), the steering row (probe layer, steering layer, alpha), and all 16 IFEval values. All match.
   - **Two of my own overstatements found and corrected before finalizing**: (a) I wrote the IFEval prompt-level drop as "four to five points"; computing all four deltas gives 3.6 to 4.6 pp, so it now reads "about four points". Instruction level is 3.2 to 3.4 pp and reads "about three points". (b) I wrote the shared residual window as "half to two thirds of depth", but gemma's denoising window reaches 0.76, so it now reads "half to three quarters".
   - IFEval prompt totals cross-checked: 464+77 and 510+31 both equal 541, which matches the benchmark size, so the skipped-prompt counts are internally consistent.
   - Character limits verified against OpenReview's 5000 cap by counting only the postable body after the `---` rule: 3990, 3606, 3029, 3453. All fit. No em-dashes or en-dashes in any file.
   - **Discrepancy found in the source material, flagged not silently resolved**: our own two documents disagree on gemma-2-9b-it's best steering layer. Both results tables say L25 (0.59 depth); the conclusion paragraph in `co_collab_ideas.pdf` says "at L23 before the L28", and L23 is the best *probe* layer. The rebuttal uses L25 and the discrepancy is logged as open question F.1.
   - **Not verified**: the underlying gemma experiments themselves. I have not seen the raw outputs, notebooks, or the Google Drive and Colab artifacts linked in the deck. Every new number is taken on the collaborators' word from the deck and is UNVERIFIED at the data level. The paper was not modified in this session.
   - **Result**: VERIFIED (materials are internally consistent and faithful to the sources read); underlying experimental data UNVERIFIED

## 2026-07-31 — Llama/vast port, stage 0: unblock, model-agnosticism, and the destroy gate

11. **Output**: `src/common/math/entropy_diversity/entropy_primitives.py` — added the missing `_EPS` re-export
   - **How verified**: traced the cascade to its root by importing each leaf module directly (`auto_export._import_safe` swallows `ImportError`, so the failure surfaces far from its cause). Before: `scripts/intertemporal/run_intertemporal_experiment.py --help` died with `ImportError: cannot import name 'PreferenceDataset'`. After: `--help` prints the full usage block, exit 0. New regression test `tests/common/test_import_integrity.py` imports 14 modules leaf-first and asserts 6 explicit re-exports: 21/21 pass, and it fails at HEAD.
   - **Result**: VERIFIED

12. **Output**: `src/intertemporal/common/chat_template_boundaries.py` — model-agnostic chat boundaries
   - **How verified**: replaced the special-token blacklist with a sentinel round-trip through the real tokenizer. Ran both tokenizers and printed the result: Qwen3-4B suffix is `['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']` with a 3-token prefix; Llama-3.1-8B suffix is `['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n']` with a 30-token prefix. `assistant` sits at relative position 3 on Qwen and 2 on Llama, so raw `chat_suffix:N` indices are not comparable across families; role aliases (`turn_end`, `turn_start`, `role_assistant`) resolve correctly for both. 8/8 tests pass, including one asserting `prefix + body + suffix` reconstructs the template exactly.
   - **Confound confirmed, not fixed**: Llama's 30-token prefix contains `Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024`. Independently found by the paper session (entry 9). Not yet suppressed; awaiting a decision on whether to pin the date or run it as an A/B.
   - **Result**: VERIFIED (boundaries); date confound OPEN

13. **Output**: coarse sweep now covers all layers by default
   - **How verified**: `min_layer_depth` defaulted to 0.45 and `CoarsePatchingConfig` had no field for it, so `process_coarse` could never override it. Read the swept layers straight off the flagship run rather than trusting the source: every component in `out/experiments/investment/aggregated/coarse/*.json` has `by_start` keys exactly `[16..35]`, so layers 0-15 were never patched and the paper's "L17-35" cannot be separated from the sweep's own lower bound. Changed the default to 0.0, added `min_layer_depth`/`max_layer_depth` to the config, and forwarded them at the call site. 7/7 tests pass, 6 of which fail at HEAD.
   - **Result**: VERIFIED (code); the published L17-35 claim remains BOUNDED BY THE SWEEP and is flagged for the authors

14. **Output**: `src/intertemporal/common/model_layers.py` — layer selections projected by fractional depth
   - **How verified**: `geometry_utils.LAYERS` and `semantic_positions.DEFAULT_LAYERS` both contain 34 and 35, which do not exist on Llama-3.1-8B (32 layers). Scaling by fractional depth (the coordinate Appendix Q found to be scale-invariant) maps the geometry list to `[0,1,3,11,16,17,18,19,20,21,22,25,27,30,31]`. 11/11 tests pass, asserting endpoints are preserved, fractional depth is held within 0.02, and the raw lists still raise for a 32-layer model.
   - **Result**: VERIFIED

15. **Output**: no regressions from any of the above
   - **How verified**: ran the full suite in a pristine `git worktree` at HEAD and in the working tree with identical exclusions, then diffed the failed-test sets. HEAD: 300 passed / 29 failed / 60 errors. Working tree: 404 passed / 27 failed / 8 errors. The single test that fails for me but not at HEAD (`test_intervention_config_from_sample_file_format`) is shown by the HEAD log to have ERRORed at setup there, so it never ran; it is newly visible, not newly broken. Zero true regressions. Worktree removed afterwards.
   - **Result**: VERIFIED

16. **Output**: `cloud/` instance-ownership tracking and `scripts/verify_experiment_output.py` destroy gate
   - **How verified**: this vast account is shared with other agents. Baseline scan recorded the one pre-existing instance (`46467709`, RTX 3090, label=None) into `cloud/.instances_foreign`; the ours-ledger is empty. Tested the reaper against it: reaping the foreign id prints `[FOREIGN] ... REFUSING`, an unknown id prints `[UNKNOWN] ... REFUSING`, and `--all` on an empty ledger is a no-op. Re-listed instances afterwards: `46467709` still running, untouched. Ownership requires BOTH a ledger entry AND a live label matching `ta-tp`, so a stale ledger line cannot authorise a destroy.
   - Verifier validated against known-good data before being trusted as a gate: it passes on `out/experiments/investment` (71 pairs, 5 components, layers 16..35 read from the JSON) and on `out/geo/investment_geometry` (4,588 of 4,588 samples, 3,116 PCA targets), and returns BROKEN with exit 1 for a missing run and for `--pulled` with no manifest. An absent or empty pull manifest is treated as failure, not success.
   - **Result**: VERIFIED

## 2026-07-31 — Reward rendering, Gemma support, and run-scoping facts

17. **Output**: reward values no longer collapse to zero or to floating-point noise
   - **How verified**: traced "0 quality-adjusted life years" to its real source. It was NOT the renderer: `RewardValue(value=round(...))` at `prompt_dataset_generator.py:446,452` rounded unconditionally, ignoring the `round_reward_units` flag, so health's intended 0.5 QALY step became 0 before any formatting happened. Confirmed by printing the generated grid: `generate_steps` returns `[0.5, 1.4, 2.3, 3.2, 4.1, 5.0]` while the stored rewards were `[0,1,2,3,4,5]`. Three coordinated fixes: `normalize_reward` snaps log-stepping noise (30000.000000000007 -> 30000), `format_reward` renders fractions faithfully, and the unconditional round is removed so the config flag governs.
   - **Regression guard, the point of the exercise**: generated all 10 domains at pristine HEAD and in the working tree and compared SHA-256 over the concatenated prompt text. 8 of 10 byte-IDENTICAL including `investment` (4,590 prompts, `f6390f88384a46b6`), which is the paper's dataset. Only `health` and `wellbeing` changed, which are exactly the two domains whose fractional rewards were being destroyed. An intermediate version also changed `cityhousing`, surfacing `30,000.000000000007 housing units`; that was a regression I introduced, caught by the same check, and fixed by setting `round_reward_units: true` for countable housing units, after which cityhousing returned to byte-identical.
   - Full-domain scan for zero or absurd-precision rewards: 0 hits across health, wellbeing, cityhousing, investment, charity. 35 tests pass.
   - **Result**: VERIFIED

18. **Output**: `resolve_role_positions` made genuinely model-agnostic (bug in my own earlier fix)
   - **How verified**: ran the resolver against Gemma-2-9B-it and got `{}` — it keyed on the literal token "assistant" and on the `<|...|>` bracket convention, and Gemma uses `model` and `<start_of_turn>`. That is the same blacklist failure mode I had just removed elsewhere. Rewritten to take the tokenizer's special-token set and identify the role name structurally as the only plain-text token in the generation suffix. Verified on all three families: Qwen `{turn_end:<|im_end|>, turn_start:<|im_start|>, role_assistant:assistant}`, Llama `{<|eot_id|>, <|start_header_id|>, assistant}`, Gemma `{<end_of_turn>, <start_of_turn>, model}`. 14 tests pass.
   - **Result**: VERIFIED

19. **Output**: run-scoping facts established for the Llama/Gemma geometry runs
   - **How verified**: queried the live APIs rather than assuming. `google/gemma-2-9b-it` and `meta-llama/Llama-3.1-8B-Instruct` are both `gated=manual` but both resolve with the local HF token, and both are in the TransformerLens registry. HF dataset `unrulyabstractions/temporal-awareness` exists, is public, and currently holds only `.gitattributes`; `whoami` confirms write access as `unrulyabstractions`. Vast supports `create volume` and `create network-volume`; `search network-volumes` returns 0 offers, but `search volumes` returns 64, cheapest 337 GB at $0.0267/GB/month in Indiana. Plain volumes are machine-bound, so the GPU instance must be scheduled on the volume's machine — a real constraint on the fleet, not yet resolved.
   - **Not verified**: no model has been loaded on a GPU, no activation extracted, no instance launched. Fleet still shows OURS=0.
   - **Result**: VERIFIED (facts); runs UNVERIFIED / NOT STARTED

## 2026-07-31 — gemma-2-9b-it results added to the paper (new Appendix R)

11. **Output**: `appendix/characterize/cross_family_replication.tex`, a new figure, three new bib entries, index renumbering, and a rebuilt PDF
   - **How verified**: URLs were taken from the PDF link annotations in `some_new_exp.pptx.pdf`, not transcribed from the rendered page. This mattered: the visible text wraps and corrupts one Colab id, which reads `DfSHInkCkrJ1ulf` on screen but is `DfSHlnkCkrJ1ulf` (lowercase L) in the annotation. After building, the four hyperlink targets embedded in `out/preprint.pdf` were extracted and compared against the source annotations: all four match exactly.
   - Figure extracted from the deck with `pdfimages` (1784x1480) and VIEWED with image tokens before use. It corroborates the localization table independently: L28_attn leads, then L38_mlp and L39_mlp, matching the stated depths.
   - Depth arithmetic checked for every layer claim: gemma L28/42 = 0.667, L38/42 = 0.905, L39/42 = 0.929, L25/42 = 0.595, L26/42 = 0.619, L23/42 = 0.548; Qwen L24/36 = 0.667, L21/36 = 0.583, L31/36 = 0.861, L26/36 = 0.722, L22/36 = 0.611. All agree with the deck's stated fractional depths. Steering coefficient ratio 150/50 = 3 exactly.
   - **Three errors in my own draft, caught before the final build**: (a) I referenced `app:latent-vs-constrained` for the classification experiment; reading the source shows it is `app:causal-contrastive`. (b) I wrote that the classification experiment "does not recruit the late Qwen3 MLPs at all", but `causal_contrastive.tex` explicitly reports L33 and L35 MLPs producing robust effects, and lists L31 in the primary MLP writer band. The collaborator's claim that L31 and L35 are "not so important there" contradicts the paper's own appendix, so I removed the assertion instead of repeating it. (c) I claimed the error bars were computed across the 88 pairs, which the deck never states; softened to "one standard deviation".
   - **Rendering defect caught by looking at the page**: cross-references printed as "Appendix Appendix Q", because `\thesection` already expands to "Appendix Q". Fixed all five to bare `\ref`, matching house style, and confirmed zero occurrences of "Appendix Appendix" across all 154 pages of the rebuilt PDF.
   - **A build failure I caused**: killing an in-flight build with `pkill` truncated `out/camera_ready.aux` mid-token, and the next build died with "File ended while scanning use of \@writefile". Confirmed the truncation by reading the tail of the aux file, cleared the intermediates (gitignored, regenerable), and rebuilt clean.
   - Final build: `REAL_EXIT=0` with output redirected rather than piped. All three variants: 0 fatal, 0 undefined citations, 0 undefined references, 0 multiply-defined, 154 pages each (up from 150). The three new bib entries resolve in the `.bbl`.
   - **Index renumbering verified programmatically**: inserting Appendix R shifted 12 hardcoded index letters. Every one of the 30 index rows was checked by opening its claimed page in the built PDF and matching the rendered "Appendix X" heading. 30 of 30 correct, 0 mismatches. No prose was affected because all 238 appendix cross-references use `\ref`.
   - **Anonymity gate verified in both directions**: the artifact links are wrapped in `\if@anonymous`. The preprint build renders all four URLs as live links; the review build prints the withholding sentence and contains zero drive.google or colab.research URLs anywhere in the appendix range.
   - Pages 110, 111, 112 and the index page 19 were rendered to PNG and read directly.
   - **Not verified**: the underlying gemma experiments. I have not opened the linked Drive folders or Colab notebooks, and every number still rests on the collaborators' deck. The figure is a bitmap and should be regenerated as vector art from the source, which is the same complaint Reviewer APpt raised about existing figures.
   - **Result**: VERIFIED (builds clean, renders correctly, internally consistent); underlying experimental data UNVERIFIED

## 2026-07-31 — Turn-transition geometry scoping + fleet, ready to launch (nothing launched)

20. **Output**: `--turn-only` geometry scoping verified for both target models
   - **How verified**: ran `generate_geometry_samples.py --dry-run` for each model/domain pair and read the resolved target set. Llama-3.1-8B-Instruct/health resolves layers ending at 31 (32-layer model); gemma-2-9b-it/climate ends at 41 (42-layer model), so the per-model layer projection is correct and nothing is clamped or silently dropped. Both resolve positions to exactly `chat_suffix` + `chat_suffix_tail`, components to `resid_post` + `attn_out`, dtype float16, **n_targets=60** each (15 layers x 2 components x 2 positions) versus the default full-position set. This is the change that takes the run from ~112 GB to roughly 3 GB per model.
   - **Result**: VERIFIED

21. **Output**: `cloud/sync_up.sh` will not upload the 156 GB `out/` tree
   - **How verified**: did not trust the script's own exclude list. Ran an independent `rsync -an --stats` with the same exclusions against a throwaway destination: **12,830 files, 449,887,145 bytes (450 MB)**. `data/` (103 MB, needed for the domain configs) is correctly included.
   - **Result**: VERIFIED

22. **Output**: fleet ownership still clean after all fleet-script work
   - **How verified**: `bash cloud/fleet_status.sh` reports OURS=0, ours-burn $0.000/hr. The foreign instance has now changed three times during this session (46467709 RTX 3090 -> 46472662 RTX 4090 -> 46475315 RTX 4090) as another agent cycled boxes; each new one was auto-recorded as FOREIGN by the scan and none was touched. `launch_run.sh` requires `YES=1` or an interactive confirm, labels with `$LABEL_PREFIX-$RUN`, and calls `ledger_add_ours` immediately after create and before polling, so a crash mid-poll cannot orphan an unrecorded billing box.
   - **Hardware selected but NOT rented**: bf16 is required (Llama-3.1 and Gemma-2 are bf16-trained; Gemma-2 is known to overflow in fp16), which rules out the cheapest 48 GB offers because Q RTX 8000 is Turing. Filtering `compute_cap>=800` gives RTX 3090 24 GB at ~$0.11/hr for Llama and A100 40 GB / A6000 47 GB at ~$0.40/hr for the larger Gemma. Combined ~$0.52/hr against $33.90 credit.
   - **NOT VERIFIED — nothing has been run on a GPU.** No model weights downloaded, no activation extracted, no instance created, no data written to the HF dataset. The two runs remain NOT STARTED.
   - **Result**: VERIFIED (readiness); runs NOT STARTED

## 2026-07-31 — rebuttal.pdf generated from Markdown; discussion package adopted

12. **Output**: `paper/rebuttal/rebuttal.pdf` + `build_rebuttal.sh`, five rewritten comment files, and the new appendix moved to the end of the paper
   - **Three factual claims in the supplied discussion package were checked against the paper source before adoption, and all three hold**: (a) the 30-model / five-family / three-API-provider / 28,800-sample battery exists (`extended_methods/behavioral_coherence.tex` lines 21 and 25), and the model list is subsection 3, so "Appendix AA.3" is correct; (b) the two-stage edge procedure exists (`extended_methods/attributional_contrastive.tex` line 9: EAP-IG restricts the node set, then edges are scored and pruned); (c) Kirby MCQ-27 reports both a non-drug-using control group (k about 0.013) and a heroin-dependent group (k about 0.025).
   - **This overturned my own earlier rebuttal.** In entry 10 I had conceded to Reviewer APpt that "our experiments identify important nodes and their depth, and they do not trace the edges between them". That concession contradicts the paper's own methodology and would have surrendered a defensible claim. Corrected.
   - **But the package's version was also too strong.** `appendix/localize/attributional_contrastive.tex` contains zero occurrences of "edge", and no edge count or sparsity number appears anywhere in the paper. The method is described; the results are not shown. The posted W4 answer now says exactly that, and promises the results rather than claiming they are already present.
   - **Kirby arithmetic verified independently**: 0.013/0.0041 = 3.2 and 0.013/0.0016 = 8.1, which reproduces the paper's "3 to 8 times more patient" figure. That confirms the headline comparison is against the control group, so i7iD's objection is answered by naming the group rather than by replacing the baseline.
   - **Appendix letter correction**: the package (and the PAT report) call the cross-model patching appendix "P". It is **Q**. Verified from `out/preprint.aux` label map.
   - **Structural fix**: my earlier insertion of the gemma appendix after Q had shifted 12 letters, silently invalidating the package's own "Appendix V" and "Appendix AA.3" references. Moved the new appendix to the end (now AD) so no existing letter changes. Re-verified all 30 index rows against the `.aux` label map: 0 mismatches. `V` is back to Attributional contrastive methodology (p122) and `AA` to Behavioral coherence methodology (p133). Paper rebuilt, `REAL_EXIT=0`, 154 pages.
   - **rebuttal.pdf verification**: built via `build_rebuttal.sh`, which extracts the postable body of each `.md` (everything after the `---` rule), renders it with the LaTeX markdown package under lualatex, and computes character counts live. Rendered pages 1, 2 and 7 to PNG and read them. Three rendering defects were found by looking and then fixed: the cover table overflowed the right margin (switched to tabularx), the `**Title:**` line was duplicated as both heading and body text (moved above the `---` so it is not pasted and not counted), and markdown subheadings numbered themselves "0.1" (secnumdepth). A wide results table was clipping its last column, fixed with a footnotesize hook on `tabular`.
   - **Tracking proven by round trip, not asserted**: appended a sentinel string to `ac_confidential.md`, rebuilt, confirmed it appeared in the PDF (count 0 to 1), removed it, rebuilt, confirmed it was gone (count 1 to 0).
   - All five comments are under the OpenReview 5000-character cap: 3481, 3420, 2767, 3725, 2372. No em or en dashes in any postable body.
   - **Not verified**: the underlying gemma experiments (still only the collaborators' deck), and whether edge-level results can actually be produced from existing Appendix V pipeline output. The W4 answer commits us to showing them.
   - **Result**: VERIFIED (claims checked against source, PDF renders correctly, tracking demonstrated); gemma data and edge-result availability UNVERIFIED

## 2026-08-01 — Fleet launched; three real bugs caught by verification, sync blocked

23. **Output**: two boxes running, both bf16-capable, both recorded as OURS
   - `46477302` RTX 3090 24 GB (cc=860, up 432 Mbps) = `ta-tp-llama31-8b-health`
   - `46478124` RTX A6000 46 GB (cc=860, up 5,958 Mbps, Delaware) = `ta-tp-gemma2-9b-climate`
   - **How verified**: `nvidia-smi` on each box returned the real GPU and VRAM; `which rsync/git` returned real paths. Ledger shows OURS=2, FOREIGN separate.
   - **Three bugs found and fixed, each caught only because the result was checked rather than assumed**:
     (a) `launch_run.sh` had no `compute_cap` floor, so cheapest-first rented a **Q RTX 8000 (Turing, no bf16)** for Gemma-2, which overflows in fp16. Added `MIN_CC=800`.
     (b) `reap.sh` ran `vastai destroy` without answering its interactive prompt: the destroy **aborted while the pipeline still exited 0**, so an audit line claiming DESTROY was written for a box that was still running and still billing. Now pipes `y` AND polls the API to confirm the instance is gone before touching the ledger or writing the audit line.
     (c) `is_ours` set `ID`/`PREFIX` inline before a pipeline, which binds them to `printf`, not to the `python3` that reads them. It crashed, the box fell through to the "stale" branch, and a box we owned was reclassified FOREIGN — meaning our own reaper would have refused to kill it. Now exported. Ledgers were repaired from the authoritative signal, the instance label.
   - Also swapped the first Gemma box (RTX 4090, Sichuan, **18.7 Mbps up**) for the A6000: a 3 GB HF sync would have taken ~20 minutes on it, and HF is the only durable copy. Added `MIN_UP=500` to the offer query.
   - **Result**: VERIFIED (hardware, ownership, teardown gate)

24. **Output**: repo sync to the boxes — BLOCKED, runs NOT STARTED
   - **How verified**: `sync_up.sh` fails with rsync exit 255 / broken pipe after 6 retries on both boxes. SSH itself works (`at_box.sh` runs commands fine and the login banner appears), and rsync and git are both present on the remote, so this is not a missing binary or an auth failure. Most likely the vast proxy-SSH login banner corrupting rsync's stdio protocol; the fix is to use the direct SSH endpoint or `--rsh` with the banner suppressed. Not yet resolved.
   - **NOT VERIFIED**: no weights downloaded, no activations extracted, nothing written to the HF dataset. Both boxes are idle and billing at $0.669/hr combined.
   - **Result**: BROKEN (sync), runs NOT STARTED

25. **Output**: both boxes torn down; account clean; total spend ~$0.30
   - **How verified**: before destroying, confirmed nothing had been produced — `find /root/temporal-awareness/out -type f | wc -l` returned **0** on the Llama box (42 repo files had partially transferred, no outputs). `reap.sh` then destroyed both and polled the API until each was gone; `fleet_status.sh` now reports OURS=0, FOREIGN=0, burn $0.000/hr. Nothing was lost because nothing existed.
   - **Blocker, unresolved**: the repo could not be transferred to either box. Ruled out, by testing: SSH auth (works), rsync/git present on the remote (both at /usr/bin), the vast proxy host (direct IP:port behaves the same), and a broken local `~/.ssh/config` (`usekeychain` is rejected by the Homebrew ssh on PATH, but `SSH_EPHEMERAL_OPTS` already passes `-F /dev/null`). Plain SSH commands succeed; both rsync and `tar | ssh` die with the remote closing the connection once bulk data flows. Next thing to try is the vast HTTP/scp path or `vastai copy`, or pulling the repo on the box with `git clone` instead of pushing it.
   - **Result**: VERIFIED (teardown, zero data loss); sync BROKEN, runs NOT STARTED
