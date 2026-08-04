# Task: geometry v2 re-run after the turn-position fix (2026-08-03/04)

## Why
`preference_querier.py` used to cache activations over `apply_chat_template(prompt + response)`,
which puts the response inside the user turn and moves the chat suffix after it, while
`SamplePositionMapping` indexes `chosen_traj.token_ids` (suffix before response). Every
position labelled `chat_suffix`/`chat_suffix_tail` therefore held a response token.
Fixed in 0ea7e91 (`run_with_cache(token_ids=...)`). See VERIFICATION_LOG entries 23, 28.
All published turn-transition geometry is invalid and must be regenerated.

## Runs (one box each, all in parallel)
| fleet RUN | label | model | domain config | artifact |
|---|---|---|---|---|
| geo2-qwen-investment | ta-tp-geo2-qwen-investment | Qwen/Qwen3-4B-Instruct-2507 | data/intertemporal/investment/investment_geometry.json | geometry/qwen3_4b_investment_v2.tar.gz |
| geo2-qwen-startup | ta-tp-geo2-qwen-startup | Qwen/Qwen3-4B-Instruct-2507 | data/intertemporal/startup/startup_geometry.json | geometry/qwen35_4b_startup_v2.tar.gz |
| geo2-llama-health | ta-tp-geo2-llama-health | meta-llama/Llama-3.1-8B-Instruct | data/intertemporal/health/health_geometry.json | geometry/llama31_8b_health_v2.tar.gz |
| geo2-gemma-climate | ta-tp-geo2-gemma-climate | google/gemma-2-9b-it | data/intertemporal/climate/climate_geometry.json | geometry/gemma2_9b_climate_v2.tar.gz |
| geo2-mistral-education | ta-tp-geo2-mistral-education | mistralai/Mistral-7B-Instruct-v0.3 | data/intertemporal/education/education_geometry.json | geometry/mistral7b_education_v2.tar.gz |

investment uses the full 4,588-prompt bank (no --max-samples). The other four use --max-samples 3000.
`qwen35_4b_startup` name is retained from v1; the content is Qwen3-4B-Instruct-2507 (see new_results.md naming note).

## Extraction
`generate_geometry_samples.py --config <cfg> --model <M> --resume out/geo/<run> --turn-only
 --components resid_post attn_out --dtype float16 [--max-samples 3000]`

## Mandatory per-box sanity gate (before extraction)
`scripts/intertemporal/verify_turn_positions.py` captures `hook_embed` for the first
valid sample and asserts, for EVERY named position i, that the cached embedding equals
`W_E[chosen_traj.token_ids[i]]` (allclose, atol 1e-4). It also decodes the chat_suffix
positions and requires that model's real turn tokens to be present.
A box that fails the gate produces no data and is reported as failed.

## Delivery
Code reaches the box by `git clone --branch exp/turn-geometry-llama-gemma` (public HTTPS),
so the box provably has 0ea7e91. Incremental file-level streaming to
`geometry/<artifact>/` during the run, then a single `geometry/<artifact>.tar.gz`.
Never overwrite the v1 archives.

## Fleet discipline
Shared account. Only `ta-tp-geo2-*` boxes are mine. `ta-tp-loc-*`, `ta-tp-risk-*`,
`ta-tp-steer-*` (46742505, 46742515, 46742541, 46742573, 46743982) belong to another
agent: never touch. MIN_CC=800, MIN_UP=500.
