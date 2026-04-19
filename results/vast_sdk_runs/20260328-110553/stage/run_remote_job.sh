#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q transformers accelerate scikit-learn pandas
python3 -u run_mmraz_qwen_probe_steering_remote.py --config-path remote_job_config.json
