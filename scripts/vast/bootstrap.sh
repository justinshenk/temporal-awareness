#!/usr/bin/env bash
# Runs on the vast.ai instance. Installs a PyTorch + ML stack that matches the
# detected GPU capability, pre-downloads HF models if requested, and then
# returns control to launch.sh so it can exec the user command.
#
# This script is idempotent: re-running it is safe.
#
# Env:
#   BOOTSTRAP_EXTRA   extra pip packages to install (space-separated)
#   PREDOWNLOAD_HF    HF model IDs to pre-download (space-separated)
#   SKIP_TORCH_UPGRADE  if 1, do not touch torch (trust the base image)

set -euo pipefail

BOOTSTRAP_EXTRA="${BOOTSTRAP_EXTRA:-}"
PREDOWNLOAD_HF="${PREDOWNLOAD_HF:-}"
SKIP_TORCH_UPGRADE="${SKIP_TORCH_UPGRADE:-0}"

echo "[bootstrap] === GPU detection ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>/dev/null || {
    echo "[bootstrap] WARN: no nvidia-smi"
}

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
echo "[bootstrap] compute_capability=$CC"

# Torch wheel selection.
#
#   Blackwell (sm_120, RTX 5090, B100/B200) requires CUDA 12.8 wheels.
#   Hopper (sm_90, H100, H200) works with CUDA 12.1+ wheels.
#   Ampere (sm_80/86, A100/A6000/3090/4090) works with CUDA 11.8+ wheels.
#
# The pytorch/pytorch:2.4.0 base image ships torch 2.4.0+cu121 which has NO
# kernels for sm_120. On Blackwell this fails with:
#   RuntimeError: CUDA error: no kernel image is available for execution on the device
# Fix: force-upgrade torch to 2.7.1 from the cu128 index.
NEED_TORCH_UPGRADE=0
case "$CC" in
    12.0|12.*)
        NEED_TORCH_UPGRADE=1
        TORCH_VERSION="2.7.1"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        echo "[bootstrap] Blackwell GPU detected (sm_$CC) -> torch==$TORCH_VERSION+cu128"
        ;;
    *)
        echo "[bootstrap] compute_cap=$CC -> using base image torch (no upgrade)"
        ;;
esac

if [ "$SKIP_TORCH_UPGRADE" = "1" ]; then
    NEED_TORCH_UPGRADE=0
fi

echo "[bootstrap] === installing base deps ==="
if [ "$NEED_TORCH_UPGRADE" = "1" ]; then
    pip uninstall -y torch torchvision 2>&1 | tail -3 || true
    pip install --no-cache-dir --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION" "torchvision" 2>&1 | tail -5
fi

# Pin transformers to a version compatible with transformer_lens 3.x.
# 3.0.0b3 requires transformers>=4.56. If your experiment uses a different
# transformer_lens version, override by setting BOOTSTRAP_EXTRA.
pip install --no-cache-dir \
    "transformers==4.56.2" \
    "transformer_lens==3.0.0b3" \
    "scikit-learn" "scipy" "matplotlib" "numpy" "pandas" "tqdm" \
    "accelerate" "einops" \
    2>&1 | tail -10

if [ -n "$BOOTSTRAP_EXTRA" ]; then
    echo "[bootstrap] === extra deps: $BOOTSTRAP_EXTRA ==="
    # shellcheck disable=SC2086
    pip install --no-cache-dir $BOOTSTRAP_EXTRA 2>&1 | tail -5
fi

echo "[bootstrap] === sanity check ==="
python - <<'PY'
import torch
print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"compute_cap={torch.cuda.get_device_capability(0)}")
    try:
        x = torch.zeros(1, device='cuda') + 1
        print(f"gpu_op_ok={x.item()}")
    except Exception as e:
        print(f"GPU_OP_FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
try:
    import transformers, transformer_lens  # noqa
    print(f"transformers={transformers.__version__}")
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
    raise SystemExit(1)
PY

if [ -n "$PREDOWNLOAD_HF" ]; then
    echo "[bootstrap] === predownloading HF models: $PREDOWNLOAD_HF ==="
    for m in $PREDOWNLOAD_HF; do
        echo "[bootstrap] fetching $m"
        python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('$m')
AutoModelForCausalLM.from_pretrained('$m', torch_dtype='auto')
" 2>&1 | tail -3
    done
fi

echo "[bootstrap] === done ==="
