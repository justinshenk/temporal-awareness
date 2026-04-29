# vast.ai launcher

Generic driver for running GPU experiments on [vast.ai](https://vast.ai).
Provisions a single instance, rsyncs the current repo, installs a
PyTorch + ML stack matched to the detected GPU, runs your command, pulls
results back, and destroys the instance it created.

## Quick start

```bash
# 1. Put VASTAI_API_KEY in .env (at repo root)
echo 'VASTAI_API_KEY=<your-key>' >> .env

# 2. Register your local SSH key with the vast account (once):
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"

# 3. Run any command on a GPU:
scripts/vast/launch.sh 'python scripts/my_experiment/run.py'
```

Results land in `./vast_results/<label>_<instance_id>.tar.gz` on your local
machine.

## Safety invariant: only destroys the instance it created

The launcher is explicit about this because the account may be shared:

- `launch.sh` stores the instance ID only from its own `create instance` call
- The cleanup `trap` destroys **by explicit ID** (`vastai destroy instance $INSTANCE_ID`)
- If ID parsing fails, `INSTANCE_ID` is left empty and the trap **skips destroy** rather than running a dangerous default
- The launcher never calls `destroy all`, never iterates over `show instances`, never destroys anything it didn't itself create
- Set `LEAVE_INSTANCE=1` to skip the destroy (useful for debugging a failed run)

If you see other instances on the account disappear while running this
script, it is not from this script. Other places to look: another user on a
shared account, vast's own eviction of interruptible bids, or credit
exhaustion auto-stops.

## Configuration

All env vars are optional unless marked required.

| Var | Default | Meaning |
|---|---|---|
| `REMOTE_CMD` / `$1` | **required** | command executed on the remote |
| `VAST_API_KEY` | from `.env` | API key for vast.ai |
| `GPU_QUERY` | RTX 5090, ≥120 GB disk, reliability ≥0.97 | vast offer filter |
| `IMAGE` | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | Docker image for the instance |
| `MAX_DPH_USD` | `3.00` | refuse offers over this total $/hr |
| `BUDGET_USD` | `50` | informational — surfaces runtime headroom |
| `MIN_DISK_GB` | `120` | minimum disk requested |
| `EXCLUDES` | excludes `.git`, venvs, caches, `results` | rsync exclude args |
| `INSTALL_DEPS` | `1` | run `bootstrap.sh` before `REMOTE_CMD` |
| `BOOTSTRAP_EXTRA` | *(empty)* | extra pip packages to install |
| `PREDOWNLOAD_HF` | *(empty)* | space-separated HF model IDs to pre-fetch |
| `LABEL` | `vast-launch` | prefix for the pulled results tarball |
| `LEAVE_INSTANCE` | `0` | `1` = skip destroy, useful for debugging |

## GPU tier examples

```bash
# Single H100 SXM 80GB (for 30B+ models in bf16)
GPU_QUERY='gpu_name=H100_SXM num_gpus=1 disk_space>=200 reliability>=0.97 rentable=True' \
    scripts/vast/launch.sh 'python myexp.py'

# Single A100 40GB (cheap, fits 14B bf16)
GPU_QUERY='gpu_name=A100 num_gpus=1 disk_space>=120 reliability>=0.97 rentable=True' \
    scripts/vast/launch.sh 'python myexp.py'

# Cheapest 5090 32GB (great for ≤14B in bf16)
GPU_QUERY='gpu_name=RTX_5090 num_gpus=1 disk_space>=120 reliability>=0.97 rentable=True' \
    scripts/vast/launch.sh 'python myexp.py'
```

## Blackwell / RTX 5090 caveat

**RTX 5090 is Blackwell (`sm_120`)** and needs CUDA 12.8+ wheels. The
`pytorch/pytorch:2.4.0-cuda12.4` base image ships torch 2.4.0+cu121, which
has no kernels for `sm_120`. Any CUDA op fails with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`bootstrap.sh` detects this automatically (reads `nvidia-smi --query-gpu=compute_cap`)
and force-upgrades to `torch==2.7.1+cu128` on Blackwell. On Ampere/Hopper it
leaves the image's torch alone. Override with `SKIP_TORCH_UPGRADE=1` if you
need different behavior.

## Pre-downloading HF models

Model weights can be several GB and re-downloading them on every run is
wasteful. Pass them via `PREDOWNLOAD_HF`:

```bash
PREDOWNLOAD_HF='Qwen/Qwen2.5-7B meta-llama/Llama-3.2-1B' \
    scripts/vast/launch.sh 'python myexp.py'
```

Files are cached under `$HF_HOME=/workspace/hf_cache`. The cache is
instance-local (not persisted across instances), so pre-download runs on
every launch.

## How results get back

Your remote command should write outputs to `./results/` inside the repo.
`launch.sh` tars `results/` into `/workspace/results.tar.gz` at end-of-run
and pulls it back to `./vast_results/<label>_<instance_id>.tar.gz`.

If you need a different location, set `RESULTS_REMOTE` (path on the remote)
and/or `RESULTS_LOCAL_DIR` (directory on your machine).

## Troubleshooting

### "No matching offers"

Your `GPU_QUERY` is too restrictive or `MAX_DPH_USD` is too low. Widen the
filter or raise the price cap.

### "Could not parse new instance id from create output"

The vast API returned something unexpected. The launcher intentionally aborts
here — it will **not** try to destroy anything, because it doesn't know what
ID to target. Check `vastai show instances` manually and clean up if needed.

### SSH hangs / "Connection refused"

The instance is still loading (Docker image pull can take 2–5 minutes on a
fresh machine). `launch.sh` polls for ~10 minutes; if SSH never comes up,
something is wrong with the host — destroy and try another offer.

### "expected scalar type Float but found Half/BFloat16" in transformer_lens

Known issue with grouped-query attention models (Qwen 2.x, Llama 3.x) in
`transformer_lens==3.0.0b3` when loaded in half precision. Workarounds:

- Load in `torch.float32` (costs 2× VRAM)
- Downgrade to `transformer_lens==2.11.0` (may not support newest models)
- Patch `calculate_attention_scores` to cast to fp32 before the matmul

### "billing failed" mid-run

Your account credit ran out. Instances auto-stop (not destroy) when billing
fails — your work is usually preserved on disk and resumes if you add credit
before vast evicts the instance.

## Files

- `launch.sh` — local driver; search offers → create → rsync → run → pull → destroy
- `bootstrap.sh` — runs on the remote; detects GPU, installs torch + ML stack, pre-downloads models
- `README.md` — this file
