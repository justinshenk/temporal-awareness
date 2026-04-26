"""Hugging Face upload helpers for activation caches."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

try:
    from .eap_ig_qanda_common import resolve_hf_repo_id
except ImportError:
    from eap_ig_qanda_common import resolve_hf_repo_id


def upload_cache_folder(
    *,
    save_to_hf: bool,
    local_dir: Path,
    path_in_repo: str,
    hf_repo_id: str,
    hf_repo_type: str,
) -> None:
    """Upload the complete cache directory in a single Hub commit."""
    if not save_to_hf:
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for Hub uploads.")

    hf_api = HfApi(token=hf_token)
    resolved_repo_id = resolve_hf_repo_id(hf_api, hf_repo_id)
    hf_api.create_repo(repo_id=resolved_repo_id, repo_type=hf_repo_type, exist_ok=True)

    print(
        f"[HF upload] Uploading folder={local_dir} "
        f"repo_id={resolved_repo_id} repo_path={path_in_repo} "
        f"repo_type={hf_repo_type}",
        flush=True,
    )
    hf_api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=resolved_repo_id,
        repo_type=hf_repo_type,
        commit_message=f"Upload activation cache folder {path_in_repo}",
    )
    print("[HF upload] Completed folder upload", flush=True)
