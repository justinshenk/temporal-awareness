"""
Upload support for the Q&A EAP-IG workflow.
"""

import os
import queue
import threading
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi

try:
    from .eap_ig_qanda_common import resolve_hf_repo_id
except ImportError:
    from eap_ig_qanda_common import resolve_hf_repo_id


def maybe_start_upload_worker(
    save_to_hf: bool,
    hf_repo_id: str,
    hf_repo_type: str,
) -> tuple[
    queue.Queue[tuple[Path, str] | None] | None,
    threading.Thread | None,
    Callable[[Path, str], None],
]:
    """Start the background Hub uploader when enabled."""
    if not save_to_hf:
        return None, None, lambda _local_file, _path_in_repo: None

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for Hub uploads.")

    hf_api = HfApi(token=hf_token)
    resolved_repo_id = resolve_hf_repo_id(hf_api, hf_repo_id)
    hf_api.create_repo(repo_id=resolved_repo_id, repo_type=hf_repo_type, exist_ok=True)
    print(
        f"[HF upload] Using repo_id={resolved_repo_id} repo_type={hf_repo_type}",
        flush=True,
    )

    upload_queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()

    def _upload_worker() -> None:
        while True:
            item = upload_queue.get()
            if item is None:
                upload_queue.task_done()
                break
            local_file, path_in_repo = item
            file_size = local_file.stat().st_size
            print(
                f"[HF upload] Starting file={local_file.name} "
                f"local_path={local_file} repo_path={path_in_repo} "
                f"size_gb={file_size / (1024**3):.2f}",
                flush=True,
            )
            hf_api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=path_in_repo,
                repo_id=resolved_repo_id,
                repo_type=hf_repo_type,
                commit_message=f"Upload {path_in_repo}",
            )
            print(
                f"[HF upload] Completed file={local_file.name}",
                flush=True,
            )
            upload_queue.task_done()

    upload_thread = threading.Thread(target=_upload_worker, daemon=True)
    upload_thread.start()

    def _enqueue_upload(local_file: Path, path_in_repo: str) -> None:
        upload_queue.put((local_file, path_in_repo))

    return upload_queue, upload_thread, _enqueue_upload
