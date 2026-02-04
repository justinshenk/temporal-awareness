"""Utilities: device management, memory, and path management."""

import gc
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


ROOT = Path(__file__).parent.parent.resolve()

PROJECT_ROOT = ROOT.parent.parent  # temporal-awareness/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCENARIOS_DIR = ROOT / "configs" / "scenarios"


def get_default_output_dir() -> Path:
    return ROOT / "run_data"


def get_default_test_dir() -> Path:
    return ROOT / "test_iter"


def get_default_special_dir() -> Path:
    return ROOT / "special_iter"


@dataclass
class FilepathConfig:
    output_dir: Path = field(default_factory=get_default_output_dir)

    @property
    def data_dir(self):
        return self.output_dir / "data"

    @property
    def sae_dir(self):
        return self.output_dir / "sae"

    @property
    def analysis_dir(self):
        return self.output_dir / "analysis"

    @property
    def tensorboard_dir(self):
        return self.output_dir / "tb_logs"

    def get_all_dirs(self) -> list[Path]:
        return [
            self.data_dir,
            self.sae_dir,
            self.analysis_dir,
            self.tensorboard_dir,
        ]


def reset_and_get_test_filepath_cfg() -> FilepathConfig:
    test_dir = get_default_test_dir()
    output_dir = get_default_output_dir()

    # Wipe test_iter/ and copy run_data/ into it (if it exists)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    if output_dir.exists():
        shutil.copytree(output_dir, test_dir)
        print("Copied run_data/ -> test_iter/")
    else:
        test_dir.mkdir(parents=True)

    return FilepathConfig(output_dir=test_dir)


def reset_and_get_special_filepath_cfg() -> FilepathConfig:
    special_dir = get_default_special_dir()
    output_dir = get_default_output_dir()

    # Wipe special_iter/ and copy run_data/ into it (if it exists)
    if special_dir.exists():
        shutil.rmtree(special_dir)

    if output_dir.exists():
        shutil.copytree(output_dir, special_dir)
        print("Copied run_data/ -> special_iter/")
    else:
        special_dir.mkdir(parents=True)

    return FilepathConfig(output_dir=special_dir)


# ────────────────────────────────────────────────────────────


def ensure_dirs(file_cfg: FilepathConfig | None = None):
    """Create all runtime output directories."""
    if not file_cfg:
        file_cfg = FilepathConfig()

    for d in file_cfg.get_all_dirs():
        d.mkdir(parents=True, exist_ok=True)


# ── Device / memory ────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
