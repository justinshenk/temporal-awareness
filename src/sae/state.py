"""Pipeline state for crash recovery and resumption."""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from .utils import FilepathConfig


class PipelineStage(str, Enum):
    INIT = "init"
    DATASET_GENERATED = "dataset_generated"
    INFERENCE_DONE = "inference_done"
    SAE_TRAINED = "sae_trained"
    EVALUATED = "evaluated"

    @classmethod
    def order(cls) -> list["PipelineStage"]:
        return [
            cls.INIT,
            cls.DATASET_GENERATED,
            cls.INFERENCE_DONE,
            cls.SAE_TRAINED,
            cls.EVALUATED,
        ]

    def __lt__(self, other):
        return self.order().index(self) < self.order().index(other)

    def __le__(self, other):
        return self.order().index(self) <= self.order().index(other)

    def __gt__(self, other):
        return self.order().index(self) > self.order().index(other)

    def __ge__(self, other):
        return self.order().index(self) >= self.order().index(other)


@dataclass
class PipelineConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    samples_per_iter: int = 2000
    max_iterations: int = 10
    seed: int = 42
    layers: list[int] = field(default_factory=lambda: [8, 14])
    num_time_horizons_bins: list[int] = field(default_factory=lambda: [10, 15, 20, 30])
    topk_values: list[int] = field(default_factory=lambda: [2, 3])
    max_new_tokens: int = 256
    max_epochs: int = 1
    patience: int = 1000
    batch_size: int = 128

    def compute_id(self) -> str:
        key = f"{self.model}_{self.layers}_{self.num_time_horizons_bins}_{self.topk_values}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineState:
    pipeline_id: str
    config: PipelineConfig
    config_id: str
    stage: PipelineStage
    iteration: int = 0
    filepath_cfg: FilepathConfig = field(default_factory=FilepathConfig)
    samples_path: Optional[str] = None
    inference_path: Optional[str] = None
    sae_results: list[dict] = field(default_factory=list)
    evaluation_results: dict = field(default_factory=dict)
    analysis_results: list[dict] = field(default_factory=list)
    iteration_history: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        state = {
            "pipeline_id": self.pipeline_id,
            "config": self.config.to_dict(),
            "config_id": self.config_id,
            "stage": self.stage.value,
            "iteration": self.iteration,
            "samples_path": self.samples_path,
            "inference_path": self.inference_path,
            "sae_results": self.sae_results,
            "evaluation_results": self.evaluation_results,
            "analysis_results": self.analysis_results,
            "iteration_history": self.iteration_history,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
        return state

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return self.__str__()  # optional: makes REPL output pretty too

    def save(self, path: str | Path):
        path = Path(path)
        self.last_updated = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        with open(path) as f:
            data = json.load(f)
        cfg = PipelineConfig.from_dict(data["config"])
        return cls(
            pipeline_id=data["pipeline_id"],
            config_id=cfg.compute_id(),
            config=cfg,
            stage=PipelineStage(data["stage"]),
            iteration=data.get("iteration", 0),
            samples_path=data.get("samples_path"),
            inference_path=data.get("inference_path"),
            sae_results=data.get("sae_results", []),
            evaluation_results=data.get("evaluation_results", {}),
            analysis_results=data.get("analysis_results", []),
            iteration_history=data.get("iteration_history", []),
            started_at=data.get("started_at", time.time()),
            last_updated=data.get("last_updated", time.time()),
        )

    @classmethod
    def create_new(cls, config: PipelineConfig) -> "PipelineState":
        return cls(
            pipeline_id=uuid.uuid4().hex,
            config=config,
            config_id=config.compute_id(),
            stage=PipelineStage.INIT,
            iteration=0,
        )

    def update_config(self, config: PipelineConfig):
        self.config = config


def find_state(pipeline_id: str) -> Optional[PipelineState]:
    """Load a specific pipeline state by its pipeline_id."""
    data_dir = FilepathConfig().data_dir
    state_file = data_dir / f"state_{pipeline_id}.json"
    if not state_file.exists():
        return None
    return PipelineState.load(state_file)


def find_latest_state() -> Optional[PipelineState]:
    data_dir = FilepathConfig().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    state_files = list(data_dir.glob("state_*.json"))
    if not state_files:
        return None
    state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return PipelineState.load(state_files[0])


def show_status(state: PipelineState | None):
    print("\n" + "=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)

    if state is None:
        print("No active pipeline found.")
        return

    print(f"\nPipeline ID: {state.pipeline_id}")
    print(f"Stage: {state.stage.value}")
    print(f"Iteration: {state.iteration} / {state.config.max_iterations}")

    print("\nConfig:")
    print(f"  Model: {state.config.model}")
    print(f"  Samples/iter: {state.config.samples_per_iter}")
    print(f"  Max iterations: {state.config.max_iterations}")
    print(f"  Layers: {state.config.layers}")
    print(f"  Clusters: {state.config.num_time_horizons_bins}")
    print(f"  TopK: {state.config.topk_values}")

    if state.evaluation_results and "best" in state.evaluation_results:
        best = state.evaluation_results["best"]
        print("\nBest Result:")
        print(f"  Config: {best.get('name', 'unknown')}")
        print(f"  Horizon NMI: {best.get('horizon_nmi', 0):.4f}")
        print(f"  Choice NMI: {best.get('choice_nmi', 0):.4f}")

    print()
