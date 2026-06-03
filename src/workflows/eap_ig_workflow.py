"""Single-command workflow for the Q&A EAP-IG analysis pipeline."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import pearsonr

StageName: TypeAlias = Literal[
    "eap-ig",
    "top-components",
    "node-selection",
    "activation-patch",
    "visualize",
]
MergedArrayGroup: TypeAlias = dict[str, np.ndarray]
PromptFormatResults: TypeAlias = dict[str, MergedArrayGroup]
CaseResults: TypeAlias = dict[str, PromptFormatResults]
TopComponent: TypeAlias = tuple[str, int, float]
SelectedNode: TypeAlias = tuple[str, int]

REPO_ROOT = Path(__file__).resolve().parents[2]
EAP_IG_INPUTS_SCRIPT = (
    REPO_ROOT / "scripts" / "experiments" / "eap_ig" / "eap_ig_inputs_QandA.py"
)
ACTIVATION_PATCH_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "eap_ig"
    / "classification_short_to_long_activation_patch.py"
)

DEFAULT_EAP_IG_CONFIG_DIR = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "eap_ig"
    / "config"
    / "final_configs"
    / "Q_A"
)
DEFAULT_CLASSIFICATION_CONFIG = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "eap_ig"
    / "config"
    / "final_configs"
    / "classification"
    / "classification_explicit_GC.yaml"
)
DEFAULT_EAP_IG_RESULTS_DIR = REPO_ROOT / "results" / "node_wise"
DEFAULT_TOP_COMPONENTS_DIR = REPO_ROOT / "data" / "top_n_nodes"
DEFAULT_SELECTED_NODES_DIR = REPO_ROOT / "data" / "selected_nodes"
DEFAULT_COMPLETENESS_FIGURES_DIR = (
    REPO_ROOT / "results" / "figures" / "eap_ig_completeness"
)
DEFAULT_CTP_FIGURES_DIR = (
    REPO_ROOT / "results" / "figures" / "cross_task_performance"
)

DEFAULT_NODE_CLASSES = (
    "p_generic",
    "n_generic",
    "p_common_ST",
    "p_common_LT",
    "sym_ST_p",
    "sym_LT_p",
    "n_common_LT",
    "n_common_ST",
)
DEFAULT_CASES = ("explicit", "implicit")
STAGE_ORDER: tuple[StageName, ...] = (
    "eap-ig",
    "top-components",
    "node-selection",
    "activation-patch",
    "visualize",
)
STAGE_TO_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}
COMPLETENESS_PLOT_ROWS = 3
COMPONENT_ARRAY_NAMES = frozenset({"z", "mlp_hidden"})
LOGIT_ARRAY_NAMES = frozenset({"clean_logits", "corrupted_logits"})
NPZ_FILENAME_PATTERN = re.compile(
    r"^(?P<stem>.+)_(?P<option_order>short_first|long_first)_(?P<metric>logit_[AB])_batch_\d+\.npz$"
)
LOGIT_INDEX_BY_LABEL = {"A": 0, "B": 1}


@dataclass(frozen=True)
class WorkflowConfig:
    """Resolved runtime configuration for the Q&A workflow."""

    eap_ig_config_dir: Path
    classification_config: Path
    eap_ig_results_dir: Path
    top_components_dir: Path
    selected_nodes_dir: Path
    completeness_figures_dir: Path
    ctp_figures_dir: Path
    top_n: int
    selection_limit: int
    node_classes: tuple[str, ...]
    activation_output_filename: str | None
    min_logit_diff: float
    num_random_sources: int
    save_to_hf: bool
    start_at: StageName
    stop_after: StageName

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WorkflowConfig":
        """Build a resolved workflow config from parsed CLI args."""
        return cls(
            eap_ig_config_dir=resolve_repo_path(args.eap_ig_config_dir),
            classification_config=resolve_repo_path(args.classification_config),
            eap_ig_results_dir=resolve_repo_path(args.eap_ig_results_dir),
            top_components_dir=resolve_repo_path(args.top_components_dir),
            selected_nodes_dir=resolve_repo_path(args.selected_nodes_dir),
            completeness_figures_dir=resolve_repo_path(
                args.completeness_figures_dir
            ),
            ctp_figures_dir=resolve_repo_path(args.ctp_figures_dir),
            top_n=args.top_n,
            selection_limit=args.selection_limit,
            node_classes=tuple(args.node_class),
            activation_output_filename=args.activation_output_filename,
            min_logit_diff=args.min_logit_diff,
            num_random_sources=args.num_random_sources,
            save_to_hf=args.save_to_hf,
            start_at=args.start_at,
            stop_after=args.stop_after,
        )

    @property
    def selected_nodes_path(self) -> Path:
        """Return the selected-node artifact path for this workflow run."""
        return resolve_selected_nodes_path(self.selected_nodes_dir, self.top_n)

    @property
    def activation_patch_output_path(self) -> Path:
        """Return the expected activation-patching artifact path."""
        return expected_activation_patch_path(
            self.classification_config,
            node_classes=self.node_classes,
            top_n=self.top_n,
            output_filename=self.activation_output_filename,
        )

    def includes_stage(self, stage: StageName) -> bool:
        """Return whether the requested stage lies inside the stage window."""
        return is_stage_requested(stage, self.start_at, self.stop_after)


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repository-relative paths from CLI args and config files."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_pickle(path: Path, payload: Any) -> None:
    """Write a pickle artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_python_script(script_path: Path, args: Sequence[str]) -> None:
    """Run a repository script with the current Python interpreter."""
    command = [sys.executable, str(script_path), *args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def discover_eap_ig_configs(config_dir: Path) -> list[Path]:
    """Return the Q&A YAML configs in stable filename order."""
    configs = sorted(config_dir.glob("*.yaml"), key=lambda path: path.name.lower())
    if not configs:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")
    return configs


def validate_stage_window(start_at: StageName, stop_after: StageName) -> None:
    """Validate that the requested stage window is well ordered."""
    if STAGE_TO_INDEX[start_at] > STAGE_TO_INDEX[stop_after]:
        raise ValueError(
            f"start-at={start_at!r} must come before stop-after={stop_after!r}"
        )


def is_stage_requested(
    stage: StageName,
    start_at: StageName,
    stop_after: StageName,
) -> bool:
    """Return whether ``stage`` lies inside the requested stage window."""
    start_index = STAGE_TO_INDEX[start_at]
    stop_index = STAGE_TO_INDEX[stop_after]
    stage_index = STAGE_TO_INDEX[stage]
    return start_index <= stage_index <= stop_index


def run_eap_ig_stage(
    *,
    config_dir: Path,
    save_to_hf: bool,
) -> list[Path]:
    """Run the Q&A EAP-IG script once per config file."""
    config_paths = discover_eap_ig_configs(config_dir)
    for config_path in config_paths:
        args = ["--config", str(config_path)]
        args.append("--save-to-hf" if save_to_hf else "--no-save-to-hf")
        run_python_script(EAP_IG_INPUTS_SCRIPT, args)
    return config_paths


def extract_merged_array_name(array_key: str) -> str | None:
    """Map a saved NPZ key to the merged output key consumed downstream."""
    parts = array_key.split("__")
    if parts[-1] in LOGIT_ARRAY_NAMES:
        return parts[-1]
    if len(parts) >= 3 and parts[-2] in COMPONENT_ARRAY_NAMES:
        return f"{parts[-2]}/{parts[-1]}"
    return None


def process_eap_ig_result_folder(folder: Path) -> PromptFormatResults:
    """Merge batched NPZ outputs for a single prompt-format folder."""
    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(folder.glob("*.npz")):
        match = NPZ_FILENAME_PATTERN.match(file_path.name)
        if not match:
            continue
        group_name = f"{match.group('option_order')}_{match.group('metric')}"
        grouped_files[group_name].append(file_path)

    merged_results: PromptFormatResults = {}
    for group_name, file_paths in grouped_files.items():
        merged_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for file_path in file_paths:
            with np.load(file_path, allow_pickle=False) as data:
                for array_key in data.files:
                    merged_name = extract_merged_array_name(array_key)
                    if merged_name is None:
                        # Metadata is useful at the raw file level, but the
                        # downstream selection code only consumes merged logits
                        # and component tensors.
                        continue
                    merged_arrays[merged_name].append(data[array_key])

        if merged_arrays:
            merged_results[group_name] = {
                merged_name: np.concatenate(arrays, axis=0)
                for merged_name, arrays in merged_arrays.items()
            }

    return merged_results


def load_case_results(
    results_dir: Path,
    case_name: str,
) -> CaseResults:
    """Load all Q&A prompt-format folders for a case like ``explicit``."""
    case_dir = results_dir / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")

    case_results: CaseResults = {}
    for folder in sorted(case_dir.iterdir(), key=lambda path: path.name.lower()):
        if not folder.is_dir() or folder.name == "short":
            continue
        folder_results = process_eap_ig_result_folder(folder)
        if folder_results:
            case_results[folder.name] = folder_results

    if not case_results:
        raise FileNotFoundError(f"No Q&A result folders found in {case_dir}")

    return case_results


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation without crashing on constant arrays."""
    try:
        result = pearsonr(x, y)
        return float(result.correlation), float(result.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def compute_completeness_arrays(
    group: MergedArrayGroup,
    logit_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the notebook-style predicted and target completeness arrays."""
    target = group["clean_logits"][:, logit_index] - group["corrupted_logits"][
        :, logit_index
    ]
    component_predictions = [
        values.sum(axis=-1)
        for key, values in group.items()
        if key not in LOGIT_ARRAY_NAMES
    ]
    predicted = np.stack(component_predictions, axis=-1).sum(axis=-1)
    return predicted, target


def plot_case_completeness(
    case_name: str,
    case_results: CaseResults,
    output_path: Path,
    *,
    logit_label: str = "B",
) -> Path:
    """Save the completeness scatter plots produced in the notebook workflow."""
    variants = list(case_results.keys())
    columns = max(1, math.ceil(len(variants) / COMPLETENESS_PLOT_ROWS))
    fig, axes = plt.subplots(
        COMPLETENESS_PLOT_ROWS,
        columns,
        figsize=(5 * columns, 4.5 * COMPLETENESS_PLOT_ROWS),
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, variant_name in zip(axes, variants, strict=False):
        group = case_results[variant_name][f"short_first_logit_{logit_label}"]
        predicted, target = compute_completeness_arrays(
            group,
            LOGIT_INDEX_BY_LABEL[logit_label],
        )
        correlation, p_value = safe_pearsonr(predicted, target)
        axis.scatter(predicted, target, s=5)
        axis.set_title(
            f"{variant_name}_r:{correlation:.2f}_p:{p_value:.4f}",
            fontsize=10,
        )
        axis.set_xlabel("Predicted sum")
        axis.set_ylabel("Target logit delta")

    for axis in axes[len(variants) :]:
        axis.axis("off")

    fig.suptitle(f"{case_name}_{logit_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def get_denom(group: MergedArrayGroup, logit_index: int) -> np.ndarray:
    """Match the notebook's normalization denominator."""
    denom = np.abs(
        group["clean_logits"][:, logit_index]
        - group["corrupted_logits"][:, logit_index]
    )
    return np.where(denom < 0.5, np.nan, denom)


def get_attr_normalized(
    group: MergedArrayGroup,
    logit_index: int,
) -> dict[str, np.ndarray]:
    """Normalize component attribution arrays by the logit delta magnitude."""
    denom = get_denom(group, logit_index).reshape(-1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            key: values / denom
            for key, values in group.items()
            if key not in LOGIT_ARRAY_NAMES
        }


def aggregate_case_scores(
    case_results: CaseResults,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate attribution scores exactly as in ``datasetwise_top_n.ipynb``."""
    aggregated: dict[str, dict[str, np.ndarray]] = {}

    for horizon in ("ST", "LT"):
        net_scores: dict[str, list[np.ndarray]] = defaultdict(list)

        for variant_results in case_results.values():
            if horizon == "ST":
                # The original notebook defines short-term salience by averaging
                # long-first answers scored on logit_B and short-first answers
                # scored on logit_A.
                long_first = get_attr_normalized(
                    variant_results["long_first_logit_B"],
                    logit_index=1,
                )
                short_first = get_attr_normalized(
                    variant_results["short_first_logit_A"],
                    logit_index=0,
                )
            else:
                long_first = get_attr_normalized(
                    variant_results["long_first_logit_A"],
                    logit_index=0,
                )
                short_first = get_attr_normalized(
                    variant_results["short_first_logit_B"],
                    logit_index=1,
                )

            for component_name in long_first:
                net_scores[component_name].append(
                    (long_first[component_name] + short_first[component_name]) / 2
                )

        aggregated[horizon] = {
            component_name: np.nanmean(
                np.stack(component_values, axis=-1),
                axis=(0, -1),
            )
            for component_name, component_values in net_scores.items()
        }

    return aggregated


def topk_components(
    component_dict: dict[str, np.ndarray],
    k: int,
) -> list[TopComponent]:
    """Return the top-k components by absolute attribution magnitude."""
    if not component_dict or k <= 0:
        return []

    component_names: list[str] = []
    flattened_arrays: list[np.ndarray] = []
    for component_name, values in component_dict.items():
        component_names.append(component_name)
        flattened_arrays.append(np.asarray(values).ravel())

    concatenated = np.concatenate(flattened_arrays)
    component_ids = np.concatenate(
        [
            np.full(len(component_values), index, dtype=np.int32)
            for index, component_values in enumerate(flattened_arrays)
        ]
    )
    local_indices = np.concatenate(
        [
            np.arange(len(component_values), dtype=np.int32)
            for component_values in flattened_arrays
        ]
    )

    k = min(k, len(concatenated))
    absolute_values = np.abs(concatenated)
    topk_indices = np.argpartition(absolute_values, -k)[-k:]
    topk_indices = topk_indices[np.argsort(absolute_values[topk_indices])[::-1]]

    return [
        (
            component_names[component_ids[index]],
            int(local_indices[index]),
            float(concatenated[index]),
        )
        for index in topk_indices
    ]


def serialize_top_components(
    top_components: dict[str, list[TopComponent]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert top-component tuples to JSON-friendly records."""
    return {
        horizon: [
            {"component": component, "index": index, "score": score}
            for component, index, score in entries
        ]
        for horizon, entries in top_components.items()
    }


def expected_top_components_path(
    top_components_dir: Path,
    top_n: int,
    case_name: str,
) -> Path:
    """Return the expected pickle output path for a case."""
    return top_components_dir / f"top_{top_n}_components_{case_name}.pkl"


def build_top_components_for_case(
    *,
    case_name: str,
    results_dir: Path,
    top_components_dir: Path,
    completeness_figures_dir: Path,
    top_n: int,
) -> tuple[Path, Path]:
    """Compute, plot, and save top components for a single case."""
    case_results = load_case_results(results_dir, case_name)
    figure_path = plot_case_completeness(
        case_name,
        case_results,
        completeness_figures_dir / f"{case_name}_completeness.png",
    )
    aggregated_scores = aggregate_case_scores(case_results)
    top_components = {
        horizon: topk_components(aggregated_scores[horizon], top_n)
        for horizon in aggregated_scores
    }

    pickle_path = expected_top_components_path(top_components_dir, top_n, case_name)
    write_pickle(pickle_path, top_components)
    write_json(
        pickle_path.with_suffix(".json"),
        serialize_top_components(top_components),
    )

    return pickle_path, figure_path


def build_top_components(
    *,
    results_dir: Path,
    top_components_dir: Path,
    completeness_figures_dir: Path,
    top_n: int,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Build top-component artifacts for both explicit and implicit cases."""
    pickles: dict[str, Path] = {}
    figures: dict[str, Path] = {}
    for case_name in DEFAULT_CASES:
        pickle_path, figure_path = build_top_components_for_case(
            case_name=case_name,
            results_dir=results_dir,
            top_components_dir=top_components_dir,
            completeness_figures_dir=completeness_figures_dir,
            top_n=top_n,
        )
        pickles[case_name] = pickle_path
        figures[case_name] = figure_path
    return pickles, figures


def load_top_components_for_selection(
    top_components_dir: Path,
    top_n: int,
) -> dict[str, dict[str, list[TopComponent]]]:
    """Load explicit and implicit top-component pickles for node selection."""
    loaded: dict[str, dict[str, list[TopComponent]]] = {}
    for case_name in DEFAULT_CASES:
        path = expected_top_components_path(top_components_dir, top_n, case_name)
        with path.open("rb") as f:
            loaded[case_name] = pickle.load(f)
    return loaded


def select_signed_nodes(
    components: list[TopComponent],
    selection_limit: int,
    *,
    positive: bool,
) -> set[SelectedNode]:
    """Take the notebook's first-``selection_limit`` components filtered by sign."""
    score_predicate = (
        (lambda score: score > 0) if positive else (lambda score: score < 0)
    )
    return {
        (component, index)
        for component, index, score in components[:selection_limit]
        if score_predicate(score)
    }


def build_selected_node_groups(
    top_components_dir: Path,
    *,
    top_n: int,
    selection_limit: int,
) -> dict[str, set[SelectedNode]]:
    """Reproduce the node grouping from ``node_selection.ipynb``."""
    top_components = load_top_components_for_selection(top_components_dir, top_n)

    def selected(case_name: str, horizon: str, *, positive: bool) -> set[SelectedNode]:
        return select_signed_nodes(
            top_components[case_name][horizon],
            selection_limit,
            positive=positive,
        )

    p_nodes_exp_lt = selected("explicit", "LT", positive=True)
    p_nodes_exp_st = selected("explicit", "ST", positive=True)
    p_nodes_imp_lt = selected("implicit", "LT", positive=True)
    p_nodes_imp_st = selected("implicit", "ST", positive=True)

    n_nodes_exp_lt = selected("explicit", "LT", positive=False)
    n_nodes_exp_st = selected("explicit", "ST", positive=False)
    n_nodes_imp_lt = selected("implicit", "LT", positive=False)
    n_nodes_imp_st = selected("implicit", "ST", positive=False)

    p_generic = p_nodes_exp_lt & p_nodes_exp_st & p_nodes_imp_lt & p_nodes_imp_st
    n_generic = n_nodes_exp_lt & n_nodes_exp_st & n_nodes_imp_lt & n_nodes_imp_st

    p_nodes_exp_lt = p_nodes_exp_lt - p_generic
    p_nodes_imp_lt = p_nodes_imp_lt - p_generic
    n_nodes_exp_lt = n_nodes_exp_lt - n_generic
    n_nodes_imp_lt = n_nodes_imp_lt - n_generic
    p_nodes_exp_st = p_nodes_exp_st - p_generic
    p_nodes_imp_st = p_nodes_imp_st - p_generic
    n_nodes_exp_st = n_nodes_exp_st - n_generic
    n_nodes_imp_st = n_nodes_imp_st - n_generic

    p_common_lt = p_nodes_exp_lt & p_nodes_imp_lt
    n_common_lt = n_nodes_exp_lt & n_nodes_imp_lt
    p_common_st = p_nodes_exp_st & p_nodes_imp_st
    n_common_st = n_nodes_exp_st & n_nodes_imp_st

    sym_lt_p = p_common_lt & n_common_st
    sym_st_p = p_common_st & n_common_lt

    p_common_lt = p_common_lt - sym_lt_p
    n_common_st = n_common_st - sym_lt_p
    p_common_st = p_common_st - sym_st_p

    # Preserve the notebook's final assignment exactly for compatibility with
    # previously generated selected-node artifacts, even though it looks
    # asymmetric relative to the earlier set construction.
    n_common_st = n_common_lt - sym_st_p

    return {
        "p_generic": p_generic,
        "n_generic": n_generic,
        "sym_LT_p": sym_lt_p,
        "sym_ST_p": sym_st_p,
        "p_common_LT": p_common_lt,
        "p_common_ST": p_common_st,
        "n_common_LT": n_common_lt,
        "n_common_ST": n_common_st,
    }


def serialize_selected_node_groups(
    selected_nodes: dict[str, set[SelectedNode]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert selected-node sets to JSON-friendly records."""
    return {
        node_class: [
            {"component": component, "index": index}
            for component, index in sorted(nodes)
        ]
        for node_class, nodes in selected_nodes.items()
    }


def expected_selected_nodes_path(selected_nodes_dir: Path, top_n: int) -> Path:
    """Return the expected selected-nodes pickle path."""
    return selected_nodes_dir / f"final_{top_n}_eap_ig.pkl"


def legacy_selected_nodes_path(selected_nodes_dir: Path, top_n: int) -> Path:
    """Return the legacy selected-nodes pickle path."""
    return selected_nodes_dir / f"final_{top_n}_QnA.pkl"


def resolve_selected_nodes_path(selected_nodes_dir: Path, top_n: int) -> Path:
    """Return the preferred selected-node artifact, falling back to the legacy name."""
    preferred_path = expected_selected_nodes_path(selected_nodes_dir, top_n)
    if preferred_path.exists():
        return preferred_path

    legacy_path = legacy_selected_nodes_path(selected_nodes_dir, top_n)
    if legacy_path.exists():
        return legacy_path

    return preferred_path


def write_selected_nodes(
    *,
    top_components_dir: Path,
    selected_nodes_dir: Path,
    top_n: int,
    selection_limit: int,
) -> tuple[Path, Path]:
    """Save the notebook-equivalent node-selection artifacts."""
    selected_nodes = build_selected_node_groups(
        top_components_dir,
        top_n=top_n,
        selection_limit=selection_limit,
    )

    pickle_path = expected_selected_nodes_path(selected_nodes_dir, top_n)
    json_path = pickle_path.with_suffix(".json")
    write_pickle(pickle_path, selected_nodes)
    write_json(json_path, serialize_selected_node_groups(selected_nodes))
    return pickle_path, json_path


def derive_activation_output_filename(
    classification_config: Path,
    *,
    top_n: int,
    output_filename: str | None,
) -> str:
    """Derive the activation-patching output stem."""
    if output_filename is not None:
        return output_filename
    config = load_yaml(classification_config)
    config_output = config.get("output", {}).get(
        "filename",
        Path(config["input"]["data_file"]).stem,
    )
    return f"{top_n}_{config_output}"


def expected_activation_patch_path(
    classification_config: Path,
    *,
    node_classes: Sequence[str],
    top_n: int,
    output_filename: str | None,
) -> Path:
    """Return the activation-patching JSON path produced by the script."""
    config = load_yaml(classification_config)
    save_loc = resolve_repo_path(config["paths"]["save_loc"])
    derived_output_filename = derive_activation_output_filename(
        classification_config,
        top_n=top_n,
        output_filename=output_filename,
    )
    return save_loc / (
        f"{sanitize_filename(derived_output_filename)}__"
        f"{sanitize_filename('__'.join(node_classes))}__"
        "bidirectional_activation_patch.json"
    )


def run_activation_patch_stage(
    *,
    classification_config: Path,
    selected_nodes_path: Path,
    node_classes: Sequence[str],
    top_n: int,
    output_filename: str | None,
    min_logit_diff: float,
    num_random_sources: int,
) -> Path:
    """Run the classification activation-patching stage."""
    derived_output_filename = derive_activation_output_filename(
        classification_config,
        top_n=top_n,
        output_filename=output_filename,
    )
    args = [
        "--config",
        str(classification_config),
        "--node-class",
        *node_classes,
        "--nodes-path",
        str(selected_nodes_path),
        "--output-filename",
        derived_output_filename,
        "--min-logit-diff",
        str(min_logit_diff),
        "--num-random-sources",
        str(num_random_sources),
    ]
    run_python_script(ACTIVATION_PATCH_SCRIPT, args)
    return expected_activation_patch_path(
        classification_config,
        node_classes=node_classes,
        top_n=top_n,
        output_filename=output_filename,
    )


def infer_patch_directions(
    payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Infer the short-to-long and long-to-short patch direction keys."""
    patched = payload.get("patched_responses", {})
    metadata = payload.get("metadata", {})
    option_keys = metadata.get("option_keys", [])
    if len(option_keys) >= 2:
        # Prefer metadata because serialized dict key order is incidental,
        # whereas the label ordering is semantically meaningful.
        return (
            f"{option_keys[0]}_to_{option_keys[1]}",
            f"{option_keys[1]}_to_{option_keys[0]}",
        )

    keys = list(patched.keys())
    if len(keys) >= 2:
        return keys[0], keys[1]
    if len(keys) == 1:
        return keys[0], None
    return None, None


def plot_cross_task_performance(
    result_path: Path,
    output_dir: Path,
) -> Path:
    """Save the scatter plot produced by ``visualize_ctp.ipynb``."""
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    base_records = [record for record in payload["base_responses"] if record["kept"]]
    short_values = [record["short_logit"] for record in base_records]
    long_values = [record["long_logit"] for record in base_records]

    short_to_long_key, long_to_short_key = infer_patch_directions(payload)
    patched = payload.get("patched_responses", {})

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(short_values, long_values, s=6, label="base")

    if short_to_long_key and short_to_long_key in patched:
        stl_records = patched[short_to_long_key]
        axis.scatter(
            [record["short_logit"] for record in stl_records],
            [record["long_logit"] for record in stl_records],
            s=6,
            alpha=0.4,
            label=short_to_long_key,
        )

    if long_to_short_key and long_to_short_key in patched:
        lts_records = patched[long_to_short_key]
        axis.scatter(
            [record["short_logit"] for record in lts_records],
            [record["long_logit"] for record in lts_records],
            s=6,
            alpha=0.4,
            label=long_to_short_key,
        )

    axis.set_xlabel("short")
    axis.set_ylabel("long")
    axis.legend()
    axis.set_title(result_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result_path.stem}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the workflow entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the Q&A EAP-IG workflow end-to-end from the command line.",
    )

    path_group = parser.add_argument_group("paths")
    path_group.add_argument(
        "--eap-ig-config-dir",
        type=Path,
        default=DEFAULT_EAP_IG_CONFIG_DIR,
        help="Directory containing the EAP-IG Q&A YAML configs.",
    )
    path_group.add_argument(
        "--classification-config",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CONFIG,
        help="Classification config used for the activation-patching stage.",
    )
    path_group.add_argument(
        "--eap-ig-results-dir",
        type=Path,
        default=DEFAULT_EAP_IG_RESULTS_DIR,
        help="Root directory holding the EAP-IG NPZ outputs grouped by case.",
    )
    path_group.add_argument(
        "--top-components-dir",
        type=Path,
        default=DEFAULT_TOP_COMPONENTS_DIR,
        help="Directory where top-component artifacts should be written.",
    )
    path_group.add_argument(
        "--selected-nodes-dir",
        type=Path,
        default=DEFAULT_SELECTED_NODES_DIR,
        help="Directory where selected-node artifacts should be written.",
    )
    path_group.add_argument(
        "--completeness-figures-dir",
        type=Path,
        default=DEFAULT_COMPLETENESS_FIGURES_DIR,
        help="Directory where completeness figures should be written.",
    )
    path_group.add_argument(
        "--ctp-figures-dir",
        type=Path,
        default=DEFAULT_CTP_FIGURES_DIR,
        help="Directory where cross-task performance figures should be written.",
    )

    selection_group = parser.add_argument_group("selection")
    selection_group.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Number of top components to keep per horizon before node selection.",
    )
    selection_group.add_argument(
        "--selection-limit",
        type=int,
        default=300,
        help="How many top components per horizon/sign to consider during node selection.",
    )
    selection_group.add_argument(
        "--node-class",
        nargs="+",
        default=list(DEFAULT_NODE_CLASSES),
        help="Selected-node classes to patch during activation patching.",
    )
    selection_group.add_argument(
        "--activation-output-filename",
        type=str,
        default=None,
        help="Optional override for the activation-patching output filename stem.",
    )

    execution_group = parser.add_argument_group("execution")
    execution_group.add_argument(
        "--min-logit-diff",
        type=float,
        default=1.0,
        help="Discard examples below this correct-logit margin before patching.",
    )
    execution_group.add_argument(
        "--num-random-sources",
        type=int,
        default=5,
        help="Number of random source examples patched into each target example.",
    )
    execution_group.add_argument(
        "--save-to-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload EAP-IG NPZ outputs to the configured Hugging Face dataset repo.",
    )
    execution_group.add_argument(
        "--start-at",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[0],
        help="Stage at which to start the workflow.",
    )
    execution_group.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[-1],
        help="Stage after which to stop the workflow.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the workflow."""
    return build_parser().parse_args(argv)


def run_workflow(config: WorkflowConfig) -> None:
    """Execute the requested workflow stages."""
    validate_stage_window(config.start_at, config.stop_after)

    if config.includes_stage("eap-ig"):
        run_eap_ig_stage(
            config_dir=config.eap_ig_config_dir,
            save_to_hf=config.save_to_hf,
        )

    if config.includes_stage("top-components"):
        build_top_components(
            results_dir=config.eap_ig_results_dir,
            top_components_dir=config.top_components_dir,
            completeness_figures_dir=config.completeness_figures_dir,
            top_n=config.top_n,
        )

    selected_nodes_path = config.selected_nodes_path
    if config.includes_stage("node-selection"):
        selected_nodes_path, _ = write_selected_nodes(
            top_components_dir=config.top_components_dir,
            selected_nodes_dir=config.selected_nodes_dir,
            top_n=config.top_n,
            selection_limit=config.selection_limit,
        )

    activation_patch_path = config.activation_patch_output_path
    if config.includes_stage("activation-patch"):
        activation_patch_path = run_activation_patch_stage(
            classification_config=config.classification_config,
            selected_nodes_path=selected_nodes_path,
            node_classes=config.node_classes,
            top_n=config.top_n,
            output_filename=config.activation_output_filename,
            min_logit_diff=config.min_logit_diff,
            num_random_sources=config.num_random_sources,
        )

    if config.includes_stage("visualize"):
        plot_cross_task_performance(activation_patch_path, config.ctp_figures_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the workflow with the requested stage window."""
    args = parse_args(argv)
    run_workflow(WorkflowConfig.from_args(args))


if __name__ == "__main__":
    main()
