from __future__ import annotations

from pathlib import Path


QWEN3_4B_INSTRUCT_2507_VARIANT = "qwen3_4b_instruct_2507"

QWEN3_4B_INSTRUCT_2507_LAYERS = [14, 16, 18, 20, 22, 24, 26, 28]


def add_qwen3_4b_instruct_2507_flag(parser) -> None:
    parser.add_argument(
        "--qwen3-4b-instruct-2507",
        action="store_true",
        help=(
            "Use Qwen/Qwen3-4B-Instruct-2507 instead of Qwen3-32B, writing artifacts "
            "under results/qwen3_4b_instruct/."
        ),
    )


def probe_training_overrides_for_qwen3_4b_instruct_2507() -> dict[str, object]:
    return {
        "model_variant": QWEN3_4B_INSTRUCT_2507_VARIANT,
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "output_root_relative": "results/qwen3_4b_instruct/question_only_probe_variations",
        "selected_layers": list(QWEN3_4B_INSTRUCT_2507_LAYERS),
        "artifact_file_prefix": "qwen3_4b_instruct_question_only_probe",
        "result_root_name": "qwen3_4b_instruct",
    }


def probe_artifact_steering_overrides_for_qwen3_4b_instruct_2507() -> dict[str, object]:
    return {
        "model_variant": QWEN3_4B_INSTRUCT_2507_VARIANT,
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "result_root_name": "qwen3_4b_instruct",
        "artifact_file_prefix": "qwen3_4b_instruct_question_only_probe",
        "result_file_prefix": "mmraz_qwen3_4b_instruct_probe_artifact_steering_question_options_answer",
        "layers_to_test": list(QWEN3_4B_INSTRUCT_2507_LAYERS),
        "output_root_name": "qwen3_4b_instruct/probe_artifact_steering_question_options_answer_vast",
    }


def time_utility_steering_overrides_for_qwen3_4b_instruct_2507() -> dict[str, object]:
    return {
        "model_variant": QWEN3_4B_INSTRUCT_2507_VARIANT,
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "result_root_name": "qwen3_4b_instruct",
        "artifact_file_prefix": "qwen3_4b_instruct_question_only_probe",
        "output_file_prefix": "mmraz_time_utility_qwen3_4b_instruct_probe_steered",
        "plot_file_prefix": "qwen3_4b_instruct",
        "steering_layer": 22,
        "output_root_relative": "results/qwen3_4b_instruct/time_utility_experiment_probe_steered",
    }


def default_probe_artifact_search_roots(root: Path, result_root_name: str) -> list[Path]:
    return [
        root / "results" / result_root_name / "question_only_probe_variations",
        root / "results" / result_root_name,
        root / "results",
        Path(f"/workspace/results/{result_root_name}/question_only_probe_variations"),
        Path(f"/workspace/results/{result_root_name}"),
        Path("/workspace/results"),
        Path("/workspace"),
    ]


def default_probe_steering_reuse_roots(root: Path, output_dir: Path, partial_dir: Path, result_root_name: str) -> list[Path]:
    return [
        output_dir,
        partial_dir,
        root / "results" / result_root_name / "probe_artifact_steering_question_options_answer_vast",
        root / "results" / result_root_name / "probe_artifact_steering_question_options_answer_colab",
        root / "results" / result_root_name,
        Path(f"/workspace/results/{result_root_name}/probe_artifact_steering_question_options_answer_vast"),
        Path(f"/workspace/results/{result_root_name}/probe_artifact_steering_question_options_answer_colab"),
        Path(f"/workspace/results/{result_root_name}"),
    ]
