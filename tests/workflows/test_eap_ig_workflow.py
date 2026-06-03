import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.workflows.eap_ig_workflow import (
    build_selected_node_groups,
    plot_cross_task_performance,
    process_eap_ig_result_folder,
    topk_components,
)


def test_topk_components_orders_by_absolute_value() -> None:
    components = {
        "z/0": np.array([0.1, -0.8, 0.3], dtype=np.float32),
        "mlp_hidden/1": np.array([0.2, -0.4], dtype=np.float32),
    }

    result = topk_components(components, 3)

    assert result == [
        ("z/0", 1, -0.800000011920929),
        ("mlp_hidden/1", 1, -0.4000000059604645),
        ("z/0", 2, 0.30000001192092896),
    ]


def test_process_eap_ig_result_folder_merges_batches(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "demo_short_first_logit_A_batch_00000.npz",
        **{
            "step_20__z__0": np.array([[1.0, 2.0]], dtype=np.float32),
            "step_20__mlp_hidden__1": np.array([[3.0, 4.0]], dtype=np.float32),
            "step_20__clean_logits": np.array([[5.0, 6.0]], dtype=np.float32),
            "step_20__corrupted_logits": np.array([[7.0, 8.0]], dtype=np.float32),
            "metadata__config_json": np.array('{"x": 1}'),
        },
    )
    np.savez_compressed(
        tmp_path / "demo_short_first_logit_A_batch_00001.npz",
        **{
            "step_20__z__0": np.array([[9.0, 10.0]], dtype=np.float32),
            "step_20__mlp_hidden__1": np.array([[11.0, 12.0]], dtype=np.float32),
            "step_20__clean_logits": np.array([[13.0, 14.0]], dtype=np.float32),
            "step_20__corrupted_logits": np.array([[15.0, 16.0]], dtype=np.float32),
        },
    )

    result = process_eap_ig_result_folder(tmp_path)

    assert set(result) == {"short_first_logit_A"}
    merged = result["short_first_logit_A"]
    np.testing.assert_array_equal(
        merged["z/0"],
        np.array([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        merged["mlp_hidden/1"],
        np.array([[3.0, 4.0], [11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        merged["clean_logits"],
        np.array([[5.0, 6.0], [13.0, 14.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        merged["corrupted_logits"],
        np.array([[7.0, 8.0], [15.0, 16.0]], dtype=np.float32),
    )


def test_build_selected_node_groups_matches_existing_1000_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_path = repo_root / "data" / "selected_nodes" / "final_1000_QnA.pkl"
    top_components_dir = repo_root / "data" / "top_n_nodes"

    expected = pickle.load(expected_path.open("rb"))
    actual = build_selected_node_groups(
        top_components_dir,
        top_n=1000,
        selection_limit=300,
    )

    assert actual == expected


def test_plot_cross_task_performance_writes_png(tmp_path: Path) -> None:
    payload = {
        "metadata": {"option_keys": ["short", "long"]},
        "base_responses": [
            {"short_logit": 1.0, "long_logit": 0.0, "kept": True},
            {"short_logit": 0.0, "long_logit": 1.0, "kept": False},
        ],
        "patched_responses": {
            "short_to_long": [{"short_logit": 0.2, "long_logit": 0.9}],
            "long_to_short": [{"short_logit": 0.8, "long_logit": 0.1}],
        },
    }
    result_path = tmp_path / "example.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path = plot_cross_task_performance(result_path, tmp_path / "figures")

    assert output_path.exists()
    assert output_path.suffix == ".png"
