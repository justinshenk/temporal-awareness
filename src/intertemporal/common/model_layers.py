"""Project a layer selection onto models of different depth.

The layer lists in this repo were chosen for Qwen3-4B-Instruct-2507 (36 layers)
and contain indices 34 and 35, which do not exist on a 32-layer model. Truncating
them would silently drop the late-MLP band the analysis depends on.

Cross-model patching found that recovery curves align on *fractional depth*
(layer / n_layers) and not on absolute index, so fractional depth is the
principled coordinate to carry a layer selection across model sizes.
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

# Qwen3-4B-Instruct-2507: the model every hardcoded layer list was written for.
REFERENCE_N_LAYERS = 36


def scale_layers(
    reference_layers: list[int],
    n_layers: int,
    reference_n_layers: int = REFERENCE_N_LAYERS,
) -> list[int]:
    """Map layers chosen for a reference model onto a model of depth `n_layers`.

    Endpoints are preserved: the first layer maps to 0 and the last maps to
    `n_layers - 1`. Collisions are deduplicated, so the result may be shorter
    than the input.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if reference_n_layers < 2:
        raise ValueError(f"reference_n_layers must be >= 2, got {reference_n_layers}")
    if n_layers == reference_n_layers:
        return sorted(set(reference_layers))

    span, ref_span = n_layers - 1, reference_n_layers - 1
    scaled = {
        min(span, max(0, round(layer * span / ref_span))) for layer in reference_layers
    }
    return sorted(scaled)


def evenly_spaced_layers(n: int, n_layers: int) -> list[int]:
    """Pick `n` layers spread evenly across a model of depth `n_layers`.

    Endpoints are included, so the selection always spans the full depth.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 to span a depth, got {n}; use explicit layers")
    if n > n_layers:
        raise ValueError(f"cannot pick {n} layers from a {n_layers}-layer model")

    span = n_layers - 1
    return sorted({round(i * span / (n - 1)) for i in range(n)})


def validate_layers(layers: list[int], n_layers: int) -> list[int]:
    """Raise if any layer is out of range for a model of depth `n_layers`."""
    bad = [layer for layer in layers if not 0 <= layer < n_layers]
    if bad:
        raise ValueError(
            f"layers {bad} are out of range for a {n_layers}-layer model "
            f"(valid 0..{n_layers - 1}). Use scale_layers() to project them."
        )
    return layers


def fetch_n_layers(model_name: str) -> int:
    """Read a model's depth from its hub config.json.

    Only the config file is downloaded, never the weights, so this stays cheap
    enough to run before every extraction.
    """
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Multimodal checkpoints keep the transformer depth under text_config.
    for source in (config, config.get("text_config", {})):
        n_layers = source.get("num_hidden_layers")
        if n_layers is not None:
            return int(n_layers)

    raise ValueError(
        f"config.json for {model_name} has no num_hidden_layers; "
        "pass --n-model-layers to state the depth explicitly."
    )
