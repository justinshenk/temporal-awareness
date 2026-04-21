"""
Shared helpers for the Q&A EAP-IG workflow.
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config"
HF_REPO_ID = os.getenv("HF_REPO_ID", "Temporal_Awareness_EAP_IG")
SUPPORTED_QUADRATURES = {
    "gauss-chebyshev",
    "gauss-legendre",
    "riemann-midpoint",
}
QUADRATURE_ALIASES = {
    "midpoint": "riemann-midpoint",
}
DOT_CONFIG_SYMBOLS = {"â—", "â– "}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, normalizing unsupported dtypes."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor.numpy()


def load_and_merge_pairs(
    input_file: Path,
    template: str,
    option_keys: list[str],
    text_order: list[str],
) -> tuple[list[str], list[str]]:
    """Load pairs from ``input_file`` and return both clean and swapped prompts."""
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts = []
    swapped_prompts = []
    option_a, option_b = option_keys
    pairs = data.get("pairs", [])

    for pair in pairs:
        if isinstance(pair, str):
            prompt = pair
        elif isinstance(pair, dict):
            prompt = template.format(
                pair.get(text_order[0], ""),
                pair.get(text_order[1], ""),
                pair.get(text_order[2], ""),
            )
        else:
            raise RuntimeError("Incorrect type for pairs")

        prompt = prompt.replace("(A)", option_a)
        prompt = prompt.replace("(B)", option_b)

        clean_prompts.append(prompt)

        swapped_prompt = re.sub(
            f"{re.escape(option_a)}|{re.escape(option_b)}",
            lambda m: option_b if m.group(0) == option_a else option_a,
            prompt,
        )
        swapped_prompts.append(swapped_prompt)

    return clean_prompts, swapped_prompts


def load_config(config_path: Path) -> dict:
    """Load configuration from a YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_mech_interp_toolkit_installed() -> None:
    """Raise a clear error when ``mech_interp_toolkit`` is unavailable."""
    try:
        import mech_interp_toolkit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mech_interp_toolkit is required for the EAP-IG workflow. "
            'Install the pinned dependency with `pip install -e ".[eap_ig]"` '
            "before running this script."
        ) from exc


def extract_alnum(s: str) -> str:
    """Extract the semantic token string from an option marker."""
    out = []
    for c in s:
        if c.isalnum() or c in DOT_CONFIG_SYMBOLS:
            out.append(c)
    if out:
        return "".join(out)

    raise ValueError(f"malformed option string {s}")


def resolve_hf_repo_id(hf_api: HfApi, repo_id: str) -> str:
    """Return a fully qualified Hub repo id."""
    if "/" in repo_id:
        return repo_id

    whoami = hf_api.whoami()
    username = whoami.get("name")
    if not username:
        raise ValueError(
            "HF repo id must include a namespace like 'username/repo', or the "
            "HF token must expose an account name so one can be inferred."
        )
    return f"{username}/{repo_id}"


def resolve_quadrature(config: dict) -> str:
    """Resolve config quadrature into a supported value."""
    raw_quadrature = config["setup"].get("quadrature")
    if raw_quadrature is None:
        raw_quadrature = config["parameters"].get("quadrature", "riemann-midpoint")

    quadrature = QUADRATURE_ALIASES.get(raw_quadrature, raw_quadrature)
    if quadrature not in SUPPORTED_QUADRATURES:
        supported = ", ".join(sorted(SUPPORTED_QUADRATURES))
        raise ValueError(
            f"Unsupported quadrature '{raw_quadrature}'. Use one of: {supported}."
        )
    return quadrature


def resolve_config_path(config_path: Path) -> Path:
    """Resolve config paths relative to the local config directory by default."""
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return CONFIG_PATH / config_path


def build_layer_components(
    n_layers: int,
    granularity: str,
    layer_components: list[list[Any]] | list[tuple[Any, ...]] | None,
) -> list[tuple[int, str]]:
    """Build or normalize the requested layer/component list."""
    if layer_components is None:
        if granularity == "coarse":
            return [
                (layer, component)
                for layer in range(n_layers)
                for component in ("attn", "mlp")
            ]
        if granularity == "fine":
            return [
                (layer, component)
                for layer in range(n_layers)
                for component in ("z", "mlp_hidden")
            ]
        raise ValueError(f"Invalid granularity: {granularity}")

    return [tuple(lc) for lc in layer_components]  # type: ignore[misc]


def chunk_list(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
