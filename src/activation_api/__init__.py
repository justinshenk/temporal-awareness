"""
Activation Extraction API for studying LLM internals.

A memory-efficient API for extracting internal activations from large language
models during forward passes. Supports selective layer extraction, streaming
to CPU/disk, and batched processing over large datasets.

Usage:
    from src.activation_api import ExtractionConfig, ActivationExtractor

    config = ExtractionConfig(
        layers=[0, 8, 16, 24, 31],
        positions=["last", "all"],
        module_types=["resid_post", "attn_out", "mlp_out"],
        stream_to="cpu",           # "cpu", "disk", or "gpu"
        output_dir="./activations",
        batch_size=4,
    )

    extractor = ActivationExtractor(model_name="meta-llama/Llama-3.1-8B-Instruct", config=config)
    result = extractor.extract(texts=["Hello, world!", "What is 2+2?"])

    # Access activations
    acts = result["resid_post", 16, "last"]  # (n_samples, d_model)
"""

from .config import ExtractionConfig, ModuleSpec
from .extractor import ActivationExtractor
from .result import ActivationResult
from .hooks import HookManager

__all__ = [
    "ExtractionConfig",
    "ModuleSpec",
    "ActivationExtractor",
    "ActivationResult",
    "HookManager",
]
