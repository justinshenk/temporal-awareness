"""Main activation extractor — the core of the API.

Handles model loading, tokenization, forward passes, and orchestrating
hook-based activation capture with memory-efficient streaming.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ExtractionConfig, ModuleSpec
from .hooks import HookManager
from .result import ActivationResult


class ActivationExtractor:
    """Extract internal activations from language models.

    Supports two backends:
    1. TransformerLens (HookedTransformer) — clean hook interface, ideal for
       mech interp research. Supports all hook types natively.
    2. Raw HuggingFace — works with any AutoModelForCausalLM. Uses PyTorch
       forward hooks.

    Memory management:
    - Streaming to CPU: activations are moved off GPU after each forward pass
    - Streaming to disk: activations are written incrementally, never all in RAM
    - Batched processing: configurable batch size
    - Selective extraction: only hooks the layers/modules you ask for

    Usage:
        config = ExtractionConfig(
            layers=[0, 16, 31],
            module_types=["resid_post", "attn_out"],
            positions="last",
            stream_to="cpu",
            batch_size=4,
        )

        extractor = ActivationExtractor("meta-llama/Llama-3.1-8B-Instruct", config)
        result = extractor.extract(["Hello world", "What is 2+2?"])

        # result["resid_post", 16, "last"] -> (2, 4096) tensor
    """

    def __init__(
        self,
        model: Optional[Union[str, nn.Module, Any]] = None,
        config: Optional[ExtractionConfig] = None,
        model_name: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        """Initialize the extractor.

        Args:
            model: Either a model name string to load, or a pre-loaded model
                (nn.Module, HookedTransformer, or HuggingFace model).
            config: Extraction configuration. Uses defaults if not provided.
            model_name: Alias for model (string name). Use if model is already loaded.
            tokenizer: Pre-loaded tokenizer. Auto-loaded if not provided.
            device: Override device. Auto-detected if not set.
        """
        self.config = config or ExtractionConfig()
        self._model = None
        self._tokenizer = tokenizer
        self._model_name = model_name or (model if isinstance(model, str) else None)
        self._is_transformer_lens = False
        self._n_layers = None
        self._d_model = None
        self._hook_manager = HookManager(self.config)

        # Resolve device
        if device:
            self._device = device
        elif self.config.device:
            self._device = self.config.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model if string name provided
        if isinstance(model, str):
            self._load_model(model)
        elif model is not None:
            self._set_model(model)

    def _load_model(self, model_name: str):
        """Load a model by name, auto-detecting the best backend."""
        use_tl = self.config.use_transformer_lens

        if use_tl is None:
            # Auto-detect: try TransformerLens first for smaller models
            use_tl = self._should_use_transformer_lens(model_name)

        if use_tl:
            self._load_transformer_lens(model_name)
        else:
            self._load_huggingface(model_name)

    def _should_use_transformer_lens(self, model_name: str) -> bool:
        """Heuristic for whether to use TransformerLens."""
        # TransformerLens works well with these model families
        tl_supported = [
            "gpt2", "pythia", "gemma", "llama", "qwen",
            "mistral", "opt", "gpt-neo", "gpt-j",
        ]
        name_lower = model_name.lower()
        return any(family in name_lower for family in tl_supported)

    def _load_transformer_lens(self, model_name: str):
        """Load model via TransformerLens."""
        from transformer_lens import HookedTransformer

        print(f"Loading {model_name} via TransformerLens on {self._device}...")
        t0 = time.time()

        kwargs = {"device": self._device}
        if self.config.dtype:
            kwargs["dtype"] = self.config.resolve_dtype()

        self._model = HookedTransformer.from_pretrained(model_name, **kwargs)
        self._tokenizer = self._model.tokenizer
        self._is_transformer_lens = True
        self._n_layers = self._model.cfg.n_layers
        self._d_model = self._model.cfg.d_model

        print(f"  Loaded in {time.time() - t0:.1f}s "
              f"({self._n_layers} layers, d_model={self._d_model})")

    def _load_huggingface(self, model_name: str):
        """Load model via HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name} via HuggingFace on {self._device}...")
        t0 = time.time()

        kwargs = {"device_map": self._device}

        # Determine model loading dtype: prefer model_dtype, fall back to dtype
        load_dtype_str = self.config.model_dtype or self.config.dtype
        if load_dtype_str:
            load_dtype = self.config.resolve_dtype(load_dtype_str)
            # Newer transformers uses 'dtype'; fall back to 'torch_dtype'
            try:
                from transformers import __version__ as _tf_version
                _major, _minor = (int(x) for x in _tf_version.split(".")[:2])
                _dtype_key = "dtype" if (_major, _minor) >= (4, 50) else "torch_dtype"
            except Exception:
                _dtype_key = "torch_dtype"
            kwargs[_dtype_key] = load_dtype

        self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        self._is_transformer_lens = False
        self._n_layers = self._detect_n_layers()
        self._d_model = self._detect_d_model()

        print(f"  Loaded in {time.time() - t0:.1f}s "
              f"({self._n_layers} layers, d_model={self._d_model})")

    def _set_model(self, model: Any):
        """Set a pre-loaded model."""
        self._model = model

        # Detect if TransformerLens
        try:
            from transformer_lens import HookedTransformer
            if isinstance(model, HookedTransformer):
                self._is_transformer_lens = True
                self._n_layers = model.cfg.n_layers
                self._d_model = model.cfg.d_model
                if self._tokenizer is None:
                    self._tokenizer = model.tokenizer
                return
        except ImportError:
            pass

        self._is_transformer_lens = False
        self._n_layers = self._detect_n_layers()
        self._d_model = self._detect_d_model()

    def _detect_n_layers(self) -> int:
        """Detect number of layers in a HuggingFace model."""
        model = self._model
        # Try common attributes
        if hasattr(model, "config"):
            for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        # Count layer modules
        for name in ("model.layers", "transformer.h", "gpt_neox.layers"):
            mod = HookManager._get_module_by_path(model, name)
            if mod is not None and hasattr(mod, "__len__"):
                return len(mod)
        raise ValueError("Could not detect number of layers")

    def _detect_d_model(self) -> int:
        """Detect model hidden dimension."""
        model = self._model
        if hasattr(model, "config"):
            for attr in ("hidden_size", "d_model", "n_embd"):
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        raise ValueError("Could not detect d_model")

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("No model loaded. Pass a model name or model object.")
        return self._model

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def d_model(self) -> int:
        return self._d_model

    def extract(
        self,
        texts: Union[list[str], str],
        return_tokens: bool = False,
    ) -> ActivationResult:
        """Extract activations from a list of texts.

        This is the main entry point. Handles batching, hooks, memory
        management, and returns a structured result.

        Args:
            texts: Input text(s) to extract activations from.
            return_tokens: Whether to include tokenization in the result.

        Returns:
            ActivationResult with activations keyed by module spec.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Resolve which modules to hook
        modules = self.config.resolve_modules(self._n_layers)
        if not modules:
            raise ValueError("No modules to extract. Check your config.")

        print(f"Extracting from {len(texts)} texts, "
              f"{len(modules)} hooks, batch_size={self.config.batch_size}")
        for m in modules:
            print(f"  Hook: {m.key}")

        # Route to appropriate backend
        if self._is_transformer_lens:
            return self._extract_transformer_lens(texts, modules, return_tokens)
        else:
            return self._extract_pytorch(texts, modules, return_tokens)

    def _extract_transformer_lens(
        self,
        texts: list[str],
        modules: list[ModuleSpec],
        return_tokens: bool,
    ) -> ActivationResult:
        """Extract using TransformerLens run_with_cache."""
        names_filter = self._hook_manager.build_transformer_lens_filter(modules)
        all_activations: dict[str, list[torch.Tensor]] = {m.key: [] for m in modules}
        all_tokens = [] if return_tokens else None
        all_token_strs = [] if return_tokens else None

        n_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in tqdm(range(n_batches), desc="Extracting"):
            start = batch_idx * self.config.batch_size
            end = min(start + self.config.batch_size, len(texts))
            batch_texts = texts[start:end]

            for text in batch_texts:
                with torch.no_grad():
                    tokens = self._model.to_tokens(text)
                    if self.config.max_seq_len:
                        tokens = tokens[:, :self.config.max_seq_len]

                    _, cache = self._model.run_with_cache(
                        tokens, names_filter=names_filter
                    )

                seq_len = tokens.shape[1]
                extracted = self._hook_manager.extract_from_cache(
                    cache, modules, seq_len
                )

                for key, tensor in extracted.items():
                    all_activations[key].append(tensor)

                if return_tokens:
                    tok_ids = tokens[0].tolist()
                    all_tokens.append(tok_ids)
                    all_token_strs.append(
                        [self._tokenizer.decode([t]) for t in tok_ids]
                    )

                # Free cache memory
                del cache
                if self._device == "cuda":
                    torch.cuda.empty_cache()

        # Concatenate batches
        final_activations = {}
        for key, tensor_list in all_activations.items():
            if tensor_list:
                # Each tensor is (1, ...) from a single sample
                # Squeeze batch dim if present, then stack
                squeezed = []
                for t in tensor_list:
                    if t.dim() > 1 and t.shape[0] == 1:
                        squeezed.append(t.squeeze(0))
                    else:
                        squeezed.append(t)
                final_activations[key] = torch.stack(squeezed, dim=0)

        # Handle disk streaming
        if self.config.stream_to == "disk":
            return self._save_and_return(
                final_activations, texts, all_tokens, all_token_strs, modules
            )

        return ActivationResult(
            activations=final_activations,
            metadata={
                "model_name": self._model_name,
                "n_layers": self._n_layers,
                "d_model": self._d_model,
                "backend": "transformer_lens",
                "config": {
                    "layers": self.config.layers,
                    "module_types": self.config.module_types,
                    "positions": self.config.positions,
                    "stream_to": self.config.stream_to,
                },
            },
            tokens=all_tokens,
            token_strings=all_token_strs,
            texts=texts,
        )

    def _extract_pytorch(
        self,
        texts: list[str],
        modules: list[ModuleSpec],
        return_tokens: bool,
    ) -> ActivationResult:
        """Extract using raw PyTorch forward hooks."""
        self._hook_manager.register_hooks_pytorch(self._model, modules)
        all_tokens = [] if return_tokens else None
        all_token_strs = [] if return_tokens else None

        n_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        try:
            with self._hook_manager:
                for batch_idx in tqdm(range(n_batches), desc="Extracting"):
                    start = batch_idx * self.config.batch_size
                    end = min(start + self.config.batch_size, len(texts))
                    batch_texts = texts[start:end]

                    # Tokenize batch
                    encoded = self._tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len,
                    ).to(self._device)

                    with torch.no_grad():
                        self._model(**encoded)

                    if return_tokens:
                        for i in range(len(batch_texts)):
                            mask = encoded["attention_mask"][i].bool()
                            tok_ids = encoded["input_ids"][i][mask].tolist()
                            all_tokens.append(tok_ids)
                            all_token_strs.append(
                                [self._tokenizer.decode([t]) for t in tok_ids]
                            )

                    # Free GPU memory
                    del encoded
                    if self._device == "cuda":
                        torch.cuda.empty_cache()

            # Collect from buffer
            final_activations = {}
            for key, tensor_list in self._hook_manager.buffer.items():
                if tensor_list:
                    final_activations[key] = torch.cat(tensor_list, dim=0)

        finally:
            self._hook_manager.clear()

        if self.config.stream_to == "disk":
            return self._save_and_return(
                final_activations, texts, all_tokens, all_token_strs, modules
            )

        return ActivationResult(
            activations=final_activations,
            metadata={
                "model_name": self._model_name,
                "n_layers": self._n_layers,
                "d_model": self._d_model,
                "backend": "pytorch",
                "config": {
                    "layers": self.config.layers,
                    "module_types": self.config.module_types,
                    "positions": self.config.positions,
                    "stream_to": self.config.stream_to,
                },
            },
            tokens=all_tokens,
            token_strings=all_token_strs,
            texts=texts,
        )

    def _save_and_return(
        self,
        activations: dict[str, torch.Tensor],
        texts: list[str],
        tokens: Optional[list],
        token_strings: Optional[list],
        modules: list[ModuleSpec],
    ) -> ActivationResult:
        """Save activations to disk and return a lightweight result."""
        result = ActivationResult(
            activations=activations,
            metadata={
                "model_name": self._model_name,
                "n_layers": self._n_layers,
                "d_model": self._d_model,
            },
            tokens=tokens,
            token_strings=token_strings,
            texts=texts,
        )
        result.save(self.config.output_dir, format=self.config.output_format)
        print(f"  Saved activations to {self.config.output_dir}")
        return result

    def extract_dataset(
        self,
        dataset: list[dict],
        text_key: str = "text",
        label_key: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> ActivationResult:
        """Extract activations from a dataset of dicts.

        Convenience method for processing structured datasets.

        Args:
            dataset: List of dicts with at least a text field.
            text_key: Key for the text field in each dict.
            label_key: Optional key for labels (stored in metadata).
            max_samples: Maximum number of samples to process.

        Returns:
            ActivationResult with optional label metadata.
        """
        if max_samples:
            dataset = dataset[:max_samples]

        texts = [d[text_key] for d in dataset]
        result = self.extract(texts, return_tokens=True)

        if label_key:
            labels = [d.get(label_key) for d in dataset]
            result.metadata["labels"] = labels

        return result

    def compare_layers(
        self,
        text: str,
        layers: Optional[list[int]] = None,
        module_type: str = "resid_post",
    ) -> dict[int, torch.Tensor]:
        """Quick comparison of activations across layers for a single input.

        Useful for visualizing how representations evolve through the model.

        Args:
            text: Single input text.
            layers: Layers to compare. Defaults to evenly spaced across model.
            module_type: Which module type to compare.

        Returns:
            Dict mapping layer index to activation tensor.
        """
        if layers is None:
            # Sample ~8 evenly spaced layers
            n = min(8, self._n_layers)
            layers = [int(i * (self._n_layers - 1) / (n - 1)) for i in range(n)]

        # Create a temporary config for this comparison
        temp_config = ExtractionConfig(
            layers=layers,
            module_types=[module_type],
            positions=self.config.positions,
            stream_to="cpu",
            batch_size=1,
        )

        # Swap config temporarily
        original_config = self.config
        self.config = temp_config
        self._hook_manager = HookManager(temp_config)

        try:
            result = self.extract([text])
        finally:
            self.config = original_config
            self._hook_manager = HookManager(original_config)

        return {
            layer: result[module_type, layer]
            for layer in layers
            if (module_type, layer) in result
        }
