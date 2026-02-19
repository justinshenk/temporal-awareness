"""HuggingFace Transformers backend implementation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .model_backend import Backend
from ..interventions import Intervention


class HuggingFaceBackend(Backend):
    """Backend using HuggingFace Transformers for model inference."""

    def __init__(self, runner: Any, tokenizer: Any):
        super().__init__(runner)
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return self.runner._model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner._model.config.hidden_size

    def _get_layers(self):
        """Get the transformer layers from the model."""
        model = self.runner._model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers  # Llama, Mistral, Qwen
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2, GPT-Neo
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers  # GPT-NeoX
        raise ValueError(f"Unknown model architecture: {type(model)}")

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self.runner._tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        input_ids = tokens["input_ids"].to(self.runner.device)
        if prepend_bos and self.runner._tokenizer.bos_token_id is not None:
            bos = torch.tensor(
                [[self.runner._tokenizer.bos_token_id]], device=self.runner.device
            )
            input_ids = torch.cat([bos, input_ids], dim=1)
        return input_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        if intervention is not None:
            raise NotImplementedError(
                "HuggingFace backend does not support interventions"
            )

        input_ids = self.encode(prompt)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.runner._tokenizer.eos_token_id,
            # Override model's default generation config to ensure greedy decoding
            "repetition_penalty": 1.0,
            "num_beams": 1,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.runner._model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        for token_str in target_tokens:
            ids = self.runner._tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def _register_cache_hooks(
        self, cache: dict, names_filter: Optional[callable]
    ) -> list:
        """Register forward hooks to capture activations."""
        handles = []
        layers = self._get_layers()

        for i, layer in enumerate(layers):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):

                def make_hook(hook_name):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            cache[hook_name] = output[0].detach()
                        else:
                            cache[hook_name] = output.detach()

                    return hook_fn

                handle = layer.register_forward_hook(make_hook(name))
                handles.append(handle)

        return handles

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        cache = {}
        handles = self._register_cache_hooks(cache, names_filter)

        try:
            with torch.no_grad():
                outputs = self.runner._model(input_ids)
                logits = outputs.logits
        finally:
            for handle in handles:
                handle.remove()

        return logits, cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        cache = {}
        handles = self._register_cache_hooks(cache, names_filter)

        try:
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        finally:
            for handle in handles:
                handle.remove()

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError(
            "HuggingFace backend does not support cache generation yet"
        )

    def init_kv_cache(self):
        raise NotImplementedError("HuggingFace backend does not support KV cache yet")

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            return outputs.logits

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "HuggingFace backend does not support interventions yet"
        )

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError(
            "HuggingFace backend does not support interventions yet"
        )
