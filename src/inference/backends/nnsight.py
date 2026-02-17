"""NNsight backend implementation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .backend import Backend
from ..interventions import Intervention


class NNsightBackend(Backend):
    """Backend using NNsight for model inference and interventions."""

    supports_inference_mode = False  # NNsight trace conflicts with inference_mode

    def __init__(self, runner):
        super().__init__(runner)
        model = self.runner._model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style
            self._layers = model.transformer.h
            self._layers_path = "transformer.h"
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            # Pythia / GPT-NeoX style
            self._layers = model.gpt_neox.layers
            self._layers_path = "gpt_neox.layers"
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama / Mistral / Qwen style
            self._layers = model.model.layers
            self._layers_path = "model.layers"
        else:
            raise ValueError(f"Unknown model architecture: {type(model)}")

    def _get_layer(self, layer_idx: int):
        """Get layer module through model path (works inside trace context)."""
        if self._layers_path == "transformer.h":
            return self.runner._model.transformer.h[layer_idx]
        elif self._layers_path == "gpt_neox.layers":
            return self.runner._model.gpt_neox.layers[layer_idx]
        else:
            return self.runner._model.model.layers[layer_idx]

    def get_tokenizer(self):
        return self.runner._model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner._model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner._model.config.hidden_size

    def _get_lm_head(self):
        """Get the language model head module for logits."""
        model = self.runner._model
        if hasattr(model, "lm_head"):
            return model.lm_head  # GPT-2, Llama, Qwen, etc.
        elif hasattr(model, "embed_out"):
            return model.embed_out  # Pythia / GPT-NeoX
        else:
            raise ValueError(f"Cannot find lm_head for model: {type(model)}")

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        tokenizer = self.get_tokenizer()
        ids = tokenizer(text, return_tensors="pt").input_ids
        if prepend_bos:
            bos_id = tokenizer.bos_token_id
            if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
                bos = torch.tensor([[bos_id]], dtype=ids.dtype)
                ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text with optional interventions using native NNsight generate."""
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.get_tokenizer().eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        # Check if we have an intervention to apply
        if (
            intervention is not None
            and isinstance(intervention, Intervention)
            and intervention.mode == "add"
        ):
            steering_layer_idx = intervention.layer
            steering_direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

            # Use native generate with intervention via tracer.all()
            with self.runner._model.generate(input_ids, **gen_kwargs) as tracer:
                with tracer.all():
                    layer = self._get_layer(steering_layer_idx)
                    # Apply steering to all generation steps
                    layer.output[0][:, :, :] += steering_direction
                output_ids = self.runner._model.generator.output.save()

            return self.decode(output_ids[0, prompt_len:])
        else:
            # No intervention - use native generate directly
            with self.runner._model.generate(input_ids, **gen_kwargs) as tracer:
                output_ids = self.runner._model.generator.output.save()

            return self.decode(output_ids[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with self.runner._model.trace(input_ids):
            logits = self._get_lm_head().output.save()

        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
        result = {}
        tokenizer = self.get_tokenizer()
        for token_str in target_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.tokenize(prompt)
        with self.runner._model.trace(input_ids):
            logits = self._get_lm_head().output.save()

        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def _get_component_module(self, layer, component: str):
        """Get the module for a specific component within a layer."""
        if component in ("resid_post", "resid_pre", "resid_mid"):
            return layer
        elif component == "attn_out":
            if self._layers_path == "transformer.h":
                return layer.attn  # GPT-2
            elif self._layers_path == "gpt_neox.layers":
                return layer.attention  # Pythia / GPT-NeoX
            else:
                return layer.self_attn  # Llama / Mistral / Qwen
        elif component == "mlp_out":
            return layer.mlp
        else:
            raise ValueError(f"Unknown component: {component}")

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        cache = {}

        hooks_to_capture = set()
        for i in range(len(self._layers)):
            for component in ["attn_out", "mlp_out", "resid_post"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.add((i, component, name))

        with self.runner._model.trace(input_ids):
            for layer_idx in range(len(self._layers)):
                layer = self._get_layer(layer_idx)

                for component in ["attn_out", "mlp_out", "resid_post"]:
                    name = f"blocks.{layer_idx}.hook_{component}"
                    if (layer_idx, component, name) not in hooks_to_capture:
                        continue

                    module = self._get_component_module(layer, component)

                    if component == "mlp_out":
                        out = module.output.save()
                    else:
                        out = module.output[0].save()
                    cache[name] = out

            logits = self._get_lm_head().output.save()

        result_cache = {}
        for k, v in cache.items():
            v = v.detach().clone()
            if v.ndim == 2:
                v = v.unsqueeze(0)
            result_cache[k] = v
        return logits.detach(), result_cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run with gradients enabled - nnsight preserves gradients by default."""
        return self.run_with_cache(input_ids, names_filter, None)

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Not implemented for nnsight backend."""
        raise NotImplementedError(
            "generate_from_cache not supported for nnsight backend"
        )

    def init_kv_cache(self):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits."""
        with self.runner._model.trace(input_ids):
            logits = self._get_lm_head().output.save()
        return logits.detach()

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        with self.runner._model.trace(input_ids):
            for intervention in interventions:
                layer = self._layers[intervention.layer]
                module = self._get_component_module(layer, intervention.component)
                values = torch.tensor(
                    intervention.scaled_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                target = intervention.target
                mode = intervention.mode
                component = intervention.component

                if component == "mlp_out":
                    out = module.output
                else:
                    out = module.output[0]

                if target.axis == "all":
                    if mode == "add":
                        out[:, :] += values
                    elif mode == "set":
                        out[:, :] = values
                    elif mode == "mul":
                        out[:, :] *= values
                elif target.axis == "position":
                    seq_len = out.shape[0] if out.ndim == 2 else out.shape[1]
                    for pos in target.positions:
                        if pos < seq_len:
                            if out.ndim == 2:
                                if mode == "add":
                                    out[pos, :] += values
                                elif mode == "set":
                                    out[pos, :] = values
                                elif mode == "mul":
                                    out[pos, :] *= values
                            else:
                                if mode == "add":
                                    out[:, pos, :] += values
                                elif mode == "set":
                                    out[:, pos, :] = values
                                elif mode == "mul":
                                    out[:, pos, :] *= values

            logits = self._get_lm_head().output.save()

        return logits.detach()

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients."""
        cache = {}

        intervention_lookup = {}
        for intervention in interventions:
            key = (intervention.layer, intervention.component)
            intervention_lookup[key] = intervention

        layers_to_capture = set()
        for i in range(len(self._layers)):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):
                layers_to_capture.add(i)

        with self.runner._model.trace(input_ids):
            for layer_idx in range(len(self._layers)):
                layer = self._get_layer(layer_idx)

                for component in ["resid_post", "attn_out", "mlp_out"]:
                    key = (layer_idx, component)
                    intervention = intervention_lookup.get(key)

                    should_cache = (
                        layer_idx in layers_to_capture and component == "resid_post"
                    )

                    if intervention is None and not should_cache:
                        continue

                    module = self._get_component_module(layer, component)

                    if component == "mlp_out":
                        out = module.output
                    else:
                        out = module.output[0]

                    if intervention is not None:
                        values = torch.tensor(
                            intervention.scaled_values,
                            dtype=self.runner.dtype,
                            device=self.runner.device,
                        )
                        target = intervention.target
                        mode = intervention.mode

                        if target.axis == "all":
                            if mode == "add":
                                out[:, :] += values
                            elif mode == "set":
                                out[:, :] = values
                            elif mode == "mul":
                                out[:, :] *= values
                            elif mode == "interpolate":
                                out[:, :] = values
                        elif target.axis == "position":
                            seq_len = out.shape[0] if out.ndim == 2 else out.shape[1]
                            for pos in target.positions:
                                if pos < seq_len:
                                    if out.ndim == 2:
                                        if mode == "add":
                                            out[pos, :] += values
                                        elif mode == "set":
                                            out[pos, :] = values
                                        elif mode == "mul":
                                            out[pos, :] *= values
                                        elif mode == "interpolate":
                                            out[pos, :] = values
                                    else:
                                        if mode == "add":
                                            out[:, pos, :] += values
                                        elif mode == "set":
                                            out[:, pos, :] = values
                                        elif mode == "mul":
                                            out[:, pos, :] *= values
                                        elif mode == "interpolate":
                                            out[:, pos, :] = values

                    if should_cache:
                        name = f"blocks.{layer_idx}.hook_resid_post"
                        cache[name] = out.save()

            logits = self._get_lm_head().output.save()

        result_cache = {}
        for k, v in cache.items():
            v = v.detach().clone()
            if v.ndim == 2:
                v = v.unsqueeze(0)
            result_cache[k] = v

        return logits.detach(), result_cache
