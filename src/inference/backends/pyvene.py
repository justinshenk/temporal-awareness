"""Pyvene backend implementation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .model_backend import Backend
from ..interventions import Intervention


class PyveneBackend(Backend):
    """Backend using pyvene for interventions."""

    def __init__(self, runner: Any, tokenizer: Any):
        super().__init__(runner)
        self._tokenizer = tokenizer
        if hasattr(self.runner._model, "transformer"):
            self._layers_attr = "transformer.h"
            self._layers = self.runner._model.transformer.h
            self._n_layers = len(self._layers)
            self._d_model = self.runner._model.config.n_embd
        elif hasattr(self.runner._model, "gpt_neox"):
            self._layers_attr = "gpt_neox.layers"
            self._layers = self.runner._model.gpt_neox.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner._model.config.hidden_size
        elif hasattr(self.runner._model, "model") and hasattr(
            self.runner._model.model, "layers"
        ):
            self._layers_attr = "model.layers"
            self._layers = self.runner._model.model.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner._model.config.hidden_size
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner._model)}")

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into token IDs tensor."""
        tokenizer = self.get_tokenizer()
        ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
        if prepend_bos:
            bos_id = tokenizer.bos_token_id
            if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
                bos = torch.tensor([[bos_id]], dtype=ids.dtype)
                ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=False)

    def _get_component_module(self, layer_idx: int, component: str):
        """Get the module for a specific component within a layer."""
        layer = self._layers[layer_idx]
        if component in ("resid_post", "resid_pre", "resid_mid"):
            return layer
        elif component == "attn_out":
            if hasattr(layer, "attn"):
                return layer.attn
            elif hasattr(layer, "attention"):
                return layer.attention
            elif hasattr(layer, "self_attn"):
                return layer.self_attn
            else:
                raise ValueError(
                    f"Cannot find attention module in layer: {type(layer)}"
                )
        elif component == "mlp_out":
            return layer.mlp
        else:
            raise ValueError(f"Unknown component: {component}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.encode(prompt)
        prompt_len = input_ids.shape[1]

        if (
            intervention is not None
            and isinstance(intervention, Intervention)
            and intervention.mode == "add"
        ):
            direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            layer_module = self._layers[intervention.layer]

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    steered = hidden + direction.unsqueeze(0).unsqueeze(0)
                    return (steered,) + output[1:]
                else:
                    return output + direction.unsqueeze(0).unsqueeze(0)

            generated = input_ids.clone()
            eos_id = self.get_tokenizer().eos_token_id

            for _ in range(max_new_tokens):
                hook = layer_module.register_forward_hook(steering_hook)

                with torch.no_grad():
                    outputs = self.runner._model(generated)
                    logits = outputs.logits

                hook.remove()

                if temperature > 0:
                    probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).unsqueeze(0)
                else:
                    next_token = (
                        logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                    )
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == eos_id:
                    break
        else:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                # Override model's default generation config to ensure greedy decoding
                "repetition_penalty": 1.0,
                "num_beams": 1,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = self.runner._model.generate(input_ids, **gen_kwargs)
            generated = output_ids

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits[0, -1, :], dim=-1)
        result = {}
        tokenizer = self.get_tokenizer()
        for token_str in target_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
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

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        cache = {}
        hooks = []

        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_hook(hook_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, tuple):
                        cache[hook_name] = out[0].detach()
                    else:
                        cache[hook_name] = out.detach()

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                outputs = self.runner._model(input_ids)
            logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run with gradients enabled for attribution patching."""
        cache = {}
        hooks = []

        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_hook(hook_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, tuple):
                        cache[hook_name] = out[0]
                    else:
                        cache[hook_name] = out

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen KV cache."""
        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        next_logits = prefill_logits[0, -1, :]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)

                generated_ids.append(next_token.item())

                if next_token.item() == eos_token_id:
                    break

                outputs = self.runner._model(
                    next_token.unsqueeze(0),
                    past_key_values=frozen_kv_cache,
                    use_cache=True,
                )
                next_logits = outputs.logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        """Initialize a KV cache wrapper for HF models."""

        class HFKVCache:
            def __init__(self):
                self.past_key_values = None
                self._frozen = False

            def freeze(self):
                self._frozen = True

            def unfreeze(self):
                self._frozen = False

            @property
            def frozen(self):
                return self._frozen

        return HFKVCache()

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits."""
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
        return outputs.logits

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        hooks = []
        for intervention in interventions:
            values = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            target = intervention.target
            mode = intervention.mode
            module = self._get_component_module(
                intervention.layer, intervention.component
            )

            def make_hook(values, target, mode):
                def intervention_hook(mod, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if target.is_all_positions:
                        if mode == "add":
                            hidden = hidden + values
                        elif mode == "set":
                            # Handle sequence length mismatch for 2D values
                            if values.ndim == 2:
                                seq_len = min(hidden.shape[1], values.shape[0])
                                new_hidden = hidden.clone()
                                new_hidden[:, :seq_len, :] = (
                                    values[:seq_len]
                                    .unsqueeze(0)
                                    .expand(hidden.shape[0], -1, -1)
                                )
                                hidden = new_hidden
                            else:
                                hidden = values.expand_as(hidden)
                        elif mode == "mul":
                            hidden = hidden * values
                    else:
                        for i, pos in enumerate(target.positions):
                            if pos < hidden.shape[1]:
                                # values may be [n_positions, hidden_size] or [hidden_size]
                                pos_values = (
                                    values[i]
                                    if values.ndim > 1 and i < len(values)
                                    else values
                                )
                                if mode == "add":
                                    hidden[:, pos, :] = hidden[:, pos, :] + pos_values
                                elif mode == "set":
                                    hidden[:, pos, :] = pos_values
                                elif mode == "mul":
                                    hidden[:, pos, :] = hidden[:, pos, :] * pos_values

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

                return intervention_hook

            hook = module.register_forward_hook(make_hook(values, target, mode))
            hooks.append(hook)

        with torch.no_grad():
            outputs = self.runner._model(input_ids)

        for hook in hooks:
            hook.remove()

        return outputs.logits

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients."""
        cache = {}
        hooks = []

        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_cache_hook(hook_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, tuple):
                        cache[hook_name] = out[0]
                    else:
                        cache[hook_name] = out

                return hook_fn

            hooks.append(module.register_forward_hook(make_cache_hook(name)))

        for intervention in interventions:
            values = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            target = intervention.target
            mode = intervention.mode
            alpha = getattr(intervention, "alpha", 1.0)
            target_values = None
            if (
                hasattr(intervention, "target_values")
                and intervention.target_values is not None
            ):
                target_values = torch.tensor(
                    intervention.target_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
            module = self._get_component_module(
                intervention.layer, intervention.component
            )

            def make_intervention_hook(values, target, mode, alpha, target_values):
                def intervention_hook(mod, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if target.is_all_positions:
                        if mode == "add":
                            hidden = hidden + values
                        elif mode == "set":
                            # Handle sequence length mismatch for 2D values
                            if values.ndim == 2:
                                seq_len = min(hidden.shape[1], values.shape[0])
                                new_hidden = hidden.clone()
                                new_hidden[:, :seq_len, :] = (
                                    values[:seq_len]
                                    .unsqueeze(0)
                                    .expand(hidden.shape[0], -1, -1)
                                )
                                hidden = new_hidden
                            else:
                                hidden = values.expand_as(hidden)
                        elif mode == "mul":
                            hidden = hidden * values
                        elif mode == "interpolate":
                            # Interpolation: hidden + alpha * (target - hidden)
                            if target_values is not None:
                                if target_values.ndim == 2:
                                    seq_len = min(
                                        hidden.shape[1], target_values.shape[0]
                                    )
                                    new_hidden = hidden.clone()
                                    tgt = (
                                        target_values[:seq_len]
                                        .unsqueeze(0)
                                        .expand(hidden.shape[0], -1, -1)
                                    )
                                    new_hidden[:, :seq_len, :] = hidden[
                                        :, :seq_len, :
                                    ] + alpha * (tgt - hidden[:, :seq_len, :])
                                    hidden = new_hidden
                                else:
                                    tgt = target_values.expand_as(hidden)
                                    hidden = hidden + alpha * (tgt - hidden)
                            else:
                                # Fallback: treat values as target
                                if values.ndim == 2:
                                    seq_len = min(hidden.shape[1], values.shape[0])
                                    new_hidden = hidden.clone()
                                    tgt = (
                                        values[:seq_len]
                                        .unsqueeze(0)
                                        .expand(hidden.shape[0], -1, -1)
                                    )
                                    new_hidden[:, :seq_len, :] = hidden[
                                        :, :seq_len, :
                                    ] + alpha * (tgt - hidden[:, :seq_len, :])
                                    hidden = new_hidden
                                else:
                                    tgt = values.expand_as(hidden)
                                    hidden = hidden + alpha * (tgt - hidden)
                    else:
                        for i, pos in enumerate(target.positions):
                            if pos < hidden.shape[1]:
                                # values may be [n_positions, hidden_size] or [hidden_size]
                                pos_values = (
                                    values[i]
                                    if values.ndim > 1 and i < len(values)
                                    else values
                                )
                                if mode == "add":
                                    hidden[:, pos, :] = hidden[:, pos, :] + pos_values
                                elif mode == "set":
                                    hidden[:, pos, :] = pos_values
                                elif mode == "mul":
                                    hidden[:, pos, :] = hidden[:, pos, :] * pos_values
                                elif mode == "interpolate":
                                    # Get target values for this position
                                    if target_values is not None:
                                        tgt_val = (
                                            target_values[i]
                                            if target_values.ndim > 1
                                            and i < len(target_values)
                                            else target_values
                                        )
                                    else:
                                        tgt_val = pos_values
                                    hidden[:, pos, :] = hidden[:, pos, :] + alpha * (
                                        tgt_val - hidden[:, pos, :]
                                    )

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

                return intervention_hook

            hook = module.register_forward_hook(
                make_intervention_hook(values, target, mode, alpha, target_values)
            )
            hooks.append(hook)

        try:
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache
