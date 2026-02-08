"""Pyvene backend implementation."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import Backend
from ..interventions import Intervention


class PyveneBackend(Backend):
    """Backend using pyvene for interventions."""

    def __init__(self, runner):
        super().__init__(runner)
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
        return self.runner._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

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
        input_ids = self.tokenize(prompt)
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
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = self.runner._model.generate(input_ids, **gen_kwargs)
            generated = output_ids

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
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
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.tokenize(prompt)
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

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
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

                    if target.axis == "all":
                        if mode == "add":
                            hidden = hidden + values
                        elif mode == "set":
                            hidden = values.expand_as(hidden)
                        elif mode == "mul":
                            hidden = hidden * values
                    elif target.axis == "position":
                        for pos in target.positions:
                            if pos < hidden.shape[1]:
                                if mode == "add":
                                    hidden[:, pos, :] = hidden[:, pos, :] + values
                                elif mode == "set":
                                    hidden[:, pos, :] = values
                                elif mode == "mul":
                                    hidden[:, pos, :] = hidden[:, pos, :] * values

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

    def forward_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
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
            module = self._get_component_module(
                intervention.layer, intervention.component
            )

            def make_intervention_hook(values, target, mode):
                def intervention_hook(mod, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if target.axis == "all":
                        if mode == "add":
                            hidden = hidden + values
                        elif mode == "set":
                            hidden = values.expand_as(hidden)
                        elif mode == "mul":
                            hidden = hidden * values
                        elif mode == "interpolate":
                            hidden = values.expand_as(hidden)
                    elif target.axis == "position":
                        for pos in target.positions:
                            if pos < hidden.shape[1]:
                                if mode == "add":
                                    hidden[:, pos, :] = hidden[:, pos, :] + values
                                elif mode == "set":
                                    hidden[:, pos, :] = values
                                elif mode == "mul":
                                    hidden[:, pos, :] = hidden[:, pos, :] * values
                                elif mode == "interpolate":
                                    hidden[:, pos, :] = values

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

                return intervention_hook

            hook = module.register_forward_hook(
                make_intervention_hook(values, target, mode)
            )
            hooks.append(hook)

        try:
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache
