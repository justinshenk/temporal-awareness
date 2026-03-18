"""Pyvene backend implementation using actual pyvene IntervenableModel."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

try:
    import pyvene as pv
    from pyvene import (
        IntervenableConfig,
        IntervenableModel,
        RepresentationConfig,
    )
    from pyvene.models.interventions import (
        AdditionIntervention,
        VanillaIntervention,
        SourcelessIntervention,
    )

    PYVENE_AVAILABLE = True
except ImportError:
    PYVENE_AVAILABLE = False

from .model_backend import Backend
from ..interventions import Intervention


class MultiplyIntervention(SourcelessIntervention if PYVENE_AVAILABLE else object):
    """Custom intervention that multiplies activations by a value."""

    def __init__(self, embed_dim: int, multiplier: torch.Tensor):
        if not PYVENE_AVAILABLE:
            raise ImportError("pyvene is required for PyveneBackend")
        super().__init__(embed_dim=embed_dim)
        self.multiplier = multiplier

    def forward(self, base, source=None, subspaces=None):
        return base * self.multiplier


class InterpolateIntervention(SourcelessIntervention if PYVENE_AVAILABLE else object):
    """Custom intervention that interpolates between base and target values."""

    def __init__(
        self, embed_dim: int, target_values: torch.Tensor, alpha: float = 0.5
    ):
        if not PYVENE_AVAILABLE:
            raise ImportError("pyvene is required for PyveneBackend")
        super().__init__(embed_dim=embed_dim)
        self.target_values = target_values
        self.alpha = alpha

    def forward(self, base, source=None, subspaces=None):
        return base + self.alpha * (self.target_values - base)


class PyveneBackend(Backend):
    """Backend using pyvene's IntervenableModel for interventions.

    This backend wraps the model with pyvene's IntervenableModel to leverage
    pyvene's intervention infrastructure for activation patching and steering.
    """

    def __init__(self, runner: Any, tokenizer: Any):
        if not PYVENE_AVAILABLE:
            raise ImportError(
                "pyvene is required for PyveneBackend. "
                "Install with: pip install pyvene"
            )
        super().__init__(runner)
        self._tokenizer = tokenizer

        # Detect model architecture
        model = self.runner._model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._layers_attr = "transformer.h"
            self._layers = model.transformer.h
            self._n_layers = len(self._layers)
            self._d_model = model.config.n_embd
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            self._layers_attr = "gpt_neox.layers"
            self._layers = model.gpt_neox.layers
            self._n_layers = len(self._layers)
            self._d_model = model.config.hidden_size
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers_attr = "model.layers"
            self._layers = model.model.layers
            self._n_layers = len(self._layers)
            self._d_model = model.config.hidden_size
        else:
            raise ValueError(f"Unknown model architecture: {type(model)}")

    def _get_pyvene_component(self, layer_idx: int, component: str) -> str:
        """Convert layer index and component to pyvene component path.

        Args:
            layer_idx: Layer index
            component: Component name (resid_post, attn_out, mlp_out, etc.)

        Returns:
            Pyvene component path like "model.layers[0].mlp.output"
        """
        base = self._layers_attr

        if component == "resid_post":
            # Output of the entire layer block
            return f"{base}[{layer_idx}].output"
        elif component == "resid_pre":
            # Input to the layer block
            return f"{base}[{layer_idx}].input"
        elif component == "mlp_out":
            return f"{base}[{layer_idx}].mlp.output"
        elif component == "attn_out":
            # Attention output varies by architecture
            if self._layers_attr == "transformer.h":
                return f"{base}[{layer_idx}].attn.output"
            elif self._layers_attr == "gpt_neox.layers":
                return f"{base}[{layer_idx}].attention.output"
            else:
                return f"{base}[{layer_idx}].self_attn.output"
        else:
            raise ValueError(f"Unknown component: {component}")

    def _create_intervenable_model(
        self, interventions: Sequence[Intervention]
    ) -> IntervenableModel:
        """Create an IntervenableModel configured for the given interventions."""
        configs = []

        for intervention in interventions:
            component_path = self._get_pyvene_component(
                intervention.layer, intervention.component
            )

            # Create the appropriate pyvene intervention type
            if intervention.mode == "add":
                # AdditionIntervention adds source to base
                values = torch.tensor(
                    intervention.scaled_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                intervention_type = AdditionIntervention(
                    embed_dim=self._d_model,
                    source_representation=values,
                )
            elif intervention.mode == "set":
                # VanillaIntervention replaces base with source
                values = torch.tensor(
                    intervention.scaled_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                intervention_type = VanillaIntervention(
                    embed_dim=self._d_model,
                    source_representation=values,
                )
            elif intervention.mode == "mul":
                values = torch.tensor(
                    intervention.scaled_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                intervention_type = MultiplyIntervention(
                    embed_dim=self._d_model,
                    multiplier=values,
                )
            elif intervention.mode == "interpolate":
                target_values = torch.tensor(
                    intervention.target_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                intervention_type = InterpolateIntervention(
                    embed_dim=self._d_model,
                    target_values=target_values,
                    alpha=intervention.alpha,
                )
            else:
                raise ValueError(f"Unknown intervention mode: {intervention.mode}")

            config = RepresentationConfig(
                layer=intervention.layer,
                component=component_path,
                intervention=intervention_type,
            )
            configs.append(config)

        intervenable_config = IntervenableConfig(representations=configs)
        return IntervenableModel(intervenable_config, model=self.runner._model)

    def _get_unit_locations(
        self, interventions: Sequence[Intervention], seq_len: int
    ) -> dict:
        """Convert intervention targets to pyvene unit_locations format."""
        # pyvene expects unit_locations as a dict mapping intervention index
        # to position specifications
        locations = {}

        for idx, intervention in enumerate(interventions):
            target = intervention.target
            if target.is_all_positions:
                # All positions: use range from 0 to seq_len
                locations[idx] = list(range(seq_len))
            else:
                # Specific positions
                locations[idx] = list(target.positions)

        return locations

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
        ids = self._tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
        if prepend_bos:
            bos_id = self._tokenizer.bos_token_id
            if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
                bos = torch.tensor([[bos_id]], dtype=ids.dtype)
                ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text with optional intervention using pyvene."""
        input_ids = self.encode(prompt)
        prompt_len = input_ids.shape[1]

        if intervention is not None:
            # Use pyvene's IntervenableModel for generation with intervention
            intervenable = self._create_intervenable_model([intervention])
            unit_locations = self._get_unit_locations([intervention], input_ids.shape[1])

            generated = input_ids.clone()
            eos_id = self._tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    # Run intervened forward pass
                    _, outputs = intervenable(
                        {"input_ids": generated},
                        unit_locations={"sources->base": (None, unit_locations)},
                    )
                    logits = outputs.logits

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
            # No intervention - use standard generation
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
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
        for token_str in target_tokens:
            ids = self._tokenizer.encode(token_str, add_special_tokens=False)
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

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run with cache using pyvene's activation collection."""
        cache = {}
        hooks = []

        # Use standard PyTorch hooks for caching (pyvene's getter hooks work similarly)
        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_hook(hook_name, use_input=False):
                def hook_fn(mod, inp, out):
                    if use_input:
                        val = inp[0] if isinstance(inp, tuple) else inp
                    else:
                        val = out[0] if isinstance(out, tuple) else out
                    cache[hook_name] = val.detach()

                return hook_fn

            use_input = component == "resid_pre"
            hooks.append(module.register_forward_hook(make_hook(name, use_input)))

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
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_hook(hook_name, use_input=False):
                def hook_fn(mod, inp, out):
                    if use_input:
                        val = inp[0] if isinstance(inp, tuple) else inp
                    else:
                        val = out[0] if isinstance(out, tuple) else out
                    cache[hook_name] = val

                return hook_fn

            use_input = component == "resid_pre"
            hooks.append(module.register_forward_hook(make_hook(name, use_input)))

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
        eos_token_id = self._tokenizer.eos_token_id
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
        """Run forward with interventions using pyvene's IntervenableModel."""
        if not interventions:
            return self.forward(input_ids)

        intervenable = self._create_intervenable_model(interventions)
        unit_locations = self._get_unit_locations(interventions, input_ids.shape[1])

        with torch.no_grad():
            _, outputs = intervenable(
                {"input_ids": input_ids},
                unit_locations={"sources->base": (None, unit_locations)},
            )

        return outputs.logits

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations.

        Note: This uses pyvene for interventions but standard hooks for caching,
        since pyvene's activation collection is separate from interventions.
        """
        cache = {}
        hooks = []

        # Set up cache hooks
        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)

            def make_cache_hook(hook_name, use_input=False):
                def hook_fn(mod, inp, out):
                    if use_input:
                        val = inp[0] if isinstance(inp, tuple) else inp
                    else:
                        val = out[0] if isinstance(out, tuple) else out
                    cache[hook_name] = val

                return hook_fn

            use_input = component == "resid_pre"
            hooks.append(module.register_forward_hook(make_cache_hook(name, use_input)))

        try:
            if interventions:
                intervenable = self._create_intervenable_model(interventions)
                unit_locations = self._get_unit_locations(
                    interventions, input_ids.shape[1]
                )

                _, outputs = intervenable(
                    {"input_ids": input_ids},
                    unit_locations={"sources->base": (None, unit_locations)},
                )
                logits = outputs.logits
            else:
                outputs = self.runner._model(input_ids)
                logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache

    def _get_embed_tokens(self):
        """Get the token embedding module."""
        model = self.runner._model
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
            return model.gpt_neox.embed_in
        raise ValueError(f"Cannot find embedding module for: {type(model)}")

    def _get_lm_head(self):
        """Get the language model head module."""
        model = self.runner._model
        if hasattr(model, "lm_head"):
            return model.lm_head
        if hasattr(model, "embed_out"):
            return model.embed_out
        raise ValueError(f"Cannot find lm_head for: {type(model)}")

    def get_W_E(self) -> torch.Tensor:
        """Get the token embedding matrix W_E."""
        embed = self._get_embed_tokens()
        return embed.weight

    def get_W_U(self) -> torch.Tensor:
        """Get the unembedding matrix W_U."""
        lm_head = self._get_lm_head()
        return lm_head.weight.T

    def get_b_U(self) -> torch.Tensor | None:
        """Get the unembedding bias b_U."""
        lm_head = self._get_lm_head()
        return getattr(lm_head, "bias", None)

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory using HF generate() with KV caching."""
        input_ids = torch.tensor([token_ids], device=self.runner.device)
        prompt_len = len(token_ids)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
            "repetition_penalty": 1.0,
            "num_beams": 1,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.runner._model.generate(input_ids, **gen_kwargs)

            prefix_outputs = self.runner._model(input_ids)
            prefix_logits = prefix_outputs.logits[0]
            prefix_log_probs = torch.log_softmax(prefix_logits, dim=-1)

        all_logprobs: list[float] = [0.0]
        for i in range(prompt_len - 1):
            next_token = token_ids[i + 1]
            all_logprobs.append(prefix_log_probs[i, next_token].item())

        all_token_ids = outputs.sequences[0].tolist()
        generated_ids = all_token_ids[prompt_len:]

        for score, token_id in zip(outputs.scores, generated_ids):
            log_probs = torch.log_softmax(score[0], dim=-1)
            all_logprobs.append(log_probs[token_id].item())

        return all_token_ids, all_logprobs
