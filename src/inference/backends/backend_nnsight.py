"""NNsight backend implementation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .model_backend import Backend
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

        # Cache the actual device from the model (handles device_map="auto")
        self._device = next(model._model.parameters()).device

    @property
    def device(self) -> torch.device:
        """Get the actual device the model is on (handles device_map='auto')."""
        return self._device

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
        return ids.to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text with optional interventions using native NNsight generate."""
        input_ids = self.encode(prompt)
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
                device=self.device,
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
        input_ids = self.encode(prompt)
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
        input_ids = self.encode(prompt)
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
        if component in ("resid_post", "resid_pre"):
            # resid_pre: input to the layer
            # resid_post: output of the layer
            return layer
        elif component == "resid_mid":
            # resid_mid: residual stream after attention, before MLP
            # This is the INPUT to post_attention_layernorm
            # Llama/Mistral: post_attention_layernorm
            # GPT-2: ln_2
            if hasattr(layer, "post_attention_layernorm"):
                return layer.post_attention_layernorm
            elif hasattr(layer, "ln_2"):
                return layer.ln_2
            else:
                raise ValueError(
                    f"Cannot find post-attention layernorm in layer: {type(layer)}. "
                    "resid_mid requires a model with post_attention_layernorm (Llama/Mistral) "
                    "or ln_2 (GPT-2)."
                )
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

        # Determine which hooks to capture
        hooks_to_capture = set()
        attn_pattern_layers = set()
        attn_out_layers = set()

        for i in range(len(self._layers)):
            # Standard component hooks
            for component in ["resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.add((i, component, name))
                    if component == "attn_out":
                        attn_out_layers.add(i)

            # Attention pattern hooks (hook_attn and hook_pattern are synonyms)
            for pattern_name in [
                f"blocks.{i}.attn.hook_attn",
                f"blocks.{i}.attn.hook_pattern",
            ]:
                if names_filter is not None and names_filter(pattern_name):
                    attn_pattern_layers.add(i)

        # Use output_attentions=True if we need attention patterns
        trace_kwargs = {}
        if attn_pattern_layers:
            trace_kwargs["output_attentions"] = True

        with self.runner._model.trace(input_ids, **trace_kwargs):
            for layer_idx in range(len(self._layers)):
                layer = self._get_layer(layer_idx)

                # Handle attention output and patterns together to avoid access conflicts
                needs_attn_out = layer_idx in attn_out_layers
                needs_attn_pattern = layer_idx in attn_pattern_layers

                if needs_attn_out or needs_attn_pattern:
                    attn_module = self._get_component_module(layer, "attn_out")
                    if needs_attn_pattern:
                        # With output_attentions=True, output is (hidden_states, attn_weights, ...)
                        # Save both at once to avoid access order issues
                        attn_hidden = attn_module.output[0].save()
                        attn_weights = attn_module.output[1].save()
                        if needs_attn_out:
                            cache[f"blocks.{layer_idx}.hook_attn_out"] = attn_hidden
                        # Store patterns under both names for compatibility
                        cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_weights
                        cache[f"blocks.{layer_idx}.attn.hook_attn"] = attn_weights
                    elif needs_attn_out:
                        out = attn_module.output[0].save()
                        cache[f"blocks.{layer_idx}.hook_attn_out"] = out

                # Capture other components
                for component in ["resid_pre", "resid_mid", "mlp_out", "resid_post"]:
                    name = f"blocks.{layer_idx}.hook_{component}"
                    if (layer_idx, component, name) not in hooks_to_capture:
                        continue

                    module = self._get_component_module(layer, component)

                    # resid_pre: input to layer, resid_mid: input to post_attention_layernorm
                    # Both need to capture the INPUT to their respective modules
                    if component in ("resid_pre", "resid_mid"):
                        out = module.input[0].save()
                    elif component == "mlp_out":
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
                    device=self.device,
                )
                target = intervention.target
                mode = intervention.mode
                alpha = intervention.alpha
                component = intervention.component

                target_values = None
                if mode == "interpolate" and intervention.target_values is not None:
                    target_values = torch.tensor(
                        intervention.target_values,
                        dtype=self.runner.dtype,
                        device=self.device,
                    )

                if component == "mlp_out":
                    out = module.output
                else:
                    out = module.output[0]

                if target.is_all_positions:
                    if mode == "add":
                        out[:, :] += values
                    elif mode == "set":
                        out[:, :] = values
                    elif mode == "mul":
                        out[:, :] *= values
                    elif mode == "interpolate" and target_values is not None:
                        # Handle sequence length mismatch for 2D target_values
                        if target_values.ndim == 2:
                            if out.ndim == 3:
                                seq_len = min(out.shape[1], target_values.shape[0])
                                tv = target_values[:seq_len].unsqueeze(0)
                                out[:, :seq_len, :] = out[:, :seq_len, :] + alpha * (tv - out[:, :seq_len, :])
                            else:
                                # out is 2D: [seq_len, d_model]
                                seq_len = min(out.shape[0], target_values.shape[0])
                                out[:seq_len, :] = out[:seq_len, :] + alpha * (target_values[:seq_len] - out[:seq_len, :])
                        else:
                            out[:, :] = out[:, :] + alpha * (target_values - out[:, :])
                else:
                    seq_len = out.shape[0] if out.ndim == 2 else out.shape[1]
                    for i, pos in enumerate(target.positions):
                        if pos < seq_len:
                            tv = None
                            if target_values is not None:
                                tv = (
                                    target_values[i]
                                    if target_values.ndim > 1 and i < len(target_values)
                                    else target_values
                                )
                            if out.ndim == 2:
                                if mode == "add":
                                    out[pos, :] += values
                                elif mode == "set":
                                    out[pos, :] = values
                                elif mode == "mul":
                                    out[pos, :] *= values
                                elif mode == "interpolate" and tv is not None:
                                    out[pos, :] = out[pos, :] + alpha * (tv - out[pos, :])
                            else:
                                if mode == "add":
                                    out[:, pos, :] += values
                                elif mode == "set":
                                    out[:, pos, :] = values
                                elif mode == "mul":
                                    out[:, pos, :] *= values
                                elif mode == "interpolate" and tv is not None:
                                    out[:, pos, :] = out[:, pos, :] + alpha * (tv - out[:, pos, :])

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

                for component in ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]:
                    key = (layer_idx, component)
                    intervention = intervention_lookup.get(key)

                    should_cache = (
                        layer_idx in layers_to_capture and component == "resid_post"
                    )

                    if intervention is None and not should_cache:
                        continue

                    module = self._get_component_module(layer, component)

                    # resid_pre: input to layer, resid_mid: input to post_attention_layernorm
                    # Both need to access the INPUT to their respective modules
                    if component in ("resid_pre", "resid_mid"):
                        out = module.input[0]
                    elif component == "mlp_out":
                        out = module.output
                    else:
                        out = module.output[0]

                    if intervention is not None:
                        values = torch.tensor(
                            intervention.scaled_values,
                            dtype=self.runner.dtype,
                            device=self.device,
                        )
                        target = intervention.target
                        mode = intervention.mode
                        alpha = intervention.alpha

                        target_values = None
                        if mode == "interpolate" and intervention.target_values is not None:
                            target_values = torch.tensor(
                                intervention.target_values,
                                dtype=self.runner.dtype,
                                device=self.device,
                            )

                        if target.is_all_positions:
                            if mode == "add":
                                out[:, :] += values
                            elif mode == "set":
                                out[:, :] = values
                            elif mode == "mul":
                                out[:, :] *= values
                            elif mode == "interpolate" and target_values is not None:
                                out[:, :] = out[:, :] + alpha * (target_values - out[:, :])
                        else:
                            seq_len = out.shape[0] if out.ndim == 2 else out.shape[1]
                            for i, pos in enumerate(target.positions):
                                if pos < seq_len:
                                    tv = None
                                    if target_values is not None:
                                        tv = (
                                            target_values[i]
                                            if target_values.ndim > 1 and i < len(target_values)
                                            else target_values
                                        )
                                    if out.ndim == 2:
                                        if mode == "add":
                                            out[pos, :] += values
                                        elif mode == "set":
                                            out[pos, :] = values
                                        elif mode == "mul":
                                            out[pos, :] *= values
                                        elif mode == "interpolate" and tv is not None:
                                            out[pos, :] = out[pos, :] + alpha * (tv - out[pos, :])
                                    else:
                                        if mode == "add":
                                            out[:, pos, :] += values
                                        elif mode == "set":
                                            out[:, pos, :] = values
                                        elif mode == "mul":
                                            out[:, pos, :] *= values
                                        elif mode == "interpolate" and tv is not None:
                                            out[:, pos, :] = out[:, pos, :] + alpha * (tv - out[:, pos, :])

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

    def _get_embed_tokens(self):
        """Get the token embedding module."""
        model = self.runner._model
        if self._layers_path == "transformer.h":
            return model.transformer.wte  # GPT-2
        elif self._layers_path == "gpt_neox.layers":
            return model.gpt_neox.embed_in  # Pythia / GPT-NeoX
        else:
            return model.model.embed_tokens  # Llama / Mistral / Qwen

    def get_W_E(self) -> torch.Tensor:
        """Get the token embedding matrix W_E.

        Returns:
            Embedding matrix of shape [vocab_size, d_model]
        """
        embed = self._get_embed_tokens()
        return embed.weight

    def get_W_U(self) -> torch.Tensor:
        """Get the unembedding matrix W_U.

        Returns:
            Unembedding matrix of shape [d_model, vocab_size]
        """
        lm_head = self._get_lm_head()
        # lm_head.weight is [vocab_size, d_model], we need [d_model, vocab_size]
        return lm_head.weight.T

    def get_b_U(self) -> torch.Tensor | None:
        """Get the unembedding bias b_U.

        Returns:
            Unembedding bias of shape [vocab_size], or None if no bias
        """
        lm_head = self._get_lm_head()
        return getattr(lm_head, "bias", None)

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Not implemented for NNsight backend.

        NNsight wraps models with a tracing API that doesn't expose HuggingFace's
        KV cache. The underlying model uses meta tensors for lazy loading, making
        direct access to generate() impossible. Using nnsight's trace() for each
        token is actually slower than the baseline due to tracing overhead.

        For efficient trajectory generation, use HuggingFace or MLX backends.
        """
        raise NotImplementedError(
            "generate_trajectory not supported for NNsight backend. "
            "NNsight's tracing API doesn't expose KV cache for efficient generation. "
            "Use HuggingFace or MLX backend for trajectory generation."
        )

    # ── Model Architecture Info ─────────────────────────────────────────────

    def get_n_heads(self) -> int:
        """Get the number of attention heads per layer."""
        return self.runner._model.config.num_attention_heads

    def get_n_kv_heads(self) -> int:
        """Get the number of key-value heads (for GQA models)."""
        config = self.runner._model.config
        return getattr(config, "num_key_value_heads", config.num_attention_heads)

    def get_d_head(self) -> int:
        """Get the dimension of each attention head."""
        config = self.runner._model.config
        # Some models have head_dim in config, others compute from hidden_size
        if hasattr(config, "head_dim") and config.head_dim:
            return config.head_dim
        return config.hidden_size // config.num_attention_heads

    def get_d_mlp(self) -> int:
        """Get the MLP intermediate dimension."""
        return self.runner._model.config.intermediate_size

    # ── Attention Module Access ─────────────────────────────────────────────

    def _get_attn_module(self, layer_idx: int):
        """Get the attention module for a layer."""
        layer = self._layers[layer_idx]
        if self._layers_path == "transformer.h":
            return layer.attn  # GPT-2
        elif self._layers_path == "gpt_neox.layers":
            return layer.attention  # Pythia / GPT-NeoX
        else:
            return layer.self_attn  # Llama / Mistral / Qwen

    def _get_mlp_module(self, layer_idx: int):
        """Get the MLP module for a layer."""
        return self._layers[layer_idx].mlp

    # ── Weight Matrix Accessors ─────────────────────────────────────────────

    def get_W_Q(self, layer: int | None = None) -> torch.Tensor:
        """Get query weight matrix W_Q.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]

        Note: For GQA models, returns full n_heads (not n_kv_heads).
        """
        n_heads = self.get_n_heads()
        d_head = self.get_d_head()

        if layer is not None:
            attn = self._get_attn_module(layer)
            # q_proj.weight is [n_heads * d_head, d_model]
            W_Q_raw = attn.q_proj.weight
            # Reshape to [n_heads, d_head, d_model] then transpose to [n_heads, d_model, d_head]
            return W_Q_raw.view(n_heads, d_head, -1).transpose(1, 2)

        # All layers
        n_layers = self.get_n_layers()
        W_Q_all = []
        for i in range(n_layers):
            W_Q_all.append(self.get_W_Q(i))
        return torch.stack(W_Q_all)

    def get_W_K(self, layer: int | None = None) -> torch.Tensor:
        """Get key weight matrix W_K.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]

        Note: For GQA models, K weights are expanded from n_kv_heads to n_heads
              to maintain consistent API with TransformerLens.
        """
        n_heads = self.get_n_heads()
        n_kv_heads = self.get_n_kv_heads()
        d_head = self.get_d_head()
        n_groups = n_heads // n_kv_heads

        if layer is not None:
            attn = self._get_attn_module(layer)
            # k_proj.weight is [n_kv_heads * d_head, d_model]
            W_K_raw = attn.k_proj.weight
            # Reshape to [n_kv_heads, d_head, d_model] then transpose to [n_kv_heads, d_model, d_head]
            W_K = W_K_raw.view(n_kv_heads, d_head, -1).transpose(1, 2)
            # Expand to full n_heads for GQA
            if n_groups > 1:
                W_K = W_K.repeat_interleave(n_groups, dim=0)
            return W_K

        # All layers
        n_layers = self.get_n_layers()
        W_K_all = []
        for i in range(n_layers):
            W_K_all.append(self.get_W_K(i))
        return torch.stack(W_K_all)

    def get_W_V(self, layer: int | None = None) -> torch.Tensor:
        """Get value weight matrix W_V.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]

        Note: For GQA models, V weights are expanded from n_kv_heads to n_heads
              to maintain consistent API with TransformerLens.
        """
        n_heads = self.get_n_heads()
        n_kv_heads = self.get_n_kv_heads()
        d_head = self.get_d_head()
        n_groups = n_heads // n_kv_heads

        if layer is not None:
            attn = self._get_attn_module(layer)
            # v_proj.weight is [n_kv_heads * d_head, d_model]
            W_V_raw = attn.v_proj.weight
            # Reshape to [n_kv_heads, d_head, d_model] then transpose to [n_kv_heads, d_model, d_head]
            W_V = W_V_raw.view(n_kv_heads, d_head, -1).transpose(1, 2)
            # Expand to full n_heads for GQA
            if n_groups > 1:
                W_V = W_V.repeat_interleave(n_groups, dim=0)
            return W_V

        # All layers
        n_layers = self.get_n_layers()
        W_V_all = []
        for i in range(n_layers):
            W_V_all.append(self.get_W_V(i))
        return torch.stack(W_V_all)

    def get_W_O(self, layer: int | None = None) -> torch.Tensor:
        """Get output weight matrix W_O.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_head, d_model]
            If layer specified: [n_heads, d_head, d_model]
        """
        n_heads = self.get_n_heads()
        d_head = self.get_d_head()
        d_model = self.get_d_model()

        if layer is not None:
            attn = self._get_attn_module(layer)
            # o_proj.weight is [d_model, n_heads * d_head]
            W_O_raw = attn.o_proj.weight
            # Reshape to [d_model, n_heads, d_head] then permute to [n_heads, d_head, d_model]
            return W_O_raw.view(d_model, n_heads, d_head).permute(1, 2, 0)

        # All layers
        n_layers = self.get_n_layers()
        W_O_all = []
        for i in range(n_layers):
            W_O_all.append(self.get_W_O(i))
        return torch.stack(W_O_all)

    def get_W_OV(self, layer: int, head: int) -> torch.Tensor:
        """Get combined OV matrix for a specific head.

        W_OV = W_V @ W_O projects input through value and output matrices.

        Args:
            layer: Layer index
            head: Head index

        Returns:
            W_OV matrix of shape [d_model, d_model]
        """
        W_V = self.get_W_V(layer)[head]  # [d_model, d_head]
        W_O = self.get_W_O(layer)[head]  # [d_head, d_model]
        return W_V @ W_O  # [d_model, d_model]

    def get_W_QK(self, layer: int, head: int) -> torch.Tensor:
        """Get combined QK matrix for a specific head.

        W_QK = W_Q @ W_K^T determines attention pattern computation.

        Args:
            layer: Layer index
            head: Head index

        Returns:
            W_QK matrix of shape [d_model, d_model]
        """
        W_Q = self.get_W_Q(layer)[head]  # [d_model, d_head]
        W_K = self.get_W_K(layer)[head]  # [d_model, d_head]
        return W_Q @ W_K.T  # [d_model, d_model]

    # ── MLP Weight Accessors ────────────────────────────────────────────────

    def get_MLP_W_in(self, layer: int) -> torch.Tensor:
        """Get MLP input projection weights.

        For SwiGLU/gated MLPs (Llama, Qwen, etc.), returns gate_proj weights.
        For standard MLPs (GPT-2), returns W_in (fc1) weights.

        Returns:
            W_in of shape [d_model, d_mlp]
        """
        mlp = self._get_mlp_module(layer)
        if hasattr(mlp, "gate_proj"):
            # SwiGLU: gate_proj.weight is [d_mlp, d_model]
            return mlp.gate_proj.weight.T  # [d_model, d_mlp]
        elif hasattr(mlp, "c_fc"):
            # GPT-2: c_fc.weight is [d_model, d_mlp] (Conv1D)
            return mlp.c_fc.weight  # [d_model, d_mlp]
        elif hasattr(mlp, "dense_h_to_4h"):
            # GPT-NeoX: dense_h_to_4h.weight is [d_mlp, d_model]
            return mlp.dense_h_to_4h.weight.T  # [d_model, d_mlp]
        else:
            raise ValueError(f"Unknown MLP architecture: {type(mlp)}")

    def get_MLP_W_out(self, layer: int) -> torch.Tensor:
        """Get MLP output projection weights.

        Returns:
            W_out of shape [d_mlp, d_model]
        """
        mlp = self._get_mlp_module(layer)
        if hasattr(mlp, "down_proj"):
            # SwiGLU: down_proj.weight is [d_model, d_mlp]
            return mlp.down_proj.weight.T  # [d_mlp, d_model]
        elif hasattr(mlp, "c_proj"):
            # GPT-2: c_proj.weight is [d_mlp, d_model] (Conv1D)
            return mlp.c_proj.weight.T  # [d_mlp, d_model]
        elif hasattr(mlp, "dense_4h_to_h"):
            # GPT-NeoX: dense_4h_to_h.weight is [d_model, d_mlp]
            return mlp.dense_4h_to_h.weight  # [d_mlp, d_model] (transposed)
        else:
            raise ValueError(f"Unknown MLP architecture: {type(mlp)}")

    # ── Attention Pattern Capture ───────────────────────────────────────────

    def run_with_cache_attn_patterns(
        self,
        input_ids: torch.Tensor,
        layers: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward pass and capture attention patterns.

        Requires the model to be loaded with attn_implementation="eager".
        SDPA attention doesn't support outputting attention weights.

        Args:
            input_ids: Token IDs tensor [batch, seq_len]
            layers: Which layers to capture (None = all layers)

        Returns:
            Tuple of (logits, cache) where cache contains:
            - "blocks.{i}.attn.hook_pattern": [batch, n_heads, seq_q, seq_k]
        """
        if layers is None:
            layers = list(range(len(self._layers)))

        cache = {}

        with self.runner._model.trace(input_ids, output_attentions=True):
            for layer_idx in layers:
                layer = self._get_layer(layer_idx)
                attn_module = self._get_component_module(layer, "attn_out")
                # With output_attentions=True, self_attn.output is (hidden_states, attn_weights, ...)
                attn_weights = attn_module.output[1].save()
                cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_weights

            logits = self._get_lm_head().output.save()

        # Post-process
        result_cache = {}
        for k, v in cache.items():
            result_cache[k] = v.detach()

        return logits.detach(), result_cache

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the model.

        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, d_model]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        token_ids = token_ids.to(self.device)
        embed_module = self._get_embed_tokens()

        with torch.no_grad():
            embeds = embed_module(token_ids)

        return embeds
