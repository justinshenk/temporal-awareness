"""HuggingFace Transformers backend implementation with intervention support."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .model_backend import Backend
from ..interventions import Intervention


class HuggingFaceBackend(Backend):
    """Backend using HuggingFace Transformers for model inference and interventions.

    This backend uses raw PyTorch hooks for activation caching and interventions.
    Supports per-head attention pattern capture and weight matrix access for
    Qwen3, Llama, and similar architectures.
    """

    def __init__(self, runner: Any, tokenizer: Any):
        super().__init__(runner)
        self._tokenizer = tokenizer
        config = self.runner._model.config

        # Detect model architecture and cache layer references
        if hasattr(self.runner._model, "transformer"):
            self._layers_attr = "transformer.h"
            self._layers = self.runner._model.transformer.h
            self._n_layers = len(self._layers)
            self._d_model = config.n_embd
            self._n_heads = config.n_head
            self._d_head = self._d_model // self._n_heads
            self._n_kv_heads = getattr(config, "n_head", self._n_heads)
            self._d_mlp = getattr(config, "n_inner", self._d_model * 4)
            self._arch_type = "gpt2"
        elif hasattr(self.runner._model, "gpt_neox"):
            self._layers_attr = "gpt_neox.layers"
            self._layers = self.runner._model.gpt_neox.layers
            self._n_layers = len(self._layers)
            self._d_model = config.hidden_size
            self._n_heads = config.num_attention_heads
            self._d_head = self._d_model // self._n_heads
            self._n_kv_heads = getattr(
                config, "num_key_value_heads", self._n_heads
            )
            self._d_mlp = getattr(config, "intermediate_size", self._d_model * 4)
            self._arch_type = "gpt_neox"
        elif hasattr(self.runner._model, "model") and hasattr(
            self.runner._model.model, "layers"
        ):
            self._layers_attr = "model.layers"
            self._layers = self.runner._model.model.layers
            self._n_layers = len(self._layers)
            self._d_model = config.hidden_size
            self._n_heads = config.num_attention_heads
            self._d_head = getattr(
                config, "head_dim", self._d_model // self._n_heads
            )
            self._n_kv_heads = getattr(
                config, "num_key_value_heads", self._n_heads
            )
            self._d_mlp = getattr(config, "intermediate_size", self._d_model * 4)
            self._arch_type = "llama"  # Covers Llama, Qwen3, Mistral, etc.
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner._model)}")

        # Calculate GQA group size
        self._n_kv_groups = self._n_heads // self._n_kv_heads

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

    def get_n_heads(self) -> int:
        return self._n_heads

    def get_d_head(self) -> int:
        return self._d_head

    def get_n_kv_heads(self) -> int:
        """Get the number of key-value heads (for GQA models)."""
        return self._n_kv_heads

    def get_d_mlp(self) -> int:
        """Get the MLP intermediate dimension."""
        return self._d_mlp

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

    def decode(self, token_ids: torch.Tensor | list) -> str:
        # Convert to list to avoid potential overflow issues with tensor dtypes
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.cpu().tolist()
        else:
            ids = token_ids
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def _get_component_module(self, layer_idx: int, component: str):
        """Get the module for a specific component within a layer.

        Args:
            layer_idx: Layer index (ignored for embed component)
            component: Component name (resid_post, attn_out, mlp_out, embed)
        """
        if component == "embed":
            return self._get_embed_tokens()

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
            eos_id = self._tokenizer.eos_token_id

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

    def _get_mlp_act_fn(self, layer_idx: int):
        """Get the activation function module from an MLP layer.

        Works with Qwen, Llama, and similar architectures that use
        gate_proj, up_proj, down_proj structure.
        """
        layer = self._layers[layer_idx]
        mlp = layer.mlp
        if hasattr(mlp, "act_fn"):
            return mlp.act_fn
        elif hasattr(mlp, "activation_fn"):
            return mlp.activation_fn
        return None

    def _get_attn_module(self, layer_idx: int):
        """Get the self-attention module for a layer."""
        layer = self._layers[layer_idx]
        if hasattr(layer, "self_attn"):
            return layer.self_attn  # Llama, Qwen3, Mistral
        elif hasattr(layer, "attn"):
            return layer.attn  # GPT-2
        elif hasattr(layer, "attention"):
            return layer.attention  # GPT-NeoX
        raise ValueError(f"Cannot find attention module in layer: {type(layer)}")

    def _register_attention_pattern_hook(
        self, attn_module, layer_idx: int, cache: dict, capture_z: bool = False
    ):
        """Register a hook to capture attention patterns and optionally hook_z.

        This manually computes attention patterns because SDPA doesn't support
        output_attentions=True. We hook into the attention module and compute:
        - attn_weights = softmax(Q @ K^T / sqrt(d_head))
        - hook_z = attn_weights @ V (if capture_z=True)

        For GQA models, K and V are expanded to match Q head count.

        Args:
            attn_module: The attention module to hook
            layer_idx: Layer index
            cache: Cache dict to store results
            capture_z: Whether to also capture hook_z (attention output before O projection)
        """

        def attention_hook(module, args, kwargs, output):
            # For Llama/Qwen3/Mistral style attention, we need to intercept
            # the hidden_states input and compute Q, K ourselves
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return

            batch_size, seq_len, _ = hidden_states.shape

            # Compute Q, K projections
            if hasattr(module, "q_proj"):
                # Llama/Qwen3/Mistral style
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)

                # Reshape to [batch, n_heads, seq, d_head]
                q = q.view(batch_size, seq_len, self._n_heads, self._d_head)
                q = q.transpose(1, 2)  # [batch, n_heads, seq, d_head]

                k = k.view(batch_size, seq_len, self._n_kv_heads, self._d_head)
                k = k.transpose(1, 2)  # [batch, n_kv_heads, seq, d_head]

                # Apply Q/K normalization if present (Qwen3 uses this)
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)

                # Expand K for GQA
                if self._n_kv_heads != self._n_heads:
                    k = k.repeat_interleave(self._n_kv_groups, dim=1)

                # Compute attention scores: Q @ K^T / sqrt(d_head)
                scale = self._d_head ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

                # Softmax to get attention probabilities
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(hidden_states.dtype)

                # Store attention pattern in cache
                cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_weights.detach()

                # Compute and store hook_z if requested
                if capture_z:
                    v = module.v_proj(hidden_states)
                    v = v.view(batch_size, seq_len, self._n_kv_heads, self._d_head)
                    v = v.transpose(1, 2)  # [batch, n_kv_heads, seq, d_head]

                    # Expand V for GQA
                    if self._n_kv_heads != self._n_heads:
                        v = v.repeat_interleave(self._n_kv_groups, dim=1)

                    # Compute attention output: attn_weights @ V
                    # [batch, n_heads, seq, seq] @ [batch, n_heads, seq, d_head]
                    # -> [batch, n_heads, seq, d_head]
                    attn_output = torch.matmul(attn_weights, v)

                    # Transpose to [batch, seq, n_heads, d_head] for TransformerLens compatibility
                    attn_output = attn_output.transpose(1, 2)

                    cache[f"blocks.{layer_idx}.attn.hook_z"] = attn_output.detach()

            elif hasattr(module, "c_attn"):
                # GPT-2 style combined projection
                qkv = module.c_attn(hidden_states)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(batch_size, seq_len, self._n_heads, self._d_head)
                q = q.transpose(1, 2)
                k = k.view(batch_size, seq_len, self._n_heads, self._d_head)
                k = k.transpose(1, 2)
                v = v.view(batch_size, seq_len, self._n_heads, self._d_head)
                v = v.transpose(1, 2)

                scale = self._d_head ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(hidden_states.dtype)

                cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_weights.detach()

                # Compute and store hook_z if requested
                if capture_z:
                    attn_output = torch.matmul(attn_weights, v)
                    attn_output = attn_output.transpose(1, 2)  # [batch, seq, n_heads, d_head]
                    cache[f"blocks.{layer_idx}.attn.hook_z"] = attn_output.detach()

        return attn_module.register_forward_hook(attention_hook, with_kwargs=True)

    def _create_attn_z_intervention_hook(
        self,
        attn_module,
        layer_idx: int,
        intervention: "Intervention",
    ):
        """Create a hook that intervenes on attn_z (attention output before O projection).

        This hook:
        1. Intercepts the attention module output
        2. Reconstructs attn_z from the output (reversing the O projection)
        3. Applies the intervention to the specified head at specified positions
        4. Re-projects through O to get the final output

        For efficiency, we use a mathematical approach: instead of fully reconstructing
        attn_z, we compute the difference in the final output that would result from
        modifying the specified head's attn_z values.

        Args:
            attn_module: The attention module to hook
            layer_idx: Layer index
            intervention: The intervention to apply
        """
        from ..interventions.intervention_base import Intervention

        head = intervention.head
        positions = list(intervention.target.positions) if intervention.target.positions else None
        mode = intervention.mode
        alpha = intervention.alpha

        # Get patch values
        values = torch.tensor(
            intervention.scaled_values,
            dtype=self.runner.dtype,
            device=self.runner.device,
        )
        target_values = None
        if mode == "interpolate" and intervention.target_values is not None:
            target_values = torch.tensor(
                intervention.target_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

        # Get W_O for this layer's head
        W_O = self.get_W_O(layer_idx)  # [n_heads, d_head, d_model]
        W_O_head = W_O[head]  # [d_head, d_model]

        def intervention_hook(module, args, kwargs, output):
            """Apply attn_z intervention by modifying the attention output.

            Strategy: Compute the delta contribution from the patched head and add it to output.
            delta = (patched_z - original_z) @ W_O[head]
            output += delta
            """
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return output

            # Extract the attention output from the module output
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            batch_size, seq_len, _ = hidden_states.shape

            # Compute original attn_z for this head
            # We need to recompute Q, K, V and attention
            if hasattr(module, "q_proj"):
                # Llama/Qwen3/Mistral style
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                v = module.v_proj(hidden_states)

                # Reshape to [batch, n_heads, seq, d_head]
                q = q.view(batch_size, seq_len, self._n_heads, self._d_head)
                q = q.transpose(1, 2)
                k = k.view(batch_size, seq_len, self._n_kv_heads, self._d_head)
                k = k.transpose(1, 2)
                v = v.view(batch_size, seq_len, self._n_kv_heads, self._d_head)
                v = v.transpose(1, 2)

                # Apply Q/K normalization if present
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)

                # Expand K, V for GQA
                if self._n_kv_heads != self._n_heads:
                    k = k.repeat_interleave(self._n_kv_groups, dim=1)
                    v = v.repeat_interleave(self._n_kv_groups, dim=1)

                # Compute attention weights
                scale = self._d_head ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(hidden_states.dtype)

                # Compute attn_z: [batch, n_heads, seq, d_head]
                attn_z = torch.matmul(attn_weights, v)

            elif hasattr(module, "c_attn"):
                # GPT-2 style
                qkv = module.c_attn(hidden_states)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(batch_size, seq_len, self._n_heads, self._d_head).transpose(1, 2)
                k = k.view(batch_size, seq_len, self._n_heads, self._d_head).transpose(1, 2)
                v = v.view(batch_size, seq_len, self._n_heads, self._d_head).transpose(1, 2)

                scale = self._d_head ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                    diagonal=1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(hidden_states.dtype)
                attn_z = torch.matmul(attn_weights, v)
            else:
                # Unsupported architecture
                return output

            # Get the original attn_z values for the target head
            # attn_z shape: [batch, n_heads, seq, d_head]
            original_z_head = attn_z[:, head, :, :]  # [batch, seq, d_head]

            # Apply intervention to get patched z values
            # Clone to avoid modifying original
            patched_z_head = original_z_head.clone()

            if positions is None:
                # All positions
                if mode == "add":
                    patched_z_head = patched_z_head + values
                elif mode == "set":
                    if values.ndim == 2:
                        seq = min(patched_z_head.shape[1], values.shape[0])
                        patched_z_head[:, :seq, :] = values[:seq].unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        patched_z_head = values.expand_as(patched_z_head)
                elif mode == "mul":
                    patched_z_head = patched_z_head * values
                elif mode == "interpolate" and target_values is not None:
                    if target_values.ndim == 2:
                        seq = min(patched_z_head.shape[1], target_values.shape[0])
                        tv = target_values[:seq].unsqueeze(0).expand(batch_size, -1, -1)
                        patched_z_head[:, :seq, :] = (
                            patched_z_head[:, :seq, :] + alpha * (tv - patched_z_head[:, :seq, :])
                        )
                    else:
                        patched_z_head = patched_z_head + alpha * (target_values - patched_z_head)
            else:
                # Specific positions
                for i, pos in enumerate(positions):
                    if pos < patched_z_head.shape[1]:
                        v = values[i] if values.ndim > 1 and i < values.shape[0] else values
                        if mode == "add":
                            patched_z_head[:, pos, :] = patched_z_head[:, pos, :] + v
                        elif mode == "set":
                            patched_z_head[:, pos, :] = v
                        elif mode == "mul":
                            patched_z_head[:, pos, :] = patched_z_head[:, pos, :] * v
                        elif mode == "interpolate" and target_values is not None:
                            tv = target_values[i] if target_values.ndim > 1 and i < target_values.shape[0] else target_values
                            patched_z_head[:, pos, :] = patched_z_head[:, pos, :] + alpha * (tv - patched_z_head[:, pos, :])

            # Compute the delta contribution: (patched - original) @ W_O[head]
            # original_z_head, patched_z_head: [batch, seq, d_head]
            # W_O_head: [d_head, d_model]
            delta_z = patched_z_head - original_z_head  # [batch, seq, d_head]
            delta_out = torch.matmul(delta_z, W_O_head)  # [batch, seq, d_model]

            # Add delta to output
            new_attn_out = attn_out + delta_out

            if isinstance(output, tuple):
                return (new_attn_out,) + output[1:]
            return new_attn_out

        return attn_module.register_forward_hook(intervention_hook, with_kwargs=True)

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
            # Standard component hooks
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

            # MLP neuron activation hook (mlp.hook_post)
            mlp_post_name = f"blocks.{i}.mlp.hook_post"
            if names_filter is None or names_filter(mlp_post_name):
                hooks_to_capture.append((i, "mlp_post", mlp_post_name))

            # Attention pattern hooks (TransformerLens-compatible naming)
            attn_pattern_name = f"blocks.{i}.attn.hook_pattern"
            if names_filter is None or names_filter(attn_pattern_name):
                hooks_to_capture.append((i, "attn_pattern", attn_pattern_name))

            # Per-head attention output (hook_z)
            attn_z_name = f"blocks.{i}.attn.hook_z"
            if names_filter is None or names_filter(attn_z_name):
                hooks_to_capture.append((i, "attn_z", attn_z_name))

        for layer_idx, component, name in hooks_to_capture:
            if component == "mlp_post":
                # Hook the activation function to get neuron activations
                module = self._get_mlp_act_fn(layer_idx)
                if module is None:
                    continue

                def make_hook(hook_name, use_input=False):
                    def hook_fn(mod, inp, out):
                        if use_input:
                            val = inp[0] if isinstance(inp, tuple) else inp
                        else:
                            val = out[0] if isinstance(out, tuple) else out
                        cache[hook_name] = val.detach()

                    return hook_fn

                hooks.append(module.register_forward_hook(make_hook(name, False)))

            elif component in ("attn_pattern", "attn_z"):
                # These are handled via output_attentions below
                pass

            else:
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

        # Check if we need attention patterns or hook_z
        need_attn_patterns = any(
            c in ("attn_pattern", "attn_z") for _, c, _ in hooks_to_capture
        )
        need_attn_z = any(c == "attn_z" for _, c, _ in hooks_to_capture)

        # Set up attention pattern hooks if needed
        # We hook into the attention module to capture Q, K and compute patterns manually
        # This is needed because SDPA doesn't support output_attentions
        attn_hooks = []
        if need_attn_patterns:
            for layer_idx in range(self._n_layers):
                pattern_name = f"blocks.{layer_idx}.attn.hook_pattern"
                z_name = f"blocks.{layer_idx}.attn.hook_z"
                # Check if this layer needs a hook
                need_pattern = names_filter is None or names_filter(pattern_name)
                need_z = need_attn_z and (names_filter is None or names_filter(z_name))
                if need_pattern or need_z:
                    attn_module = self._get_attn_module(layer_idx)
                    hook = self._register_attention_pattern_hook(
                        attn_module, layer_idx, cache, capture_z=need_z
                    )
                    attn_hooks.append(hook)

        try:
            with torch.no_grad():
                outputs = self.runner._model(input_ids)
            logits = outputs.logits

        finally:
            for hook in hooks:
                hook.remove()
            for hook in attn_hooks:
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
            # Standard component hooks
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

            # MLP neuron activation hook (mlp.hook_post)
            mlp_post_name = f"blocks.{i}.mlp.hook_post"
            if names_filter is None or names_filter(mlp_post_name):
                hooks_to_capture.append((i, "mlp_post", mlp_post_name))

            # Attention pattern hooks
            attn_pattern_name = f"blocks.{i}.attn.hook_pattern"
            if names_filter is None or names_filter(attn_pattern_name):
                hooks_to_capture.append((i, "attn_pattern", attn_pattern_name))

        for layer_idx, component, name in hooks_to_capture:
            if component == "mlp_post":
                module = self._get_mlp_act_fn(layer_idx)
                if module is None:
                    continue

                def make_hook(hook_name):
                    def hook_fn(mod, inp, out):
                        val = out[0] if isinstance(out, tuple) else out
                        cache[hook_name] = val

                    return hook_fn

                hooks.append(module.register_forward_hook(make_hook(name)))

            elif component == "attn_pattern":
                # Handled via output_attentions
                pass

            else:
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

        # Check if we need attention patterns
        need_attn_patterns = any(c == "attn_pattern" for _, c, _ in hooks_to_capture)

        # Set up attention pattern hooks if needed
        attn_hooks = []
        if need_attn_patterns:
            for layer_idx in range(self._n_layers):
                pattern_name = f"blocks.{layer_idx}.attn.hook_pattern"
                if names_filter is None or names_filter(pattern_name):
                    attn_module = self._get_attn_module(layer_idx)
                    hook = self._register_attention_pattern_hook(
                        attn_module, layer_idx, cache
                    )
                    attn_hooks.append(hook)

        try:
            outputs = self.runner._model(input_ids)
            logits = outputs.logits

        finally:
            for hook in hooks:
                hook.remove()
            for hook in attn_hooks:
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
        hooks = []
        for intervention in interventions:
            # Handle attn_z interventions specially
            if intervention.component == "attn_z":
                attn_module = self._get_attn_module(intervention.layer)
                hook = self._create_attn_z_intervention_hook(
                    attn_module, intervention.layer, intervention
                )
                hooks.append(hook)
                continue

            values = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            target = intervention.target
            mode = intervention.mode
            alpha = intervention.alpha
            target_values = None
            if mode == "interpolate" and intervention.target_values is not None:
                target_values = torch.tensor(
                    intervention.target_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
            module = self._get_component_module(
                intervention.layer, intervention.component
            )

            def make_hook(values, target, mode, target_values, alpha):
                def intervention_hook(mod, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if target.is_all_positions:
                        if mode == "add":
                            hidden = hidden + values
                        elif mode == "set":
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
                            if target_values is not None and target_values.ndim == 2:
                                seq_len = min(hidden.shape[1], target_values.shape[0])
                                new_hidden = hidden.clone()
                                tv = (
                                    target_values[:seq_len]
                                    .unsqueeze(0)
                                    .expand(hidden.shape[0], -1, -1)
                                )
                                new_hidden[:, :seq_len, :] = (
                                    hidden[:, :seq_len, :]
                                    + alpha * (tv - hidden[:, :seq_len, :])
                                )
                                hidden = new_hidden
                            elif target_values is not None:
                                hidden = hidden + alpha * (target_values - hidden)
                    else:
                        for i, pos in enumerate(target.positions):
                            if pos < hidden.shape[1]:
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
                                    if target_values is not None:
                                        tv = (
                                            target_values[i]
                                            if target_values.ndim > 1
                                            and i < len(target_values)
                                            else target_values
                                        )
                                        hidden[:, pos, :] = hidden[:, pos, :] + alpha * (
                                            tv - hidden[:, pos, :]
                                        )

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

                return intervention_hook

            hook = module.register_forward_hook(
                make_hook(values, target, mode, target_values, alpha)
            )
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

        # Set up intervention hooks
        for intervention in interventions:
            # Handle attn_z interventions specially
            if intervention.component == "attn_z":
                attn_module = self._get_attn_module(intervention.layer)
                hook = self._create_attn_z_intervention_hook(
                    attn_module, intervention.layer, intervention
                )
                hooks.append(hook)
                continue

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

    def _get_embed_tokens(self):
        """Get the token embedding module."""
        model = self.runner._model
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens  # Llama, Mistral, Qwen
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte  # GPT-2
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
            return model.gpt_neox.embed_in  # GPT-NeoX
        raise ValueError(f"Cannot find embedding module for: {type(model)}")

    def _get_lm_head(self):
        """Get the language model head module."""
        model = self.runner._model
        if hasattr(model, "lm_head"):
            return model.lm_head  # Most models
        if hasattr(model, "embed_out"):
            return model.embed_out  # GPT-NeoX
        raise ValueError(f"Cannot find lm_head for: {type(model)}")

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
        return lm_head.weight.T

    def get_b_U(self) -> torch.Tensor | None:
        """Get the unembedding bias b_U.

        Returns:
            Unembedding bias of shape [vocab_size], or None if no bias
        """
        lm_head = self._get_lm_head()
        return getattr(lm_head, "bias", None)

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the model.

        For models with learned position embeddings (GPT-2), this includes
        position embeddings. For models with RoPE (Llama, Mistral), this
        returns just token embeddings (RoPE is applied in attention).

        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, d_model]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        token_ids = token_ids.to(self.runner.device)
        embed_module = self._get_embed_tokens()

        with torch.no_grad():
            # Get token embeddings
            embeds = embed_module(token_ids)

            # Add position embeddings for GPT-2 style models
            model = self.runner._model
            if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
                seq_len = token_ids.shape[1]
                position_ids = torch.arange(seq_len, device=self.runner.device)
                pos_embeds = model.transformer.wpe(position_ids)
                embeds = embeds + pos_embeds

        return embeds

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

            # Compute logprobs for prefilled tokens via forward pass
            prefix_outputs = self.runner._model(input_ids)
            prefix_logits = prefix_outputs.logits[0]
            prefix_log_probs = torch.log_softmax(prefix_logits, dim=-1)

        # For position i, get logprob of token[i+1]
        all_logprobs: list[float] = [0.0]  # First token has no prior context
        for i in range(prompt_len - 1):
            next_token = token_ids[i + 1]
            all_logprobs.append(prefix_log_probs[i, next_token].item())

        all_token_ids = outputs.sequences[0].tolist()
        generated_ids = all_token_ids[prompt_len:]

        # Append logprobs for generated tokens from scores
        for score, token_id in zip(outputs.scores, generated_ids):
            log_probs = torch.log_softmax(score[0], dim=-1)
            all_logprobs.append(log_probs[token_id].item())

        return all_token_ids, all_logprobs

    def _get_qkv_proj_weights(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get Q, K, V projection weights for a layer.

        Returns weights in TransformerLens format: [n_heads, d_model, d_head]
        Handles both separate q/k/v projections and combined qkv projections.
        Also handles GQA where K/V may have fewer heads than Q.
        """
        attn = self._get_attn_module(layer_idx)

        if self._arch_type == "llama":
            # Llama, Qwen3, Mistral use separate projections
            # q_proj: [n_heads * d_head, d_model]
            # k_proj: [n_kv_heads * d_head, d_model]
            # v_proj: [n_kv_heads * d_head, d_model]
            q_weight = attn.q_proj.weight  # [n_heads * d_head, d_model]
            k_weight = attn.k_proj.weight  # [n_kv_heads * d_head, d_model]
            v_weight = attn.v_proj.weight  # [n_kv_heads * d_head, d_model]

            # Reshape to [n_heads, d_head, d_model] then transpose to [n_heads, d_model, d_head]
            W_Q = q_weight.view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)
            W_K = k_weight.view(self._n_kv_heads, self._d_head, self._d_model).transpose(1, 2)
            W_V = v_weight.view(self._n_kv_heads, self._d_head, self._d_model).transpose(1, 2)

            # Expand K, V for GQA to match Q head count
            if self._n_kv_heads != self._n_heads:
                W_K = W_K.repeat_interleave(self._n_kv_groups, dim=0)
                W_V = W_V.repeat_interleave(self._n_kv_groups, dim=0)

            return W_Q, W_K, W_V

        elif self._arch_type == "gpt2":
            # GPT-2 uses combined c_attn projection
            # c_attn.weight: [d_model, 3 * n_heads * d_head]
            c_attn_weight = attn.c_attn.weight  # [d_model, 3 * d_model]
            qkv = c_attn_weight.chunk(3, dim=1)  # 3 x [d_model, n_heads * d_head]

            W_Q = qkv[0].T.view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)
            W_K = qkv[1].T.view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)
            W_V = qkv[2].T.view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)

            return W_Q, W_K, W_V

        elif self._arch_type == "gpt_neox":
            # GPT-NeoX uses combined query_key_value projection
            qkv_weight = attn.query_key_value.weight  # [3 * n_heads * d_head, d_model]
            qkv = qkv_weight.chunk(3, dim=0)  # 3 x [n_heads * d_head, d_model]

            W_Q = qkv[0].view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)
            W_K = qkv[1].view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)
            W_V = qkv[2].view(self._n_heads, self._d_head, self._d_model).transpose(1, 2)

            return W_Q, W_K, W_V

        raise ValueError(f"Unknown architecture type: {self._arch_type}")

    def _get_o_proj_weight(self, layer_idx: int) -> torch.Tensor:
        """Get output projection weight for a layer.

        Returns weight in TransformerLens format: [n_heads, d_head, d_model]
        """
        attn = self._get_attn_module(layer_idx)

        if self._arch_type == "llama":
            # o_proj: [d_model, n_heads * d_head]
            o_weight = attn.o_proj.weight  # [d_model, n_heads * d_head]
            # Reshape to [d_model, n_heads, d_head] then permute to [n_heads, d_head, d_model]
            W_O = o_weight.view(self._d_model, self._n_heads, self._d_head)
            W_O = W_O.permute(1, 2, 0)  # [n_heads, d_head, d_model]
            return W_O

        elif self._arch_type == "gpt2":
            # c_proj: [n_heads * d_head, d_model]
            c_proj_weight = attn.c_proj.weight  # [n_heads * d_head, d_model]
            W_O = c_proj_weight.view(self._n_heads, self._d_head, self._d_model)
            return W_O

        elif self._arch_type == "gpt_neox":
            # dense: [d_model, n_heads * d_head]
            dense_weight = attn.dense.weight  # [d_model, n_heads * d_head]
            W_O = dense_weight.view(self._d_model, self._n_heads, self._d_head)
            W_O = W_O.permute(1, 2, 0)  # [n_heads, d_head, d_model]
            return W_O

        raise ValueError(f"Unknown architecture type: {self._arch_type}")

    def get_W_Q(self, layer: int | None = None) -> torch.Tensor:
        """Get query weight matrix W_Q.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]
        """
        if layer is not None:
            W_Q, _, _ = self._get_qkv_proj_weights(layer)
            return W_Q

        all_W_Q = []
        for i in range(self._n_layers):
            W_Q, _, _ = self._get_qkv_proj_weights(i)
            all_W_Q.append(W_Q)
        return torch.stack(all_W_Q)

    def get_W_K(self, layer: int | None = None) -> torch.Tensor:
        """Get key weight matrix W_K.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]
        """
        if layer is not None:
            _, W_K, _ = self._get_qkv_proj_weights(layer)
            return W_K

        all_W_K = []
        for i in range(self._n_layers):
            _, W_K, _ = self._get_qkv_proj_weights(i)
            all_W_K.append(W_K)
        return torch.stack(all_W_K)

    def get_W_V(self, layer: int | None = None) -> torch.Tensor:
        """Get value weight matrix W_V.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_model, d_head]
            If layer specified: [n_heads, d_model, d_head]
        """
        if layer is not None:
            _, _, W_V = self._get_qkv_proj_weights(layer)
            return W_V

        all_W_V = []
        for i in range(self._n_layers):
            _, _, W_V = self._get_qkv_proj_weights(i)
            all_W_V.append(W_V)
        return torch.stack(all_W_V)

    def get_W_O(self, layer: int | None = None) -> torch.Tensor:
        """Get output weight matrix W_O.

        Args:
            layer: Layer index, or None for all layers

        Returns:
            If layer is None: [n_layers, n_heads, d_head, d_model]
            If layer specified: [n_heads, d_head, d_model]
        """
        if layer is not None:
            return self._get_o_proj_weight(layer)

        all_W_O = []
        for i in range(self._n_layers):
            W_O = self._get_o_proj_weight(i)
            all_W_O.append(W_O)
        return torch.stack(all_W_O)

    def get_W_OV(self, layer: int, head: int) -> torch.Tensor:
        """Get combined OV matrix for a specific head.

        W_OV = W_V @ W_O projects input through value and output matrices.

        Args:
            layer: Layer index
            head: Head index

        Returns:
            W_OV matrix of shape [d_model, d_model]
        """
        _, _, W_V = self._get_qkv_proj_weights(layer)
        W_O = self._get_o_proj_weight(layer)

        # W_V[head]: [d_model, d_head]
        # W_O[head]: [d_head, d_model]
        return W_V[head] @ W_O[head]  # [d_model, d_model]

    def get_W_QK(self, layer: int, head: int) -> torch.Tensor:
        """Get combined QK matrix for a specific head.

        W_QK = W_Q @ W_K^T determines attention pattern computation.

        Args:
            layer: Layer index
            head: Head index

        Returns:
            W_QK matrix of shape [d_model, d_model]
        """
        W_Q, W_K, _ = self._get_qkv_proj_weights(layer)

        # W_Q[head]: [d_model, d_head]
        # W_K[head]: [d_model, d_head]
        return W_Q[head] @ W_K[head].T  # [d_model, d_model]
