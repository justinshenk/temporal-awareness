"""MLX backend implementation for Apple Silicon."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .model_backend import Backend
from ..interventions import Intervention


def _get_mx():
    """Lazy import of mlx.core."""
    import mlx.core as mx

    return mx


def _get_generate():
    """Lazy import of mlx_lm generate."""
    from mlx_lm import generate

    return generate


class MLXBackend(Backend):
    """Backend using MLX for Apple Silicon inference."""

    def __init__(self, runner: Any, tokenizer: Any):
        """Initialize MLX backend.

        Args:
            runner: ModelRunner instance
            tokenizer: Tokenizer loaded from mlx_lm
        """
        super().__init__(runner)
        # MLX wraps HuggingFace tokenizer; store underlying one for full API compatibility
        if hasattr(tokenizer, "_tokenizer"):
            self._tokenizer = tokenizer._tokenizer
        else:
            self._tokenizer = tokenizer

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return len(self.runner._model.layers)

    def get_d_model(self) -> int:
        args = self.runner._model.args
        # Try common attribute names for hidden dimension
        if hasattr(args, "hidden_size"):
            return args.hidden_size
        if hasattr(args, "dim"):
            return args.dim
        if hasattr(args, "d_model"):
            return args.d_model
        # Some models (e.g., Gemma 3n) nest config in text_config
        if hasattr(args, "text_config"):
            text_cfg = args.text_config
            if isinstance(text_cfg, dict):
                return text_cfg.get("hidden_size") or text_cfg.get("dim")
            elif hasattr(text_cfg, "hidden_size"):
                return text_cfg.hidden_size
        raise AttributeError(f"Cannot find hidden size in model args: {args}")

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self.runner._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        if prepend_bos and self.runner._tokenizer.bos_token_id is not None:
            tokens = [self.runner._tokenizer.bos_token_id] + tokens
        return torch.tensor([tokens])

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if (
            isinstance(token_ids, list)
            and len(token_ids) > 0
            and isinstance(token_ids[0], list)
        ):
            token_ids = token_ids[0]
        return self.runner._tokenizer.decode(token_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        mx = _get_mx()

        if intervention is None:
            # Fast path: use native mlx_lm generate
            generate = _get_generate()

            if temperature > 0:
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature)
            else:
                sampler = None

            return generate(
                self.runner._model,
                self.runner._tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                sampler=sampler,
            )

        # Slow path: token-by-token with intervention
        input_ids = self.encode(prompt)
        prompt_len = input_ids.shape[1]
        # Keep everything on CPU for MLX compatibility
        generated = input_ids.cpu()

        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )

        for _ in range(max_new_tokens):
            # Run forward with intervention (returns tensor on runner.device)
            logits = self.run_with_intervention(generated, interventions)

            # Sample next token (keep on CPU)
            logits_cpu = logits.cpu()
            if temperature > 0:
                probs = torch.softmax(logits_cpu[0, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
            else:
                next_token = (
                    logits_cpu[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                )

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if next_token.item() == self.runner._tokenizer.eos_token_id:
                break

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        mx = _get_mx()

        input_ids = self.runner._tokenizer.encode(prompt)
        input_mx = mx.array([input_ids])

        logits = self.runner._model(input_mx)
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

        result = {}
        for token_str in target_tokens:
            ids = self.runner._tokenizer.encode(token_str)
            if ids:
                result[token_str] = probs[ids[0]].item()
            else:
                result[token_str] = 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        mx = _get_mx()

        input_ids = self.runner._tokenizer.encode(prompt)
        input_mx = mx.array([input_ids])

        logits = self.runner._model(input_mx)
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

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
        """Run forward pass with activation caching via class-level monkey-patching.

        MLX doesn't have PyTorch's hook system, so we monkey-patch the layer
        class's __call__ method to capture residual stream activations.
        """
        import numpy as np

        mx = _get_mx()

        if isinstance(input_ids, torch.Tensor):
            input_mx = mx.array(input_ids.cpu().numpy().astype("int32"))
        else:
            input_mx = mx.array(input_ids)

        model = self.runner._model
        cache = {}

        # Determine which layers to cache
        layers_to_cache = set()
        for i in range(len(model.layers)):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):
                layers_to_cache.add(i)

        if not layers_to_cache:
            # No caching needed, just run forward
            logits = self.forward(input_ids)
            return logits, cache

        # Build layer -> index mapping
        layer_to_idx = {id(layer): i for i, layer in enumerate(model.layers)}

        # Get the layer class and its original __call__
        layer_class = type(model.layers[0])
        original_call = layer_class.__call__

        # Create hooked version that captures outputs
        def hooked_call(self_layer, x, *args, **kwargs):
            result = original_call(self_layer, x, *args, **kwargs)
            layer_idx = layer_to_idx.get(id(self_layer))
            if layer_idx is not None and layer_idx in layers_to_cache:
                name = f"blocks.{layer_idx}.hook_resid_post"
                # Result is the hidden states tensor
                hidden = result
                # Convert to numpy immediately
                hidden_f32 = hidden.astype(mx.float32)
                mx.eval(hidden_f32)
                cache[name] = torch.from_numpy(np.array(hidden_f32))
            return result

        # Install class-level hook
        layer_class.__call__ = hooked_call

        try:
            # Run forward pass
            logits_mx = model(input_mx)
            logits_mx = logits_mx.astype(mx.float32)
            mx.eval(logits_mx)
            logits = torch.from_numpy(np.array(logits_mx)).to(self.runner.device)
        finally:
            # Restore original method
            layer_class.__call__ = original_call

        # Move cached tensors to correct device
        for k in cache:
            cache[k] = cache[k].to(self.runner.device)

        return logits, cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        # MLX doesn't support PyTorch gradients
        logits = self.forward(input_ids)
        return logits, {}

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        # Convert prefill logits to get first token
        if temperature > 0:
            probs = torch.softmax(prefill_logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = prefill_logits[0, -1, :].argmax().item()

        if next_token == self.runner._tokenizer.eos_token_id:
            return ""

        # Use mlx_lm.generate to continue from first token
        generate = _get_generate()
        first_token_text = self.runner.decode_ids([next_token])

        # Create sampler based on temperature
        if temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature)
        else:
            sampler = None

        remaining = generate(
            self.runner._model,
            self.runner._tokenizer,
            prompt=first_token_text,
            max_tokens=max_new_tokens - 1,
            sampler=sampler,
        )
        return first_token_text + remaining

    def init_kv_cache(self):
        # MLX handles caching internally in generate_step
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits as PyTorch tensor."""
        import numpy as np

        mx = _get_mx()

        if isinstance(input_ids, torch.Tensor):
            input_mx = mx.array(input_ids.cpu().numpy().astype("int32"))
        else:
            input_mx = mx.array(input_ids)

        logits_mx = self.runner._model(input_mx)
        # Convert to float32 for numpy compatibility (mlx defaults to bfloat16)
        logits_mx = logits_mx.astype(mx.float32)
        mx.eval(logits_mx)
        # Convert to torch and move to runner's device
        return torch.from_numpy(np.array(logits_mx)).to(self.runner.device)

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        """Run forward pass with interventions via class-level monkey-patching.

        Supports 'add' mode for steering on resid_post component.
        """
        import numpy as np

        mx = _get_mx()

        if isinstance(input_ids, torch.Tensor):
            input_mx = mx.array(input_ids.cpu().numpy().astype("int32"))
        else:
            input_mx = mx.array(input_ids)

        model = self.runner._model

        # Build intervention lookup: layer_idx -> intervention
        intervention_by_layer = {}
        for interv in interventions:
            if interv.component != "resid_post":
                raise NotImplementedError(
                    f"MLX only supports resid_post interventions, got {interv.component}"
                )
            if interv.mode != "add":
                raise NotImplementedError(
                    f"MLX only supports 'add' mode interventions, got {interv.mode}"
                )
            intervention_by_layer[interv.layer] = interv

        if not intervention_by_layer:
            return self.forward(input_ids)

        # Build layer -> index mapping
        layer_to_idx = {id(layer): i for i, layer in enumerate(model.layers)}

        # Get the layer class and its original __call__
        layer_class = type(model.layers[0])
        original_call = layer_class.__call__

        # Create hooked version that applies interventions
        def hooked_call(self_layer, x, *args, **kwargs):
            result = original_call(self_layer, x, *args, **kwargs)
            layer_idx = layer_to_idx.get(id(self_layer))

            if layer_idx is not None and layer_idx in intervention_by_layer:
                interv = intervention_by_layer[layer_idx]
                # Convert steering vector to MLX
                steering = mx.array(interv.scaled_values.astype("float32"))

                # Apply steering based on target type
                # result shape: [batch, seq, hidden_size]
                if not interv.target.is_all_positions and interv.target.positions:
                    # Position-specific: steering shape [n_positions, hidden_size]
                    # Only add to specified positions, clamped to sequence length
                    seq_len = result.shape[1]
                    for i, pos in enumerate(interv.target.positions):
                        if pos < seq_len and i < len(steering):
                            result = result.at[:, pos, :].add(steering[i])
                else:
                    # All positions: steering shape [hidden_size], broadcasts
                    result = result + steering

            return result

        # Install class-level hook
        layer_class.__call__ = hooked_call

        try:
            logits_mx = model(input_mx)
            logits_mx = logits_mx.astype(mx.float32)
            mx.eval(logits_mx)
            logits = torch.from_numpy(np.array(logits_mx)).to(self.runner.device)
        finally:
            layer_class.__call__ = original_call

        return logits

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations."""
        import numpy as np

        mx = _get_mx()

        if isinstance(input_ids, torch.Tensor):
            input_mx = mx.array(input_ids.cpu().numpy().astype("int32"))
        else:
            input_mx = mx.array(input_ids)

        model = self.runner._model
        cache = {}

        # Build intervention lookup
        intervention_by_layer = {}
        for interv in interventions:
            if interv.component != "resid_post":
                raise NotImplementedError(
                    f"MLX only supports resid_post interventions, got {interv.component}"
                )
            if interv.mode != "add":
                raise NotImplementedError(
                    f"MLX only supports 'add' mode interventions, got {interv.mode}"
                )
            intervention_by_layer[interv.layer] = interv

        # Determine which layers to cache
        layers_to_cache = set()
        for i in range(len(model.layers)):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):
                layers_to_cache.add(i)

        # Build layer -> index mapping
        layer_to_idx = {id(layer): i for i, layer in enumerate(model.layers)}

        # Get the layer class and its original __call__
        layer_class = type(model.layers[0])
        original_call = layer_class.__call__

        # Create hooked version that applies interventions AND caches
        def hooked_call(self_layer, x, *args, **kwargs):
            result = original_call(self_layer, x, *args, **kwargs)
            layer_idx = layer_to_idx.get(id(self_layer))

            if layer_idx is not None:
                # Apply intervention if present
                if layer_idx in intervention_by_layer:
                    interv = intervention_by_layer[layer_idx]
                    steering = mx.array(interv.scaled_values.astype("float32"))
                    result = result + steering

                # Cache if requested (after intervention)
                if layer_idx in layers_to_cache:
                    name = f"blocks.{layer_idx}.hook_resid_post"
                    hidden_f32 = result.astype(mx.float32)
                    mx.eval(hidden_f32)
                    cache[name] = torch.from_numpy(np.array(hidden_f32))

            return result

        # Install class-level hook
        layer_class.__call__ = hooked_call

        try:
            logits_mx = model(input_mx)
            logits_mx = logits_mx.astype(mx.float32)
            mx.eval(logits_mx)
            logits = torch.from_numpy(np.array(logits_mx)).to(self.runner.device)
        finally:
            layer_class.__call__ = original_call

        # Move cached tensors to correct device
        for k in cache:
            cache[k] = cache[k].to(self.runner.device)

        return logits, cache
