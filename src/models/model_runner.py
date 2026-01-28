"""
Model runner for inference with intervention support.

Supports TransformerLens and nnsight backends.

Example:
    runner = ModelRunner("Qwen/Qwen2.5-7B-Instruct")
    output = runner.generate("What is 2+2?")

    # With intervention
    from src.models.intervention_utils import steering
    intervention = steering(layer=26, direction=probe.direction, strength=100.0)
    output = runner.generate("What is 2+2?", intervention=intervention)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch

from .interventions import Intervention, create_intervention_hook


class ModelBackend(Enum):
    TRANSFORMERLENS = "transformerlens"
    NNSIGHT = "nnsight"
    PYVENE = "pyvene"


@dataclass
class LabelProbsOutput:
    """Probabilities for two label options."""

    prob1: float
    prob2: float


class ModelRunner:
    """Model runner for inference with intervention support."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend = ModelBackend.TRANSFORMERLENS,
    ):
        self.model_name = model_name
        self.backend = backend

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        self.dtype = dtype

        if backend == ModelBackend.TRANSFORMERLENS:
            self._init_transformerlens()
        elif backend == ModelBackend.NNSIGHT:
            self._init_nnsight()
        elif backend == ModelBackend.PYVENE:
            self._init_pyvene()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._is_chat_model = self._detect_chat_model()
        print(f"Model loaded: {model_name} (chat={self._is_chat_model})")
        print(f"  n_layers={self.n_layers}, d_model={self.d_model}\n")

    def _init_transformerlens(self) -> None:
        from transformer_lens import HookedTransformer

        print(f"Loading {self.model_name} on {self.device} (TransformerLens)...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self.model.eval()
        self._backend = _TransformerLensBackend(self)

    def _init_nnsight(self) -> None:
        from nnsight import LanguageModel

        print(f"Loading {self.model_name} on {self.device} (nnsight)...")
        self.model = LanguageModel(
            self.model_name, device_map=self.device, torch_dtype=self.dtype
        )
        self._backend = _NNsightBackend(self)

    def _init_pyvene(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} on {self.device} (pyvene)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = _PyveneBackend(self)

    def _detect_chat_model(self) -> bool:
        name = self.model_name.lower()
        return any(x in name for x in ["instruct", "chat", "-it", "rlhf"])

    def _apply_chat_template(self, prompt: str) -> str:
        if not self._is_chat_model:
            return prompt
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    @property
    def tokenizer(self):
        return self._backend.get_tokenizer()

    @property
    def n_layers(self) -> int:
        return self._backend.get_n_layers()

    @property
    def d_model(self) -> int:
        return self._backend.get_d_model()

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into tensor of token IDs.

        Args:
            text: Input text to tokenize
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Token IDs tensor of shape [1, seq_len]
        """
        # prepend_bos=False by default for consistent behavior across backends:
        # - TransformerLens: uses to_tokens(prepend_bos=X) directly
        # - NNsight/Pyvene: use HF tokenizer, only prepend if bos_token_id exists
        # Some models (e.g. Qwen) have bos_token_id=None, so prepend_bos=True
        # may behave differently across backends for those models.
        return self._backend.tokenize(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._backend.decode(token_ids)

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str | list[str]:
        """Generate text, optionally with intervention."""
        if isinstance(prompt, str):
            return self._generate_single(
                prompt, max_new_tokens, temperature, intervention, past_kv_cache
            )

        return [
            self._generate_single(
                p, max_new_tokens, temperature, intervention, past_kv_cache
            )
            for p in prompt
        ]

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        formatted = self._apply_chat_template(prompt)
        return self._backend.generate(
            formatted, max_new_tokens, temperature, intervention, past_kv_cache
        )

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        return self._backend.generate_from_cache(
            prefill_logits, frozen_kv_cache, max_new_tokens, temperature
        )

    def get_label_probs(
        self,
        prompt: str | list[str],
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple | list[tuple]:
        """Get probabilities for two label options."""
        if isinstance(prompt, str):
            return self._get_label_probs_single(
                prompt, choice_prefix, labels, past_kv_cache
            )
        return [
            self._get_label_probs_single(p, choice_prefix, labels, past_kv_cache)
            for p in prompt
        ]

    def get_divergent_token_ids(self, label1: str, label2: str) -> tuple[int, int]:
        """Get first divergent token IDs for two labels.

        For multi-token labels like OPTION_ONE/OPTION_TWO, finds where they diverge
        and returns the token IDs at that position.

        Args:
            label1: First label string
            label2: Second label string

        Returns:
            Tuple of (token_id_1, token_id_2) at the first divergent position
        """
        tokenizer = self.tokenizer
        ids1 = tokenizer.encode(" " + label1, add_special_tokens=False)
        ids2 = tokenizer.encode(" " + label2, add_special_tokens=False)

        diverge_pos = 0
        for i in range(min(len(ids1), len(ids2))):
            if ids1[i] != ids2[i]:
                diverge_pos = i
                break
        else:
            diverge_pos = min(len(ids1), len(ids2))

        tok1 = ids1[diverge_pos] if diverge_pos < len(ids1) else ids1[-1]
        tok2 = ids2[diverge_pos] if diverge_pos < len(ids2) else ids2[-1]
        return tok1, tok2

    def _get_label_probs_single(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple:
        """Get probabilities for two label options."""
        tokenizer = self.tokenizer
        label1, label2 = labels

        ids1 = tokenizer.encode(" " + label1, add_special_tokens=False)
        ids2 = tokenizer.encode(" " + label2, add_special_tokens=False)

        diverge_pos = 0
        for i in range(min(len(ids1), len(ids2))):
            if ids1[i] != ids2[i]:
                diverge_pos = i
                break
        else:
            diverge_pos = min(len(ids1), len(ids2))

        # Get the diverging tokens
        tok1 = ids1[diverge_pos] if diverge_pos < len(ids1) else None
        tok2 = ids2[diverge_pos] if diverge_pos < len(ids2) else None

        # Always use full prompt context (kv_cache doesn't help when we need to extend)
        base_text = self._apply_chat_template(prompt) + choice_prefix

        # If labels diverge at first token, look at next token after choice_prefix
        if diverge_pos == 0:
            probs = self._backend.get_next_token_probs_by_id(
                base_text, [tok1, tok2], past_kv_cache
            )
            return (probs.get(tok1, 0.0), probs.get(tok2, 0.0))

        # Labels have common prefix - need to condition on it
        common_prefix = tokenizer.decode(ids1[:diverge_pos])
        extended_text = base_text + common_prefix
        probs = self._backend.get_next_token_probs_by_id(
            extended_text, [tok1, tok2], past_kv_cache
        )
        return (probs.get(tok1, 0.0), probs.get(tok2, 0.0))

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache.

        Args:
            prompt: Input text
            names_filter: Function to filter which hooks to cache (e.g. lambda n: 'resid' in n)
            past_kv_cache: Optional past key-value cache for continuation
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Tuple of (logits, cache) where cache maps hook names to activation tensors
        """
        formatted = self._apply_chat_template(prompt)
        # prepend_bos ensures consistent seq_len across backends for activation comparison
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    def run_with_cache_and_grad(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled for attribution patching.

        Args:
            prompt: Input text
            names_filter: Function to filter which hooks to cache
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Tuple of (logits, cache) where cache values have requires_grad=True
        """
        formatted = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        # Unlike run_with_cache, does NOT use torch.no_grad() - gradients flow through
        return self._backend.run_with_cache_and_grad(input_ids, names_filter)

    def forward_with_intervention(
        self,
        prompt: str,
        intervention: Union[Intervention, list[Intervention]],
        prepend_bos: bool = False,
    ) -> torch.Tensor:
        """Run forward pass with intervention(s) applied.

        Args:
            prompt: Input text
            intervention: Single Intervention or list of Interventions to apply
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        formatted = self._apply_chat_template(prompt)
        # Note: intervention.positions are relative to tokenized input
        # If prepend_bos=True, position 0 is BOS, position 1 is first real token
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        # Normalize to list for uniform handling
        interventions = [intervention] if isinstance(intervention, Intervention) else intervention
        return self._backend.forward_with_intervention(input_ids, interventions)

    def init_kv_cache(self):
        return self._backend.init_kv_cache()


class _BackendBase(ABC):
    def __init__(self, runner: ModelRunner):
        self.runner = runner

    @abstractmethod
    def get_tokenizer(self): ...
    @abstractmethod
    def get_n_layers(self) -> int: ...
    @abstractmethod
    def get_d_model(self) -> int: ...
    @abstractmethod
    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor: ...
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str: ...
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str: ...
    @abstractmethod
    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]: ...
    @abstractmethod
    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]: ...
    @abstractmethod
    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]: ...

    @abstractmethod
    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled."""
        ...

    @abstractmethod
    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    def init_kv_cache(self): ...

    @abstractmethod
    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
    ) -> torch.Tensor:
        """Run forward pass with interventions, returning logits."""
        ...


class _TransformerLensBackend(_BackendBase):
    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.cfg.n_layers

    def get_d_model(self) -> int:
        return self.runner.model.cfg.d_model

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        # TransformerLens natively supports prepend_bos via to_tokens()
        # This always works regardless of model's bos_token_id setting
        return self.runner.model.to_tokens(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner.model.to_string(token_ids)

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

        # If we have a frozen kv_cache, use custom generation loop
        if past_kv_cache is not None:
            return self._generate_with_cache(
                input_ids, max_new_tokens, temperature, past_kv_cache
            )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "stop_at_eos": True,
            "verbose": False,
            "use_past_kv_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if intervention is not None:
                hook, _ = create_intervention_hook(
                    intervention,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                    tokenizer=self.get_tokenizer(),
                )
                with self.runner.model.hooks(
                    fwd_hooks=[(intervention.hook_name, hook)]
                ):
                    output_ids = self.runner.model.generate(input_ids, **gen_kwargs)
            else:
                output_ids = self.runner.model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any,
    ) -> str:
        """Generate using frozen kv_cache - only pass new tokens each step."""
        import copy

        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        # Unfreeze a copy of the cache for generation
        kv = copy.deepcopy(past_kv_cache)
        kv.unfreeze()

        # Get first token logits from prefill (cache already has prompt processed)
        # We need to do one forward pass with NO new tokens to get the next-token logits
        # TransformerLens doesn't support empty input, so we use the logits from prefill
        # Actually we need to get the logits that were computed during prefill
        # Since we don't have them, we need to recompute with the full prompt
        logits = self.runner.model(input_ids, past_kv_cache=kv)
        next_logits = logits[0, -1, :]

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

                # Pass only the new token
                step_logits = self.runner.model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            logits = self.runner.model(input_ids, past_kv_cache=past_kv_cache)
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
            logits = self.runner.model(input_ids, past_kv_cache=past_kv_cache)
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
        with torch.no_grad():
            return self.runner.model.run_with_cache(
                input_ids, names_filter=names_filter, past_kv_cache=past_kv_cache
            )

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run with gradients enabled for attribution patching."""
        cache = {}
        hooks = []

        # Determine which hooks to capture
        n_layers = self.get_n_layers()
        hooks_to_capture = []
        for i in range(n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        # Register hooks that preserve gradients
        def make_hook(hook_name):
            def hook_fn(act, hook=None):
                cache[hook_name] = act
                return act
            return hook_fn

        fwd_hooks = [(name, make_hook(name)) for _, _, name in hooks_to_capture]

        # Forward pass WITHOUT torch.no_grad()
        logits = self.runner.model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        import copy

        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        # Copy and unfreeze the cache for generation
        kv = copy.deepcopy(frozen_kv_cache)
        kv.unfreeze()

        # Start from the prefill logits (already computed in Step 1)
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

                # Pass only the new token - kv cache supplies the prefix context
                step_logits = self.runner.model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        from transformer_lens.past_key_value_caching import (
            HookedTransformerKeyValueCache,
        )

        return HookedTransformerKeyValueCache.init_cache(
            self.runner.model.cfg,
            device=self.runner.device,
            batch_size=1,
        )

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
    ) -> torch.Tensor:
        # Build hooks for all interventions
        fwd_hooks = []
        for intervention in interventions:
            hook_fn, _ = create_intervention_hook(
                intervention,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            fwd_hooks.append((intervention.hook_name, hook_fn))

        with torch.no_grad():
            logits = self.runner.model.run_with_hooks(
                input_ids,
                fwd_hooks=fwd_hooks,
            )
        return logits


class _NNsightBackend(_BackendBase):
    def __init__(self, runner):
        super().__init__(runner)
        # Detect model architecture for layer access
        # GPT2: model.transformer.h[i], LLaMA/Mistral: model.model.layers[i]
        # Store both reference (for len) and path (for trace context access)
        if hasattr(self.runner.model, "transformer"):
            self._layers = self.runner.model.transformer.h
            self._layers_path = "transformer.h"
        elif hasattr(self.runner.model, "model") and hasattr(
            self.runner.model.model, "layers"
        ):
            self._layers = self.runner.model.model.layers
            self._layers_path = "model.layers"
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner.model)}")

    def _get_layer(self, layer_idx: int):
        """Get layer module through model path (works inside trace context)."""
        # Must access through model path inside trace, not pre-fetched self._layers
        if self._layers_path == "transformer.h":
            return self.runner.model.transformer.h[layer_idx]
        else:
            return self.runner.model.model.layers[layer_idx]

    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner.model.config.hidden_size

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        # NNsight uses HuggingFace tokenizer (unlike TransformerLens which has its own)
        tokenizer = self.get_tokenizer()
        ids = tokenizer(text, return_tensors="pt").input_ids
        if prepend_bos:
            # Only prepend if tokenizer defines bos_token_id (Qwen has None)
            # This differs from TransformerLens which always has a BOS mechanism
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
        """Generate text with optional interventions."""
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]
        generated = input_ids.clone()

        # Prepare steering direction BEFORE trace context
        # (creating tensors inside trace can cause issues with NNsight's graph building)
        steering_direction = None
        steering_layer_idx = None
        if intervention is not None and isinstance(intervention, Intervention) and intervention.mode == "add":
            steering_layer_idx = intervention.layer
            steering_direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

        # Manual token-by-token generation loop
        # NNsight has model.generate() with tracer.iter for autoregressive generation,
        # but it only works with LanguageModel wrapper's .generator attribute.
        # For compatibility with base NNsight wrapper (used by toy models), use manual loop.
        for _ in range(max_new_tokens):
            # trace() creates intervention context - all tensor ops inside are recorded
            with self.runner.model.trace(generated):
                if steering_direction is not None:
                    # IMPORTANT: Must access layer through model path INSIDE trace context.
                    # Pre-fetched self._layers[i] returns Envoy proxy that doesn't work
                    # correctly when used outside its originating trace.
                    layer = self._get_layer(steering_layer_idx)

                    # KEY INSIGHT: NNsight's layer.output is tensor [batch, seq, hidden] directly
                    # Raw HuggingFace layers return tuple: (hidden_states, attn_weights, kv_cache)
                    # But NNsight unwraps this - layer.output IS the hidden_states tensor.
                    # WRONG: layer.output[0][:,:,:] - this indexes BATCH dim, gives [seq, hidden]
                    # RIGHT: layer.output[:,:,:] - this is the full [batch, seq, hidden] tensor
                    layer.output[:, :, :] += steering_direction

                # Get logits from final layer
                # Real models have lm_head, toy models expose output directly
                if hasattr(self.runner.model, "lm_head"):
                    logits = self.runner.model.lm_head.output.save()
                else:
                    logits = self.runner.model.output.save()

            # After trace context exits, saved tensors are materialized
            # NNsight 0.5+ returns tensors directly (older versions needed .value)
            if temperature > 0:
                probs = torch.softmax(logits[0, -1, :].detach() / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
            else:
                # Greedy decoding - take argmax
                next_token = logits[0, -1, :].detach().argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

        # Return only newly generated tokens (exclude prompt)
        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        # Note: Do NOT use torch.no_grad() - it interferes with nnsight's tracing
        with self.runner.model.trace(input_ids):
            logits = self.runner.model.lm_head.output.save()

        # nnsight 0.5.15 returns tensors directly (no .value)
        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
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
        # Note: Do NOT use torch.no_grad() - it interferes with nnsight's tracing
        with self.runner.model.trace(input_ids):
            logits = self.runner.model.lm_head.output.save()

        # nnsight 0.5.15 returns tensors directly (no .value)
        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def _get_component_module(self, layer, component: str):
        """Get the module for a specific component within a layer.

        Components:
            resid_post/resid_pre/resid_mid: layer output (block output)
            attn_out: attention output
            mlp_out: MLP output
        """
        if component in ("resid_post", "resid_pre", "resid_mid"):
            return layer
        elif component == "attn_out":
            # Access attn directly based on architecture (hasattr doesn't work on Envoy proxies)
            # GPT2: layer.attn, LLaMA/Qwen: layer.self_attn, Pythia: layer.attention
            if self._layers_path == "transformer.h":
                return layer.attn  # GPT2 style
            else:
                return layer.self_attn  # LLaMA/Qwen style
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
        # NNsight requires accessing modules in execution order, so only capture layer outputs
        # (resid_post). For attn_out/mlp_out, would need tracer.cache() approach.
        hooks_to_capture = []
        for i in range(len(self._layers)):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):
                hooks_to_capture.append((i, "resid_post", name))

        with self.runner.model.trace(input_ids):
            for layer_idx, component, name in hooks_to_capture:
                # Must access layer through model path inside trace (not pre-fetched self._layers)
                layer = self._get_layer(layer_idx)
                module = self._get_component_module(layer, component)
                # Module outputs differ: layer/attn return tuple, MLP returns tensor
                if component == "mlp_out":
                    # MLP returns tensor directly, ensure 3D shape [batch, seq, hidden]
                    out = module.output.save()
                else:
                    # Layer and attention return tuple, first element is hidden states
                    out = module.output[0].save()
                cache[name] = out
            logits = self.runner.model.lm_head.output.save()

        # Ensure all cached tensors have batch dimension [1, seq, hidden]
        result_cache = {}
        for k, v in cache.items():
            if v.ndim == 2:
                v = v.unsqueeze(0)
            result_cache[k] = v
        return logits, result_cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run with gradients enabled - nnsight preserves gradients by default."""
        # nnsight tracing preserves gradients, so this is same as run_with_cache
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

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
    ) -> torch.Tensor:
        # Apply all interventions within the same trace context
        with self.runner.model.trace(input_ids):
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

                # MLP returns tensor directly, layer/attn return tuple
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
                                # 2D: [seq, hidden]
                                if mode == "add":
                                    out[pos, :] += values
                                elif mode == "set":
                                    out[pos, :] = values
                                elif mode == "mul":
                                    out[pos, :] *= values
                            else:
                                # 3D: [batch, seq, hidden]
                                if mode == "add":
                                    out[:, pos, :] += values
                                elif mode == "set":
                                    out[:, pos, :] = values
                                elif mode == "mul":
                                    out[:, pos, :] *= values

            logits = self.runner.model.lm_head.output.save()

        return logits.detach()


class _PyveneBackend(_BackendBase):
    """Backend using pyvene for interventions."""

    def __init__(self, runner):
        super().__init__(runner)
        # Detect model architecture for layer access
        if hasattr(self.runner.model, "transformer"):
            # GPT2 style: model.transformer.h[i]
            self._layers_attr = "transformer.h"
            self._layers = self.runner.model.transformer.h
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.n_embd
        elif hasattr(self.runner.model, "gpt_neox"):
            # Pythia/GPT-NeoX style: model.gpt_neox.layers[i]
            self._layers_attr = "gpt_neox.layers"
            self._layers = self.runner.model.gpt_neox.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.hidden_size
        elif hasattr(self.runner.model, "model") and hasattr(
            self.runner.model.model, "layers"
        ):
            # LLaMA/Mistral style: model.model.layers[i]
            self._layers_attr = "model.layers"
            self._layers = self.runner.model.model.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.hidden_size
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner.model)}")

    def get_tokenizer(self):
        return self.runner._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        # Pyvene uses HuggingFace tokenizer (same as NNsight, unlike TransformerLens)
        tokenizer = self.get_tokenizer()
        ids = tokenizer(text, return_tensors="pt").input_ids
        if prepend_bos:
            # Only prepend if tokenizer defines bos_token_id (Qwen has None)
            # This differs from TransformerLens which always has a BOS mechanism
            bos_id = tokenizer.bos_token_id
            if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
                bos = torch.tensor([[bos_id]], dtype=ids.dtype)
                ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=True)

    def _get_component_name(self, layer: int, component: str = "block_output") -> str:
        """Get pyvene component name for a layer."""
        # Map our naming to pyvene's expected format
        if self._layers_attr == "transformer.h":
            # GPT2: use h[layer] style
            if component == "block_output":
                return f"transformer.h[{layer}]"
            elif component == "mlp_output":
                return f"transformer.h[{layer}].mlp"
            elif component == "attn_output":
                return f"transformer.h[{layer}].attn"
        elif self._layers_attr == "gpt_neox.layers":
            # Pythia/GPT-NeoX: use gpt_neox.layers[layer] style
            if component == "block_output":
                return f"gpt_neox.layers[{layer}]"
            elif component == "mlp_output":
                return f"gpt_neox.layers[{layer}].mlp"
            elif component == "attn_output":
                return f"gpt_neox.layers[{layer}].attention"
        else:
            # LLaMA style: use model.layers[layer] style
            if component == "block_output":
                return f"model.layers[{layer}]"
            elif component == "mlp_output":
                return f"model.layers[{layer}].mlp"
            elif component == "attn_output":
                return f"model.layers[{layer}].self_attn"
        return f"transformer.h[{layer}]"

    def _get_component_module(self, layer_idx: int, component: str):
        """Get the module for a specific component within a layer.

        Components:
            resid_post/resid_pre/resid_mid: layer (block output)
            attn_out: attention module
            mlp_out: MLP module
        """
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
                raise ValueError(f"Cannot find attention module in layer: {type(layer)}")
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

        if intervention is not None and isinstance(intervention, Intervention) and intervention.mode == "add":
            # Use direct PyTorch hooks for steering (pyvene's subspace projection
            # doesn't work well for direct addition to hidden states)
            direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

            # Get the layer to hook
            layer_module = self._layers[intervention.layer]

            # Create steering hook
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    steered = hidden + direction.unsqueeze(0).unsqueeze(0)
                    return (steered,) + output[1:]
                else:
                    return output + direction.unsqueeze(0).unsqueeze(0)

            # Generate token by token with intervention
            generated = input_ids.clone()
            eos_id = self.get_tokenizer().eos_token_id

            for _ in range(max_new_tokens):
                # Register hook for this forward pass
                hook = layer_module.register_forward_hook(steering_hook)

                with torch.no_grad():
                    outputs = self.runner.model(generated)
                    logits = outputs.logits

                hook.remove()

                if temperature > 0:
                    probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).unsqueeze(0)
                else:
                    next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == eos_id:
                    break
        else:
            # No intervention - use standard HF generate
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = self.runner.model.generate(input_ids, **gen_kwargs)
            generated = output_ids

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            outputs = self.runner.model(input_ids)
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
            outputs = self.runner.model(input_ids)
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

        # Determine which hooks to capture
        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        # Use PyTorch forward hooks to capture activations
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
                outputs = self.runner.model(input_ids)
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
                    # Don't detach - preserve gradients
                    if isinstance(out, tuple):
                        cache[hook_name] = out[0]
                    else:
                        cache[hook_name] = out
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            # No torch.no_grad() - preserve gradients
            outputs = self.runner.model(input_ids)
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
        # pyvene uses HF models which support past_key_values
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

                # Forward with KV cache
                outputs = self.runner.model(
                    next_token.unsqueeze(0),
                    past_key_values=frozen_kv_cache,
                    use_cache=True,
                )
                next_logits = outputs.logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        """Initialize a KV cache wrapper for HF models."""
        # Return a simple wrapper that stores past_key_values
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

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
    ) -> torch.Tensor:
        # Register hooks for all interventions
        hooks = []
        for intervention in interventions:
            values = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            target = intervention.target
            mode = intervention.mode
            module = self._get_component_module(intervention.layer, intervention.component)

            # Create closure to capture intervention-specific values
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
            outputs = self.runner.model(input_ids)

        # Remove all hooks
        for hook in hooks:
            hook.remove()

        return outputs.logits
