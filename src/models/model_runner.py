"""
Model runner for inference with intervention support.

Supports TransformerLens and nnsight backends.

Example:
    runner = ModelRunner("Qwen/Qwen2.5-7B-Instruct")
    output = runner.generate("What is 2+2?")

    # With intervention
    from src.interventions import SteeringConfig
    config = SteeringConfig(direction=probe.direction, layer=26, strength=100.0)
    output = runner.generate("What is 2+2?", intervention=config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from .interventions import InterventionConfig, SteeringConfig, create_intervention_hook


class ModelBackend(Enum):
    TRANSFORMERLENS = "transformerlens"
    NNSIGHT = "nnsight"


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
        else:
            self._init_nnsight()

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

    def tokenize(self, text: str) -> torch.Tensor:
        return self._backend.tokenize(text)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._backend.decode(token_ids)

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[InterventionConfig] = None,
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
        intervention: Optional[InterventionConfig],
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

    def _get_label_probs_single(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple:
        """Get probabilities for two label options.

        Handles both simple labels (a/b, 1/2) and complex multi-token labels
        (OPTION_ONE/OPTION_TWO) by finding where they diverge and looking at
        the appropriate token position.
        """
        tokenizer = self.tokenizer
        label1, label2 = labels

        # Tokenize both labels with space prefix (model outputs space after "I select:")
        ids1 = tokenizer.encode(" " + label1, add_special_tokens=False)
        ids2 = tokenizer.encode(" " + label2, add_special_tokens=False)

        # Find first position where tokens differ
        diverge_pos = 0
        for i in range(min(len(ids1), len(ids2))):
            if ids1[i] != ids2[i]:
                diverge_pos = i
                break
        else:
            # One is prefix of the other
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
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache."""
        formatted = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

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
    def tokenize(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str: ...
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[InterventionConfig],
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
    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    def init_kv_cache(self): ...


class _TransformerLensBackend(_BackendBase):
    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.cfg.n_layers

    def get_d_model(self) -> int:
        return self.runner.model.cfg.d_model

    def tokenize(self, text: str) -> torch.Tensor:
        return self.runner.model.to_tokens(text, prepend_bos=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner.model.to_string(token_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[InterventionConfig],
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
                    model_dtype=self.runner.dtype,
                    model_device=self.runner.device,
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


class _NNsightBackend(_BackendBase):
    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner.model.config.hidden_size

    def tokenize(self, text: str) -> torch.Tensor:
        return self.get_tokenizer()(text, return_tensors="pt").input_ids.to(
            self.runner.device
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[InterventionConfig],
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if intervention is not None and isinstance(intervention, SteeringConfig):
                with self.runner.model.trace(input_ids) as tracer:
                    layer_output = self.runner.model.model.layers[
                        intervention.layer
                    ].output[0]
                    direction = torch.tensor(
                        intervention.direction * intervention.strength,
                        dtype=self.runner.dtype,
                        device=self.runner.device,
                    )
                    layer_output[:, :, :] += direction
                    output_ids = self.runner.model.generate(tracer.input, **gen_kwargs)
            else:
                output_ids = self.runner.model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            with self.runner.model.trace(input_ids) as tracer:
                logits = self.runner.model.lm_head.output.save()

        probs = torch.softmax(logits.value[0, -1, :], dim=-1)
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
            with self.runner.model.trace(input_ids) as tracer:
                logits = self.runner.model.lm_head.output.save()

        probs = torch.softmax(logits.value[0, -1, :], dim=-1)
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
        with self.runner.model.trace(input_ids) as tracer:
            for i, layer in enumerate(self.runner.model.model.layers):
                name = f"blocks.{i}.hook_resid_post"
                if names_filter is None or names_filter(name):
                    cache[name] = layer.output[0].save()
            logits = self.runner.model.lm_head.output.save()
        return logits.value, {k: v.value for k, v in cache.items()}

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
