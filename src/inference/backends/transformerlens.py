"""TransformerLens backend implementation."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from .model_backend import Backend
from ..interventions import Intervention, create_intervention_hook


class TransformerLensBackend(Backend):
    """Backend using TransformerLens for model inference and interventions."""

    def get_tokenizer(self):
        return self.runner._model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner._model.cfg.n_layers

    def get_d_model(self) -> int:
        return self.runner._model.cfg.d_model

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into token IDs tensor.

        Note: TransformerLens uses prepend_bos instead of add_special_tokens.
        """
        # TransformerLens handles BOS via prepend_bos, add_special_tokens is ignored
        return self.runner._model.to_tokens(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner._model.to_string(token_ids)

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

        if past_kv_cache is not None:
            return self._generate_with_cache(
                input_ids, max_new_tokens, temperature, past_kv_cache
            )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "stop_at_eos": True,
            "verbose": False,
            # Disable KV cache when intervention is active so each generation step
            # processes the full sequence and interventions apply to all positions
            "use_past_kv_cache": intervention is None,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if intervention is not None:
                hook, _ = create_intervention_hook(
                    intervention,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                with self.runner._model.hooks(
                    fwd_hooks=[(intervention.hook_name, hook)]
                ):
                    output_ids = self.runner._model.generate(input_ids, **gen_kwargs)
            else:
                output_ids = self.runner._model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any,
    ) -> str:
        """Generate using frozen kv_cache - only pass new tokens each step."""
        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        kv = copy.deepcopy(past_kv_cache)
        kv.unfreeze()

        logits = self.runner._model(input_ids, past_kv_cache=kv)
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

                step_logits = self.runner._model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            logits = self.runner._model(input_ids, past_kv_cache=past_kv_cache)
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
            logits = self.runner._model(input_ids, past_kv_cache=past_kv_cache)
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
            return self.runner._model.run_with_cache(
                input_ids, names_filter=names_filter, past_kv_cache=past_kv_cache
            )

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run with gradients enabled for attribution patching."""
        cache = {}

        n_layers = self.get_n_layers()
        hooks_to_capture = []
        for i in range(n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        def make_hook(hook_name):
            def hook_fn(act, hook=None):
                cache[hook_name] = act
                return act

            return hook_fn

        fwd_hooks = [(name, make_hook(name)) for _, _, name in hooks_to_capture]
        logits = self.runner._model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        kv = copy.deepcopy(frozen_kv_cache)
        kv.unfreeze()

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

                step_logits = self.runner._model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        return HookedTransformerKeyValueCache.init_cache(
            self.runner._model.cfg,
            device=self.runner.device,
            batch_size=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits."""
        with torch.no_grad():
            return self.runner._model(input_ids)

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        fwd_hooks = []
        for intervention in interventions:
            hook_fn, _ = create_intervention_hook(
                intervention,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            fwd_hooks.append((intervention.hook_name, hook_fn))

        with torch.no_grad():
            logits = self.runner._model.run_with_hooks(
                input_ids,
                fwd_hooks=fwd_hooks,
            )
        return logits

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients.

        Supports both layer-level and embedding-level interventions.
        Embedding interventions use component="embed" and target hook_embed.
        """
        cache = {}

        intervention_hooks = []
        for intervention in interventions:
            hook_fn, _ = create_intervention_hook(
                intervention,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )
            intervention_hooks.append((intervention.hook_name, hook_fn))

        n_layers = self.get_n_layers()
        cache_hooks = []
        for i in range(n_layers):
            for component in ["resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):

                    def make_hook(hook_name):
                        def hook_fn(act, hook=None):
                            cache[hook_name] = act
                            return act

                        return hook_fn

                    cache_hooks.append((name, make_hook(name)))

        all_hooks = intervention_hooks + cache_hooks
        logits = self.runner._model.run_with_hooks(input_ids, fwd_hooks=all_hooks)

        return logits, cache

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the model.

        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, d_model]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        with torch.no_grad():
            # TransformerLens: hook_embed captures W_E[tokens] + W_pos
            embeds = None

            def capture_embed(act, hook=None):
                nonlocal embeds
                embeds = act.clone()
                return act

            self.runner._model.run_with_hooks(
                token_ids,
                fwd_hooks=[("hook_embed", capture_embed)],
                stop_at_layer=0,  # Stop after embedding, before layer 0
            )

        return embeds
