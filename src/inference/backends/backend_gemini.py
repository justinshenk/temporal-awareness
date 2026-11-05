"""Gemini backend implementation using the google-genai SDK."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_tokenizer import APITokenizer
from .model_backend import Backend
from ..interventions import Intervention


class GeminiBackend(Backend):
    """Backend using Google Gemini API via google-genai SDK."""

    supports_inference_mode: bool = False

    @property
    def is_cloud_api(self) -> bool:
        return True

    GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, runner: Any, model: str | None = None):
        super().__init__(runner)
        self._model = model or self.GEMINI_DEFAULT_MODEL
        self._tokenizer = APITokenizer(encoding_name="cl100k_base")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return 0

    def get_d_model(self) -> int:
        return 0

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        return torch.tensor([self._tokenizer.encode(text, add_special_tokens=add_special_tokens)])

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str:
        if intervention is not None:
            raise NotImplementedError("Gemini backend does not support interventions")

        import time
        from google.genai import errors as genai_errors

        client = self._get_client()
        config = {
            "max_output_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
        }
        last_err: Optional[Exception] = None
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                # Retry only on transient codes (rate limit / unavailable / timeout).
                if code not in (408, 429, 500, 502, 503, 504):
                    raise
                last_err = e
                wait = min(2 ** attempt, 120)  # 1, 2, 4, 8, 16, 32, 64, 120
                print(
                    f"  Gemini {self._model} got {code}; retry {attempt + 1}/8 "
                    f"after {wait}s"
                )
                time.sleep(wait)
        raise last_err  # type: ignore[misc]

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        n = len(target_tokens)
        return {t: (1.0 / n if n else 0.0) for t in target_tokens}

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        n = len(token_ids)
        return {tid: (1.0 / n if n else 0.0) for tid in token_ids}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Gemini backend has no direct forward pass.")

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Gemini backend does not support activation caching")

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Gemini backend does not support gradients")

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError("Gemini backend does not support KV cache generation")

    def init_kv_cache(self):
        return None

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        raise NotImplementedError("Gemini backend does not support interventions")

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Gemini backend does not support interventions")

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        prompt = self._tokenizer.decode(token_ids)
        text = self.generate(prompt, max_new_tokens, temperature)
        gen_ids = self._tokenizer.encode(text)
        all_ids = list(token_ids) + gen_ids
        return all_ids, [0.0] * len(all_ids)
