"""Anthropic backend implementation using the Anthropic API.

Note: Anthropic API does not provide logprobs, so trajectory generation
returns 0.0 logprobs for all tokens. This backend is suitable for:
- Text generation
- Categorical judgments (yes/no scoring)

But NOT suitable for:
- Probability-weighted analysis (all weights will be equal)
- Perplexity-based metrics

For binary choice, use the prefill technique with max_tokens=1.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_tokenizer import APITokenizer
from .model_backend import Backend, BinaryChoiceResult
from ..interventions import Intervention


class AnthropicBackend(Backend):
    """Backend using Anthropic API for inference.

    Note: Anthropic API does NOT provide logprobs. All logprob values
    returned by this backend are 0.0. This means:
    - generate_trajectory returns uniform logprobs
    - Probability-based weighting will be uniform

    For categorical judgments (yes/no scoring), this backend works fine.
    """

    supports_inference_mode: bool = False  # Not applicable for API calls

    @property
    def is_cloud_api(self) -> bool:
        return True

    # Default Anthropic model
    ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, runner: Any, model: str | None = None):
        """Initialize Anthropic backend.

        Args:
            runner: ModelRunner instance
            model: Anthropic model name (default: claude-sonnet-4-20250514)
        """
        super().__init__(runner)
        self._model = model or self.ANTHROPIC_DEFAULT_MODEL
        # Use cl100k_base as approximation for Claude tokenization
        self._tokenizer = APITokenizer(encoding_name="cl100k_base")
        self._client = None

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Set it with: export ANTHROPIC_API_KEY=your-key"
                )
            self._client = Anthropic(api_key=api_key)
        return self._client

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        # Unknown for closed models
        return 0

    def get_d_model(self) -> int:
        # Unknown for closed models
        return 0

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
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
            raise NotImplementedError(
                "Anthropic backend does not support interventions"
            )

        client = self._get_client()

        # Anthropic requires temperature >= 0
        temp = temperature if temperature > 0 else 0.0

        response = client.messages.create(
            model=self._model,
            max_tokens=max_new_tokens,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens.

        Note: Anthropic API does NOT support logprobs.
        This returns uniform probabilities as a fallback.
        """
        n = len(target_tokens)
        uniform_prob = 1.0 / n if n > 0 else 0.0
        return {token: uniform_prob for token in target_tokens}

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID.

        Note: Anthropic API does NOT support logprobs.
        This returns uniform probabilities as a fallback.
        """
        n = len(token_ids)
        uniform_prob = 1.0 / n if n > 0 else 0.0
        return {tid: uniform_prob for tid in token_ids}

    def get_binary_choice_probs(
        self,
        prompt: str,
        labels: tuple[str, str],
        choice_prefix: str = "The answer is ",
    ) -> BinaryChoiceResult:
        """Get binary choice result using prefill technique.

        Uses Anthropic's assistant prefill with max_tokens=1 to force a choice.
        Note: Anthropic does NOT provide logprobs, so this only returns
        which option was chosen, not the actual probabilities.

        Args:
            prompt: The question/task prompt
            labels: Two candidate labels, e.g. ("A", "B")
            choice_prefix: Prefill text for assistant, e.g. "The answer is "

        Returns:
            BinaryChoiceResult with binary probs (1.0/0.0 for chosen/unchosen)
        """
        client = self._get_client()

        # Build messages with prefill (strip trailing whitespace - Anthropic requirement)
        prefill = choice_prefix.rstrip()
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefill},
        ]

        response = client.messages.create(
            model=self._model,
            messages=messages,
            max_tokens=1,
            temperature=0,
        )

        # Extract the generated token
        generated = ""
        if response.content and len(response.content) > 0:
            generated = response.content[0].text.strip()

        # Determine which label was chosen
        # Check if generated matches either label (case-insensitive, prefix match)
        label_a_lower = labels[0].lower().strip()
        label_b_lower = labels[1].lower().strip()
        gen_lower = generated.lower().strip()

        choice_idx = -1  # Default: neither
        if gen_lower.startswith(label_a_lower) or label_a_lower.startswith(gen_lower):
            choice_idx = 0
        elif gen_lower.startswith(label_b_lower) or label_b_lower.startswith(gen_lower):
            choice_idx = 1

        # Set probabilities (binary since no logprobs available)
        if choice_idx == 0:
            probs = (1.0, 0.0)
            logprobs = (0.0, -float("inf"))
        elif choice_idx == 1:
            probs = (0.0, 1.0)
            logprobs = (-float("inf"), 0.0)
        else:
            # Neither matched - return uniform
            probs = (0.5, 0.5)
            logprobs = (-0.693, -0.693)  # log(0.5)

        return BinaryChoiceResult(
            choice_idx=choice_idx,
            probs=probs,
            logprobs=logprobs,
            tokens=(labels[0], labels[1]),
        )

    def compute_binary_choice_trajectories(
        self,
        prompt: str,
        labels: tuple[str, str],
        choice_prefix: str,
    ) -> tuple:
        """Compute trajectories for binary choice using API.

        Anthropic doesn't provide logprobs, so we use binary values (0.0/-inf).

        Args:
            prompt: The user prompt
            labels: Two choice labels, e.g. ("(A)", "(B)")
            choice_prefix: Text before the choice, e.g. "I choose: "

        Returns:
            Tuple of (traj_a, traj_b) GeneratedTrajectory objects
        """
        from ..generated_trajectory import GeneratedTrajectory

        # Call API with semantic context
        result = self.get_binary_choice_probs(prompt, labels, choice_prefix)

        # Build token_ids for trajectory structure
        full_a = prompt + choice_prefix + labels[0]
        full_b = prompt + choice_prefix + labels[1]
        token_ids_a = self._tokenizer.encode(full_a)
        token_ids_b = self._tokenizer.encode(full_b)

        # Find divergence point
        div_pos = 0
        min_len = min(len(token_ids_a), len(token_ids_b))
        while div_pos < min_len and token_ids_a[div_pos] == token_ids_b[div_pos]:
            div_pos += 1

        # Build logprobs arrays with API result at divergence
        logprobs_a = [0.0] * len(token_ids_a)
        logprobs_b = [0.0] * len(token_ids_b)

        if div_pos < len(token_ids_a):
            logprobs_a[div_pos] = result.logprobs[0]
        if div_pos < len(token_ids_b):
            logprobs_b[div_pos] = result.logprobs[1]

        traj_a = GeneratedTrajectory.from_logprobs(token_ids_a, logprobs_a)
        traj_b = GeneratedTrajectory.from_logprobs(token_ids_b, logprobs_b)

        return traj_a, traj_b

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass not supported for API-based backend."""
        raise NotImplementedError(
            "Anthropic backend does not support direct forward passes. "
            "Use generate() or generate_trajectory() instead."
        )

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError(
            "Anthropic backend does not support activation caching"
        )

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError("Anthropic backend does not support gradients")

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Not supported for API-based backend."""
        raise NotImplementedError(
            "Anthropic backend does not support KV cache generation"
        )

    def init_kv_cache(self):
        """Not supported for API-based backend."""
        return None

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        """Not supported for API-based backend."""
        raise NotImplementedError("Anthropic backend does not support interventions")

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError("Anthropic backend does not support interventions")

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory WITHOUT logprobs (Anthropic doesn't support them).

        Args:
            token_ids: Input token IDs (will be decoded to text)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            Tuple of (all_token_ids, logprobs) where logprobs are all 0.0

        Warning:
            All logprobs will be 0.0 since Anthropic API doesn't provide them.
            This means probability-weighted metrics will be uniform.
        """
        # Decode input tokens to text
        prompt = self._tokenizer.decode(token_ids)

        # Generate text
        generated_text = self.generate(prompt, max_new_tokens, temperature)

        # Tokenize the generated text
        generated_ids = self._tokenizer.encode(generated_text)

        # Build full token list
        all_token_ids = list(token_ids) + generated_ids

        # All logprobs are 0.0 (Anthropic doesn't provide them)
        all_logprobs = [0.0] * len(all_token_ids)

        return all_token_ids, all_logprobs
