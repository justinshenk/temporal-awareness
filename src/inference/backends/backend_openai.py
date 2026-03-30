"""OpenAI backend implementation using the OpenAI API."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_tokenizer import APITokenizer
from .model_backend import Backend, BinaryChoiceResult
from ..interventions import Intervention


class OpenAIBackend(Backend):
    """Backend using OpenAI API for inference."""

    supports_inference_mode: bool = False  # Not applicable for API calls

    @property
    def is_cloud_api(self) -> bool:
        return True

    def __init__(self, runner: Any, model: str = "gpt-4o"):
        """Initialize OpenAI backend.

        Args:
            runner: ModelRunner instance
            model: OpenAI model name (default: gpt-4o)
        """
        super().__init__(runner)
        self._model = model
        # GPT-4o uses o200k_base encoding
        self._tokenizer = APITokenizer(encoding_name="o200k_base")
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it with: export OPENAI_API_KEY=your-key"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        # Unknown for closed models - return placeholder
        return 0

    def get_d_model(self) -> int:
        # Unknown for closed models - return placeholder
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
            raise NotImplementedError("OpenAI backend does not support interventions")

        client = self._get_client()

        # Use temperature=0 for greedy, otherwise provided value
        temp = temperature if temperature > 0 else 0

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temp,
        )

        return response.choices[0].message.content or ""

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens.

        Note: OpenAI API has limited logprobs support. This uses the
        logprobs parameter to get top-k token probabilities.
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,  # Max allowed
        )

        result = {token: 0.0 for token in target_tokens}

        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            top_logprobs = choice.logprobs.content[0].top_logprobs
            logprob_dict = {lp.token: lp.logprob for lp in top_logprobs}

            for token in target_tokens:
                if token in logprob_dict:
                    result[token] = math.exp(logprob_dict[token])

        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID.

        Note: OpenAI API doesn't directly support token ID queries.
        This decodes the IDs and uses string-based lookup.
        """
        # Convert IDs to strings
        token_strs = [self._tokenizer.decode([tid]) for tid in token_ids]
        str_probs = self.get_next_token_probs(prompt, token_strs, past_kv_cache)

        # Map back to IDs
        result = {}
        for tid, tstr in zip(token_ids, token_strs):
            result[tid] = str_probs.get(tstr, 0.0)

        return result

    def get_binary_choice_probs(
        self,
        prompt: str,
        labels: tuple[str, str],
        choice_prefix: str = "",
    ) -> BinaryChoiceResult:
        """Get binary choice probabilities using logit_bias constraint.

        Uses OpenAI's logit_bias to constrain output to the first differing token
        between the two labels, then returns the probabilities from logprobs.

        Args:
            prompt: The question/task prompt
            labels: Two candidate labels, e.g. ("A", "B")
            choice_prefix: Optional prefix before the choice, e.g. "I choose: "

        Returns:
            BinaryChoiceResult with probabilities for both options
        """
        client = self._get_client()

        # Tokenize both labels fully
        tokens_a = self._tokenizer.encode(labels[0])
        tokens_b = self._tokenizer.encode(labels[1])

        # Find first position where tokens differ
        diff_pos = 0
        min_len = min(len(tokens_a), len(tokens_b))
        while diff_pos < min_len and tokens_a[diff_pos] == tokens_b[diff_pos]:
            diff_pos += 1

        # Get the differing tokens (or first token if one is prefix of other)
        if diff_pos < len(tokens_a) and diff_pos < len(tokens_b):
            token_id_a = tokens_a[diff_pos]
            token_id_b = tokens_b[diff_pos]
        elif diff_pos < len(tokens_a):
            # labels[1] is prefix of labels[0]
            token_id_a = tokens_a[diff_pos]
            token_id_b = tokens_a[diff_pos]  # Will be same, fallback to generation
        else:
            # labels[0] is prefix of labels[1]
            token_id_a = tokens_b[diff_pos] if diff_pos < len(tokens_b) else tokens_a[0]
            token_id_b = token_id_a

        # Build the shared prefix to prepend to assistant message
        shared_prefix = ""
        if diff_pos > 0:
            shared_prefix = self._tokenizer.decode(tokens_a[:diff_pos])

        # Build full prompt with choice prefix + shared prefix
        assistant_content = choice_prefix + shared_prefix
        if assistant_content:
            full_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            full_messages = [{"role": "user", "content": prompt}]

        # Use logit_bias only if tokens are different
        logit_bias = {}
        if token_id_a != token_id_b:
            logit_bias = {str(token_id_a): 100, str(token_id_b): 100}

        response = client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,  # Max allowed - need to find both A and B tokens
            logit_bias=logit_bias if logit_bias else None,
        )

        # Extract logprobs
        choice = response.choices[0]
        chosen_token = choice.message.content or ""

        # Default values
        logprob_a, logprob_b = -float("inf"), -float("inf")

        if choice.logprobs and choice.logprobs.content:
            top_logprobs = choice.logprobs.content[0].top_logprobs
            logprob_dict = {lp.token: lp.logprob for lp in top_logprobs}

            # Get the token strings we're looking for
            token_str_a = self._tokenizer.decode([token_id_a])
            token_str_b = self._tokenizer.decode([token_id_b])

            if token_str_a in logprob_dict:
                logprob_a = logprob_dict[token_str_a]
            if token_str_b in logprob_dict:
                logprob_b = logprob_dict[token_str_b]

        # Convert logprobs to probabilities
        prob_a = math.exp(logprob_a) if logprob_a > -float("inf") else 0.0
        prob_b = math.exp(logprob_b) if logprob_b > -float("inf") else 0.0

        # Normalize probabilities
        total = prob_a + prob_b
        if total > 0:
            prob_a /= total
            prob_b /= total
        else:
            # Fallback: determine from generated token
            if chosen_token.startswith(self._tokenizer.decode([token_id_a])):
                prob_a, prob_b = 1.0, 0.0
            elif chosen_token.startswith(self._tokenizer.decode([token_id_b])):
                prob_a, prob_b = 0.0, 1.0
            else:
                prob_a, prob_b = 0.5, 0.5

        # Determine choice
        choice_idx = 0 if prob_a >= prob_b else 1

        return BinaryChoiceResult(
            choice_idx=choice_idx,
            probs=(prob_a, prob_b),
            logprobs=(logprob_a, logprob_b),
            tokens=(labels[0], labels[1]),
        )

    def compute_binary_choice_trajectories(
        self,
        prompt: str,
        labels: tuple[str, str],
        choice_prefix: str,
    ) -> tuple:
        """Compute trajectories for binary choice using API.

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
            "OpenAI backend does not support direct forward passes. "
            "Use generate() or generate_trajectory() instead."
        )

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError("OpenAI backend does not support activation caching")

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError("OpenAI backend does not support gradients")

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Not supported for API-based backend."""
        raise NotImplementedError("OpenAI backend does not support KV cache generation")

    def init_kv_cache(self):
        """Not supported for API-based backend."""
        return None

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        """Not supported for API-based backend."""
        raise NotImplementedError("OpenAI backend does not support interventions")

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Not supported for API-based backend."""
        raise NotImplementedError("OpenAI backend does not support interventions")

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory with logprobs using OpenAI API.

        Args:
            token_ids: Input token IDs (will be decoded to text)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            Tuple of (all_token_ids, logprobs)
        """
        client = self._get_client()

        # Decode input tokens to text
        prompt = self._tokenizer.decode(token_ids)

        # Use temperature=0 for greedy
        temp = temperature if temperature > 0 else 0

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temp,
            logprobs=True,
        )

        # Process response
        choice = response.choices[0]

        # Build token IDs and logprobs from API response
        # Input tokens have logprob=0.0 (not available from API)
        all_token_ids = list(token_ids)
        all_logprobs = [0.0] * len(token_ids)

        # Extract tokens and logprobs from API response
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                # Get the token bytes and encode to get token ID
                token_bytes = token_info.bytes
                if token_bytes:
                    try:
                        token_str = bytes(token_bytes).decode("utf-8")
                        token_id = self._tokenizer.encode(token_str)
                        if token_id:
                            all_token_ids.append(token_id[0])
                            all_logprobs.append(token_info.logprob)
                    except (UnicodeDecodeError, IndexError):
                        pass
        else:
            # Fallback: tokenize the text and use 0.0 logprobs
            generated_text = choice.message.content or ""
            generated_ids = self._tokenizer.encode(generated_text)
            all_token_ids.extend(generated_ids)
            all_logprobs.extend([0.0] * len(generated_ids))

        return all_token_ids, all_logprobs
