"""Shared tokenizer interface for API-based backends (OpenAI, Anthropic)."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class APITokenizer:
    """Minimal tokenizer interface wrapping tiktoken for API-based models.

    Both OpenAI and Anthropic backends use this for token counting and
    encoding/decoding. The encoding is an approximation since API providers
    don't expose their exact tokenizers.

    Attributes:
        encoding_name: Name of tiktoken encoding to use.
    """

    encoding_name: str = "cl100k_base"  # Works for most models

    def __post_init__(self):
        self._encoding = tiktoken.get_encoding(self.encoding_name)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    @property
    def bos_token_id(self) -> int | None:
        return None  # API models don't expose BOS

    @property
    def eos_token_id(self) -> int | None:
        return self._encoding.eot_token

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def bos_token(self) -> str | None:
        return None

    @property
    def eos_token(self) -> str | None:
        return "<|endoftext|>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._encoding.decode(token_ids)
