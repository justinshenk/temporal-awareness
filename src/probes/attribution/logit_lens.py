"""Logit-lens readout of intermediate residual streams.

Projects a decoder layer's residual-stream output through the model's *own* final norm and unembedding
(``lm_head``), so the vocabulary distribution a layer "would emit if decoding stopped there" is read
with the exact head the model uses — not a re-derived unembed. Used to ask *where* an answer token
crystallizes across depth: at the patched layer already (the answer is present and downstream layers
transcribe it) or only after several more layers (downstream layers compute it).

Llama layout: ``model.model.norm`` is the final RMSNorm, ``model.lm_head`` the unembedding. The lens of
the final decoder layer's residual equals the model's real logits (validated in tests), which is what
makes the per-layer ranks comparable to the model's actual decision.
"""

from __future__ import annotations

import torch


class LogitLens:
    """Projects residual rows through ``final_norm`` + ``lm_head`` and ranks a target token.

    Constructed from a Llama-style causal LM; holds references to its final norm and unembedding (no
    copies, so tied weights are honored). :meth:`project` maps ``(n, d)`` residual rows to ``(n, vocab)``
    logits; :meth:`rank_of` returns the 0-based rank of ``token_id`` (0 = argmax) and its logit, for the
    chosen row of a residual tensor.
    """

    def __init__(self, model):
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    @torch.no_grad()
    def project(self, residual: torch.Tensor) -> torch.Tensor:
        w = self.lm_head.weight
        h = residual.to(device=w.device, dtype=w.dtype)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def rank_of(self, residual: torch.Tensor, token_id: int, row: int = -1) -> tuple[int, float]:
        logits = self.project(residual)[row]
        target = float(logits[token_id])
        rank = int((logits > target).sum())                # how many tokens outrank the target
        return rank, target
