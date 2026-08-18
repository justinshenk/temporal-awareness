"""Selective last-query-token attention capture (memory-light, model-family agnostic).

Hooks ``q_proj``/``k_proj`` at the target layers and recomputes attention (post-RoPE, GQA-aware)
for **only the last query token**, avoiding materialization of full N×N attention at every layer.

The reconstruction is family-agnostic by *introspection* rather than by branching on model type:

- **QK-norm** (``q_norm``/``k_norm``) is applied when the attention module exposes it. OLMo-2 does;
  Qwen2 does not. Omitting it does not perturb the result slightly — it collapses attention to
  uniform, because RMSNorm is what puts the projections on a scale where q·k separates at all.
- **GQA** is handled by expanding K to the query heads whenever the model has fewer key/value
  heads than query heads. Head counts are derived from the projection widths, not from config
  field names, so a family that names them differently still works.
- ``apply_rotary_pos_emb`` is resolved from the attention module's own defining module, so each
  family gets its own RoPE implementation.

Capture only fires while ``enabled`` is True and the forward is a prefill (``seq_len > 1``), so a
single ``model.generate(..., output_scores=True)`` call yields both the prefill's last-token
attention (here) and the own-confidence entropy (from the generation scores) — decode steps
(seq_len == 1) are skipped automatically.

Correctness condition: agreement with ``output_attentions``. See
``tests/probes/context_fatigue/test_attention_capture.py``, which pins both families against it.
"""

from __future__ import annotations

import importlib

import torch


def _resolve_rope(attn_module):
    """The ``apply_rotary_pos_emb`` belonging to this attention module's own family."""
    family = importlib.import_module(type(attn_module).__module__)
    rope = getattr(family, "apply_rotary_pos_emb", None)
    if rope is None:
        raise AttributeError(
            f"{type(attn_module).__module__} exposes no apply_rotary_pos_emb; "
            "the capture cannot reconstruct post-RoPE attention for this family.")
    return rope


class SelectiveAttentionCapture:
    """Capture ``layer -> (n_heads, seq_len)`` last-token attention at target layers."""

    def __init__(self, model, target_layers):
        self.target_layers = list(target_layers)
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        self.enabled = False
        for li in self.target_layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(
                attn.register_forward_pre_hook(self._make_hook(li), with_kwargs=True))

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, kwargs):
            if not self.enabled:
                return
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return
            if hidden_states.shape[1] <= 1:  # cached decode step
                return
            with torch.no_grad():
                self.captured[layer_idx] = self._last_token_attention(
                    module, hidden_states, position_embeddings,
                    kwargs.get("attention_mask"))
        return hook_fn

    @staticmethod
    def _last_token_attention(module, hidden_states, position_embeddings, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        head_dim = module.head_dim

        q = module.q_proj(hidden_states)
        k = module.k_proj(hidden_states)
        # QK-norm before the head split, when the family uses it (OLMo-2 does, Qwen2 does not).
        if getattr(module, "q_norm", None) is not None:
            q = module.q_norm(q)
        if getattr(module, "k_norm", None) is not None:
            k = module.k_norm(k)

        n_q = q.shape[-1] // head_dim
        n_kv = k.shape[-1] // head_dim
        q = q.view(batch, seq_len, n_q, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _resolve_rope(module)(q, k, cos, sin)

        n_rep = n_q // n_kv
        if n_rep > 1:  # GQA: expand K to the query heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch, n_q, seq_len, head_dim)

        q_last = q[:, :, -1:, :]
        scores = torch.matmul(q_last, k.transpose(-2, -1)) * head_dim ** -0.5
        if attention_mask is not None:
            # The additive mask's last query row. On a plain causal forward this row is all zeros
            # and changes nothing; it matters when something has biased the mask — which is how
            # the span clamp intervenes, so without this the capture would report the *unclamped*
            # attention during an intervention. Shape [b, 1, 1, k] broadcasts over heads.
            scores = scores + attention_mask[:, :, -1:, :].to(scores.dtype)
        weights = torch.softmax(scores.float(), dim=-1)
        return weights[0, :, 0, :].cpu()

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def attention_distribution_entropy(attn_vec) -> float:
    """Shannon entropy (nats) of one last-token attention distribution over key positions."""
    p = torch.as_tensor(attn_vec, dtype=torch.float32)
    p = p / p.sum().clamp_min(1e-12)
    return float(-(p * p.clamp_min(1e-12).log()).sum())
