"""Selective last-query-token attention capture for Qwen2 (memory-light).

Adapted from the inline capture in ``scripts/context_fatigue/run_ddxplus_attention.py``
so it can be reused for the WildChat dynamics run. Hooks ``q_proj``/``k_proj`` at the
target layers and recomputes attention (post-RoPE, GQA-aware) for **only the last query
token**, avoiding materialization of full N×N attention at every layer.

Capture only fires while ``enabled`` is True and the forward is a prefill
(``seq_len > 1``), so a single ``model.generate(..., output_scores=True)`` call yields
both the prefill's last-token attention (here) and the own-confidence entropy (from the
generation scores) — the decode steps (seq_len == 1) are skipped automatically.
"""

from __future__ import annotations

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class SelectiveAttentionCapture:
    """Capture ``layer -> (n_heads, seq_len)`` last-token attention at target layers."""

    def __init__(self, model, target_layers):
        self.target_layers = set(target_layers)
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        self.enabled = False
        for li in target_layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(
                attn.register_forward_pre_hook(self._make_hook(li), with_kwargs=True))

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, kwargs):
            if not self.enabled:
                return
            hidden_states = kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return
            if hidden_states.shape[1] <= 1:  # cached decode step
                return
            with torch.no_grad():
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                batch, seq_len, _ = hidden_states.shape
                head_dim = module.head_dim
                n_q = module.config.num_attention_heads
                n_kv = module.config.num_key_value_heads
                q = q.view(batch, seq_len, n_q, head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                n_rep = n_q // n_kv
                if n_rep > 1:
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                        batch, n_q, seq_len, head_dim)
                q_last = q[:, :, -1:, :]
                scores = torch.matmul(q_last, k.transpose(-2, -1)) * head_dim ** -0.5
                weights = torch.softmax(scores.float(), dim=-1)
                self.captured[layer_idx] = weights[0, :, 0, :].cpu()
                del q, k, q_last, scores, weights
        return hook_fn

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


def attention_distribution_entropy(attn_vec) -> float:
    """Shannon entropy (nats) of one last-token attention distribution over key positions."""
    p = torch.as_tensor(attn_vec, dtype=torch.float32)
    p = p / p.sum().clamp_min(1e-12)
    return float(-(p * p.clamp_min(1e-12).log()).sum())
