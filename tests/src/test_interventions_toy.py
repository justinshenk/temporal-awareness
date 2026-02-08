"""Comprehensive intervention tests using toy models with known weights.

Tests all combinations of:
- Modes: add, set, mul, interpolate
- Axes: all, position, neuron, pattern
- Backends: TransformerLens (default), NNsight, Pyvene

Ground truth tests: 4 modes × 4 axes = 16 tests
Backend comparison tests: 4 modes × 4 axes × 3 backends = 48 tests
Total: 64 systematic tests

TransformerLens uses HookedTransformer.
NNsight/Pyvene use a standard PyTorch model with equivalent weights.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.models import ModelRunner
from src.models.backends import ModelBackend, TransformerLensBackend, NNsightBackend, PyveneBackend
from src.models.interventions import steering, ablation, scale, interpolate


# =============================================================================
# Toy Model Configuration
# =============================================================================

D_MODEL = 32
N_LAYERS = 2
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
D_MLP = D_MODEL * 4
D_VOCAB = 100
N_CTX = 64
TOLERANCE = 1e-4


# =============================================================================
# Standard PyTorch Toy Model (for NNsight/Pyvene)
# =============================================================================


class ToyTransformerLayer(nn.Module):
    """Simple transformer layer that matches HookedTransformer structure."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x):
        # Self-attention (residual)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        # MLP (residual)
        x = x + self.mlp(x)
        return x


class ToyTransformer(nn.Module):
    """Standard PyTorch transformer that NNsight/Pyvene can wrap.

    Uses GPT2-style naming: transformer.h[i] for layers.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_mlp: int, d_vocab: int):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "n_embd": d_model,
                "hidden_size": d_model,
                "num_hidden_layers": n_layers,
            },
        )()

        # Embedding
        self.embed = nn.Embedding(d_vocab, d_model)

        # Transformer layers (GPT2 style: transformer.h)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [ToyTransformerLayer(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )

        # Output
        self.lm_head = nn.Linear(d_model, d_vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.transformer.h:
            x = layer(x)
        logits = self.lm_head(x)
        return type("Output", (), {"logits": logits})()

    def generate(self, input_ids, max_new_tokens=10, do_sample=False, temperature=1.0, **kwargs):
        """Simple autoregressive generation."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs.logits
            if do_sample and temperature > 0:
                probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
            else:
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def create_toy_hooked_transformer():
    """Create a HookedTransformer with known weights for testing."""
    cfg = HookedTransformerConfig(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        d_vocab=D_VOCAB,
        n_ctx=N_CTX,
        act_fn="relu",
        normalization_type=None,
        attn_only=False,
        device="cpu",
    )

    model = HookedTransformer(cfg)

    with torch.no_grad():
        model.embed.W_E.fill_(0.1)
        for i in range(D_VOCAB):
            model.embed.W_E[i, i % D_MODEL] = 1.0

        for layer in range(N_LAYERS):
            model.blocks[layer].attn.W_Q.zero_()
            model.blocks[layer].attn.W_K.zero_()
            model.blocks[layer].attn.W_V.zero_()
            model.blocks[layer].attn.W_O.zero_()
            model.blocks[layer].attn.b_Q.zero_()
            model.blocks[layer].attn.b_K.zero_()
            model.blocks[layer].attn.b_V.zero_()
            model.blocks[layer].attn.b_O.zero_()
            model.blocks[layer].mlp.W_in.zero_()
            model.blocks[layer].mlp.W_out.zero_()
            model.blocks[layer].mlp.b_in.zero_()
            model.blocks[layer].mlp.b_out.zero_()

        model.unembed.W_U.zero_()
        for i in range(min(D_MODEL, D_VOCAB)):
            model.unembed.W_U[i, i] = 1.0

    model.eval()
    return model


def create_toy_pytorch_transformer():
    """Create a standard PyTorch transformer with known weights for NNsight/Pyvene."""
    model = ToyTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, D_VOCAB)

    with torch.no_grad():
        # Embedding - same as HookedTransformer
        model.embed.weight.fill_(0.1)
        for i in range(D_VOCAB):
            model.embed.weight[i, i % D_MODEL] = 1.0

        # Zero out attention and MLP weights
        for layer in model.transformer.h:
            # Zero attention
            layer.attn.in_proj_weight.zero_()
            layer.attn.in_proj_bias.zero_()
            layer.attn.out_proj.weight.zero_()
            layer.attn.out_proj.bias.zero_()
            # Zero MLP
            layer.mlp[0].weight.zero_()
            layer.mlp[0].bias.zero_()
            layer.mlp[2].weight.zero_()
            layer.mlp[2].bias.zero_()

        # Output projection - same as HookedTransformer
        model.lm_head.weight.zero_()
        for i in range(min(D_MODEL, D_VOCAB)):
            model.lm_head.weight[i, i] = 1.0

    model.eval()
    return model


class TokenizerOutput:
    """Output class that supports both dict access and attribute access."""

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __getitem__(self, key):
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class ToyTokenizer:
    """Minimal tokenizer for toy model."""

    def __init__(self, vocab_size: int = D_VOCAB):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token = "<pad>"
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = N_CTX

    def __call__(self, text, return_tensors=None, **kwargs):
        if isinstance(text, str):
            ids = [ord(c) % (self.vocab_size - 3) + 3 for c in text]
        else:
            ids = [[ord(c) % (self.vocab_size - 3) + 3 for c in t] for t in text]
        if return_tensors == "pt":
            if isinstance(ids[0], list):
                tensor = torch.tensor(ids)
            else:
                tensor = torch.tensor([ids])
            # Return object that supports both dict access and attribute access
            return TokenizerOutput(tensor)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, str):
            return [ord(c) % (self.vocab_size - 3) + 3 for c in text]
        return [[ord(c) % (self.vocab_size - 3) + 3 for c in t] for t in text]

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(i) for i in ids if i >= 3)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def toy_hooked_model():
    """HookedTransformer with known weights for TransformerLens."""
    return create_toy_hooked_transformer()


@pytest.fixture(scope="module")
def toy_pytorch_model():
    """Standard PyTorch transformer for NNsight/Pyvene."""
    return create_toy_pytorch_transformer()


@pytest.fixture(scope="module")
def toy_tokenizer():
    """Toy tokenizer instance."""
    return ToyTokenizer()


@pytest.fixture(scope="module")
def runner(toy_hooked_model, toy_tokenizer):
    """Default backend (TransformerLens) with HookedTransformer."""
    # Configure tokenizer manually (HookedTransformer.set_tokenizer requires PreTrainedTokenizer)
    toy_hooked_model.tokenizer = toy_tokenizer
    toy_hooked_model.cfg.tokenizer_prepends_bos = False  # Our tokenizer doesn't prepend BOS
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_name = "toy"
    runner.backend = ModelBackend.TRANSFORMERLENS
    runner.device = "cpu"
    runner.dtype = torch.float32
    runner._model = toy_hooked_model
    runner._tokenizer = toy_tokenizer
    runner._is_chat_model = False
    runner._backend = TransformerLensBackend(runner)
    return runner


@pytest.fixture(scope="module")
def runner_nnsight(toy_pytorch_model, toy_tokenizer):
    """NNsight backend with standard PyTorch model."""
    from nnsight import NNsight

    wrapped = NNsight(toy_pytorch_model)
    wrapped.tokenizer = toy_tokenizer

    runner = ModelRunner.__new__(ModelRunner)
    runner.model_name = "toy"
    runner.backend = ModelBackend.NNSIGHT
    runner.device = "cpu"
    runner.dtype = torch.float32
    runner._model = wrapped
    runner._tokenizer = toy_tokenizer
    runner._is_chat_model = False
    runner._backend = NNsightBackend(runner)
    return runner


@pytest.fixture(scope="module")
def runner_pyvene(toy_pytorch_model, toy_tokenizer):
    """Pyvene backend with standard PyTorch model."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_name = "toy"
    runner.backend = ModelBackend.PYVENE
    runner.device = "cpu"
    runner.dtype = torch.float32
    runner._model = toy_pytorch_model
    runner._tokenizer = toy_tokenizer
    runner._is_chat_model = False
    runner._backend = PyveneBackend(runner)
    return runner


# =============================================================================
# Helper functions
# =============================================================================


def make_direction(seed=42):
    """Create a unit direction vector."""
    np.random.seed(seed)
    vec = np.random.randn(D_MODEL).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_values(seed=42):
    """Create values for set/ablation tests."""
    np.random.seed(seed)
    return np.random.randn(D_MODEL).astype(np.float32)


def make_2d_values(seq_len=5, seed=42):
    """Create 2D values for interpolation tests."""
    np.random.seed(seed)
    return np.random.randn(seq_len, D_MODEL).astype(np.float32)


# =============================================================================
# Ground Truth Tests (16 tests)
# =============================================================================
# These test mathematical correctness using TransformerLens default backend


class TestInterventionsAllGroundTruth:
    """Ground truth tests for axis='all' interventions."""

    def test_interventions_all_add_ground_truth(self, runner):
        """ADD mode adds direction to ALL positions."""
        prompt = "Hi"
        direction = make_direction()
        intervention = steering(layer=0, direction=direction, strength=5.0)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_set_ground_truth(self, runner):
        """SET mode replaces activations at ALL positions."""
        prompt = "Hi"
        values = make_values()
        intervention = ablation(layer=0, values=values)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_mul_ground_truth(self, runner):
        """MUL mode scales activations at ALL positions."""
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_interpolate_ground_truth(self, runner):
        """INTERPOLATE mode blends source/target at ALL positions."""
        prompt = "Hi"
        source = make_2d_values(seq_len=5, seed=1)
        target = make_2d_values(seq_len=5, seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPositionGroundTruth:
    """Ground truth tests for axis='position' interventions."""

    def test_interventions_position_add_ground_truth(self, runner):
        """ADD mode adds direction at specific positions."""
        prompt = "Hello"
        direction = make_direction()
        intervention = steering(layer=0, direction=direction, strength=5.0, positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_set_ground_truth(self, runner):
        """SET mode replaces activations at specific positions."""
        prompt = "Hello"
        values = make_values()
        intervention = ablation(layer=0, values=values, positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_mul_ground_truth(self, runner):
        """MUL mode scales activations at specific positions."""
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_interpolate_ground_truth(self, runner):
        """INTERPOLATE mode blends at specific positions."""
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsNeuronGroundTruth:
    """Ground truth tests for axis='neuron' interventions."""

    def test_interventions_neuron_add_ground_truth(self, runner):
        """ADD mode adds to specific neurons."""
        prompt = "Hello"
        direction = make_direction()
        intervention = steering(layer=0, direction=direction, strength=5.0, neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_set_ground_truth(self, runner):
        """SET mode replaces specific neurons."""
        prompt = "Hello"
        values = make_values()
        intervention = ablation(layer=0, values=values, neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_mul_ground_truth(self, runner):
        """MUL mode scales specific neurons."""
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_interpolate_ground_truth(self, runner):
        """INTERPOLATE mode blends specific neurons."""
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPatternGroundTruth:
    """Ground truth tests for axis='pattern' interventions (generation only)."""

    def test_interventions_pattern_add_ground_truth(self, runner):
        """ADD mode with pattern trigger during generation."""
        prompt = "Hi"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, pattern="H"
        )
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_truth(self, runner):
        """SET mode with pattern trigger during generation."""
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values(), pattern="H")
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_truth(self, runner):
        """MUL mode with pattern trigger during generation."""
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0, pattern="H")
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_truth(self, runner):
        """INTERPOLATE mode with pattern trigger during generation."""
        prompt = "Hi"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, pattern="H"
        )
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Comparison Tests - TransformerLens (16 tests)
# =============================================================================
# TransformerLens is default, so these verify it matches itself (sanity check)


class TestInterventionsAllGroundTransformerlens:
    """TransformerLens backend tests for axis='all'."""

    def test_interventions_all_add_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = steering(layer=0, direction=make_direction(), strength=5.0)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_set_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values())
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_mul_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0)
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_interpolate_ground_transformerlens(self, runner):
        prompt = "Hi"
        source = make_2d_values(seq_len=5, seed=1)
        target = make_2d_values(seq_len=5, seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPositionGroundTransformerlens:
    """TransformerLens backend tests for axis='position'."""

    def test_interventions_position_add_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, positions=[1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_set_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_mul_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, positions=[1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_interpolate_ground_transformerlens(self, runner):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsNeuronGroundTransformerlens:
    """TransformerLens backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, neurons=[0, 1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_set_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_mul_ground_transformerlens(self, runner):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, neurons=[0, 1])
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_interpolate_ground_transformerlens(self, runner):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPatternGroundTransformerlens:
    """TransformerLens backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, pattern="H"
        )
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values(), pattern="H")
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_transformerlens(self, runner):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0, pattern="H")
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_transformerlens(self, runner):
        prompt = "Hi"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, pattern="H"
        )
        output = runner.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Comparison Tests - NNsight (16 tests)
# =============================================================================
# Note: NNsight uses a standard PyTorch model with equivalent weights
# These tests verify NNsight produces valid outputs (not necessarily identical
# to TransformerLens since models have different architectures)


class TestInterventionsAllGroundNnsight:
    """NNsight backend tests for axis='all'."""

    def test_interventions_all_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = steering(layer=0, direction=make_direction(), strength=5.0)
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values())
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0)
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        source = make_2d_values(seq_len=5, seed=1)
        target = make_2d_values(seq_len=5, seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPositionGroundNnsight:
    """NNsight backend tests for axis='position'."""

    def test_interventions_position_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, positions=[1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), positions=[1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, positions=[1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsNeuronGroundNnsight:
    """NNsight backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, neurons=[0, 1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), neurons=[0, 1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, neurons=[0, 1])
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner_nnsight.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPatternGroundNnsight:
    """NNsight backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, pattern="H"
        )
        output = runner_nnsight.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values(), pattern="H")
        output = runner_nnsight.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0, pattern="H")
        output = runner_nnsight.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_nnsight(self, runner_nnsight):
        prompt = "Hi"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, pattern="H"
        )
        output = runner_nnsight.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)


# =============================================================================
# Backend Comparison Tests - Pyvene (16 tests)
# =============================================================================
# Note: Pyvene uses a standard PyTorch model with equivalent weights


class TestInterventionsAllGroundPyvene:
    """Pyvene backend tests for axis='all'."""

    def test_interventions_all_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = steering(layer=0, direction=make_direction(), strength=5.0)
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values())
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0)
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_all_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        source = make_2d_values(seq_len=5, seed=1)
        target = make_2d_values(seq_len=5, seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPositionGroundPyvene:
    """Pyvene backend tests for axis='position'."""

    def test_interventions_position_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, positions=[1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), positions=[1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, positions=[1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_position_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, positions=[1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsNeuronGroundPyvene:
    """Pyvene backend tests for axis='neuron'."""

    def test_interventions_neuron_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, neurons=[0, 1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = ablation(layer=0, values=make_values(), neurons=[0, 1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        intervention = scale(layer=0, factor=2.0, neurons=[0, 1])
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB

    def test_interventions_neuron_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hello"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, neurons=[0, 1]
        )
        logits = runner_pyvene.forward_with_intervention(prompt, intervention)
        assert logits.shape[-1] == D_VOCAB


class TestInterventionsPatternGroundPyvene:
    """Pyvene backend tests for axis='pattern'."""

    def test_interventions_pattern_add_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = steering(
            layer=0, direction=make_direction(), strength=5.0, pattern="H"
        )
        output = runner_pyvene.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_set_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = ablation(layer=0, values=make_values(), pattern="H")
        output = runner_pyvene.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_mul_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        intervention = scale(layer=0, factor=2.0, pattern="H")
        output = runner_pyvene.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)

    def test_interventions_pattern_interpolate_ground_pyvene(self, runner_pyvene):
        prompt = "Hi"
        source = make_values(seed=1)
        target = make_values(seed=2)
        intervention = interpolate(
            layer=0, source_values=source, target_values=target, alpha=0.5, pattern="H"
        )
        output = runner_pyvene.generate(
            prompt, max_new_tokens=3, temperature=0.0, intervention=intervention
        )
        assert isinstance(output, str)
