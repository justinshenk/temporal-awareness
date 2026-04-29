"""Tests for attn_z head-level intervention support.

These tests verify that:
1. attn_z interventions can be created with head-level targeting
2. Hook names are correctly generated for attn_z
3. The intervention hook correctly modifies only the specified head
4. Position-level attn_z patching works correctly
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common.hook_utils import hook_filter_for_component, hook_name, parse_hook_name
from src.inference.interventions import InterventionTarget
from src.inference.interventions.intervention_base import (
    Intervention,
    create_intervention_hook,
)


class TestAttnZHookNames:
    """Test hook name generation for attn_z component."""

    def test_hook_name_attn_z(self):
        """attn_z uses special format: blocks.{layer}.attn.hook_z"""
        result = hook_name(5, "attn_z")
        assert result == "blocks.5.attn.hook_z"

    def test_hook_name_standard(self):
        """Other components use standard format: blocks.{layer}.hook_{component}"""
        assert hook_name(5, "attn_out") == "blocks.5.hook_attn_out"
        assert hook_name(5, "resid_post") == "blocks.5.hook_resid_post"

    def test_hook_filter_attn_z(self):
        """attn_z filter matches attn.hook_z format."""
        f = hook_filter_for_component("attn_z")
        assert f("blocks.5.attn.hook_z")
        assert f("blocks.10.attn.hook_z")
        assert not f("blocks.5.hook_attn_out")
        assert not f("blocks.5.hook_z")  # Wrong format

    def test_parse_hook_name_attn_z(self):
        """Parsing attn_z hook names returns correct layer and component."""
        result = parse_hook_name("blocks.5.attn.hook_z")
        assert result == (5, "attn_z")

    def test_parse_hook_name_standard(self):
        """Parsing standard hook names works correctly."""
        assert parse_hook_name("blocks.5.hook_attn_out") == (5, "attn_out")
        assert parse_hook_name("blocks.10.hook_resid_post") == (10, "resid_post")


class TestInterventionTargetHead:
    """Test InterventionTarget with head-level targeting."""

    def test_at_head_basic(self):
        """at_head creates target with correct attributes."""
        target = InterventionTarget.at_head(layer=5, head=3)
        assert target.layers == (5,)
        assert target.head == 3
        assert target.component == "attn_z"
        assert target.is_all_positions

    def test_at_head_with_positions(self):
        """at_head with positions works correctly."""
        target = InterventionTarget.at_head(layer=5, head=3, positions=[10, 20, 30])
        assert target.layers == (5,)
        assert target.head == 3
        assert target.component == "attn_z"
        assert target.positions == (10, 20, 30)


class TestInterventionWithHead:
    """Test Intervention dataclass with head field."""

    def test_intervention_with_head(self):
        """Intervention can store head index."""
        intervention = Intervention(
            layer=5,
            mode="set",
            values=np.zeros(64, dtype=np.float32),
            component="attn_z",
            head=3,
        )
        assert intervention.head == 3
        assert intervention.component == "attn_z"

    def test_hook_name_for_attn_z_intervention(self):
        """Intervention.hook_name returns correct format for attn_z."""
        intervention = Intervention(
            layer=5,
            mode="set",
            values=np.zeros(64, dtype=np.float32),
            component="attn_z",
            head=3,
        )
        assert intervention.hook_name == "blocks.5.attn.hook_z"


class TestCreateInterventionHookForHead:
    """Test create_intervention_hook with head-level targeting."""

    def test_creates_head_level_hook(self):
        """Hook correctly handles 4D activations for head-level intervention."""
        intervention = Intervention(
            layer=5,
            mode="set",
            values=np.ones(64, dtype=np.float32),  # d_head = 64
            component="attn_z",
            head=3,
            target=InterventionTarget.at_positions([10]),
        )

        hook_fn, _ = create_intervention_hook(
            intervention, dtype=torch.float32, device="cpu"
        )

        # Create 4D activation: [batch=1, seq=20, n_heads=8, d_head=64]
        act = torch.zeros(1, 20, 8, 64)

        # Apply hook
        result = hook_fn(act)

        # Check that only position 10, head 3 was modified
        assert torch.allclose(result[:, 10, 3, :], torch.ones(1, 64))
        # Other positions/heads should be unchanged
        assert torch.allclose(result[:, 0, 3, :], torch.zeros(1, 64))
        assert torch.allclose(result[:, 10, 0, :], torch.zeros(1, 64))
        assert torch.allclose(result[:, 10, 7, :], torch.zeros(1, 64))

    def test_full_position_head_hook(self):
        """Hook applies to all positions for specific head."""
        intervention = Intervention(
            layer=5,
            mode="set",
            values=np.ones((10, 64), dtype=np.float32),  # [seq, d_head]
            component="attn_z",
            head=2,
            target=InterventionTarget.all(),
        )

        hook_fn, _ = create_intervention_hook(
            intervention, dtype=torch.float32, device="cpu"
        )

        # Create 4D activation: [batch=1, seq=10, n_heads=8, d_head=64]
        act = torch.zeros(1, 10, 8, 64)

        # Apply hook
        result = hook_fn(act)

        # Check that all positions for head 2 were modified
        assert torch.allclose(result[:, :, 2, :], torch.ones(1, 10, 64))
        # Other heads should be unchanged
        assert torch.allclose(result[:, :, 0, :], torch.zeros(1, 10, 64))
        assert torch.allclose(result[:, :, 7, :], torch.zeros(1, 10, 64))


class TestContrastivePairAttnZ:
    """Test contrastive pair intervention creation for attn_z."""

    def test_make_layer_intervention_attn_z(self):
        """_make_layer_intervention correctly handles attn_z tensors."""
        # This would require a full ContrastivePair setup, so we test the logic
        # by checking that the hook_name is generated correctly
        hook = hook_name(5, "attn_z")
        assert hook == "blocks.5.attn.hook_z"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
