"""Tests for activation patching metric calculation.

Tests PatchingMetric recovery computation, label matching across formats,
position mapping, and create_metric integration.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.analysis.patching import PatchingMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metric(
    *,
    clean_short_id=10,
    clean_long_id=20,
    corr_short_id=10,
    corr_long_id=20,
    clean_pos=5,
    corr_pos=5,
    clean_val=3.0,
    corr_val=-2.0,
    corr_val_clean_format=None,
):
    """Build a PatchingMetric with sensible defaults."""
    diff = clean_val - corr_val
    if corr_val_clean_format is None:
        corr_val_clean_format = corr_val
    return PatchingMetric(
        clean_short_id=clean_short_id,
        clean_long_id=clean_long_id,
        corr_short_id=corr_short_id,
        corr_long_id=corr_long_id,
        clean_pos=clean_pos,
        corr_pos=corr_pos,
        clean_val=clean_val,
        corr_val=corr_val,
        diff=diff,
        corr_val_clean_format=corr_val_clean_format,
    )


def _make_logits(vocab_size=100, seq_len=10):
    """Return zeroed logits tensor [1, seq_len, vocab_size]."""
    return torch.zeros(1, seq_len, vocab_size)


# ===========================================================================
# PatchingMetric.same_labels
# ===========================================================================


class TestSameLabels:
    def test_same_ids(self):
        m = _make_metric(clean_short_id=10, clean_long_id=20,
                         corr_short_id=10, corr_long_id=20)
        assert m.same_labels is True

    def test_different_short_id(self):
        m = _make_metric(clean_short_id=10, clean_long_id=20,
                         corr_short_id=30, corr_long_id=20)
        assert m.same_labels is False

    def test_different_long_id(self):
        m = _make_metric(clean_short_id=10, clean_long_id=20,
                         corr_short_id=10, corr_long_id=40)
        assert m.same_labels is False

    def test_both_different(self):
        m = _make_metric(clean_short_id=10, clean_long_id=20,
                         corr_short_id=30, corr_long_id=40)
        assert m.same_labels is False


# ===========================================================================
# PatchingMetric.__call__  —  same-label recovery
# ===========================================================================


class TestRecoverySameLabels:
    """When clean and corrupted use the same label tokens."""

    def test_corrupted_baseline_returns_zero(self):
        """Unpatched corrupted logits → recovery = 0."""
        m = _make_metric()
        logits = _make_logits()
        # Set logits so raw = corr_val = -2.0
        logits[0, m.corr_pos, m.corr_short_id] = 1.0
        logits[0, m.corr_pos, m.corr_long_id] = 3.0  # raw = 1-3 = -2
        assert m(logits) == pytest.approx(0.0)

    def test_clean_baseline_returns_one(self):
        """Fully recovered logits → recovery = 1."""
        m = _make_metric()
        logits = _make_logits()
        # raw should equal clean_val = 3.0
        logits[0, m.corr_pos, m.corr_short_id] = 5.0
        logits[0, m.corr_pos, m.corr_long_id] = 2.0  # raw = 5-2 = 3
        assert m(logits) == pytest.approx(1.0)

    def test_midpoint_returns_half(self):
        m = _make_metric()  # clean=3, corr=-2, diff=5
        logits = _make_logits()
        # raw = 0.5  →  (0.5 - (-2)) / 5 = 0.5
        logits[0, m.corr_pos, m.corr_short_id] = 2.0
        logits[0, m.corr_pos, m.corr_long_id] = 1.5
        assert m(logits) == pytest.approx(0.5)

    def test_overshoot_above_one(self):
        """Recovery can exceed 1.0 if patching overcorrects."""
        m = _make_metric()  # diff=5
        logits = _make_logits()
        # raw = 8 → (8 - (-2))/5 = 2.0
        logits[0, m.corr_pos, m.corr_short_id] = 10.0
        logits[0, m.corr_pos, m.corr_long_id] = 2.0
        assert m(logits) == pytest.approx(2.0)

    def test_negative_recovery(self):
        """Patching makes things worse → negative recovery."""
        m = _make_metric()  # diff=5
        logits = _make_logits()
        # raw = -5 → (-5 - (-2))/5 = -0.6
        logits[0, m.corr_pos, m.corr_short_id] = 0.0
        logits[0, m.corr_pos, m.corr_long_id] = 5.0
        assert m(logits) == pytest.approx(-0.6)

    def test_zero_diff_returns_zero(self):
        """When clean and corrupted have same logit diff, return 0."""
        m = _make_metric(clean_val=2.0, corr_val=2.0)
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 10.0
        assert m(logits) == 0.0


# ===========================================================================
# PatchingMetric.__call__  —  different-label recovery
# ===========================================================================


class TestRecoveryDifferentLabels:
    """When clean and corrupted use different label tokens."""

    def _make_diff_metric(self):
        """Metric where clean uses tokens 10/20, corrupted uses 30/40."""
        return _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=30, corr_long_id=40,
            clean_val=3.0,
            corr_val=-2.0,
            # Baseline for clean-format tokens on corrupted prompt
            corr_val_clean_format=-1.0,
        )

    def test_recovery_via_corrupted_format(self):
        """Recovery expressed through corrupted-format tokens only."""
        m = self._make_diff_metric()  # diff = 5
        logits = _make_logits()
        # Corrupted format: raw = 3 → (3 - (-2))/5 = 1.0
        logits[0, m.corr_pos, 30] = 5.0
        logits[0, m.corr_pos, 40] = 2.0
        # Clean format: raw = 0 → (0 - (-1))/5 = 0.2
        logits[0, m.corr_pos, 10] = 0.0
        logits[0, m.corr_pos, 20] = 0.0
        assert m(logits) == pytest.approx(1.0)  # max(1.0, 0.2) = 1.0

    def test_recovery_via_clean_format(self):
        """Recovery expressed through clean-format tokens only."""
        m = self._make_diff_metric()  # diff = 5
        logits = _make_logits()
        # Corrupted format: raw = -2 → (-2 - (-2))/5 = 0
        logits[0, m.corr_pos, 30] = 1.0
        logits[0, m.corr_pos, 40] = 3.0
        # Clean format: raw = 4 → (4 - (-1))/5 = 1.0
        logits[0, m.corr_pos, 10] = 6.0
        logits[0, m.corr_pos, 20] = 2.0
        assert m(logits) == pytest.approx(1.0)  # max(0, 1.0) = 1.0

    def test_max_of_both_formats(self):
        """Takes the max recovery across both formats."""
        m = self._make_diff_metric()  # diff = 5
        logits = _make_logits()
        # Corrupted format: raw = -0.5 → (-0.5 - (-2))/5 = 0.3
        logits[0, m.corr_pos, 30] = 1.0
        logits[0, m.corr_pos, 40] = 1.5
        # Clean format: raw = 2 → (2 - (-1))/5 = 0.6
        logits[0, m.corr_pos, 10] = 3.0
        logits[0, m.corr_pos, 20] = 1.0
        assert m(logits) == pytest.approx(0.6)  # max(0.3, 0.6)

    def test_both_negative_takes_less_negative(self):
        """When both formats show negative recovery, takes the less bad one."""
        m = self._make_diff_metric()  # diff = 5
        logits = _make_logits()
        # Corrupted format: raw = -4 → (-4 - (-2))/5 = -0.4
        logits[0, m.corr_pos, 30] = 0.0
        logits[0, m.corr_pos, 40] = 4.0
        # Clean format: raw = -2 → (-2 - (-1))/5 = -0.2
        logits[0, m.corr_pos, 10] = 0.0
        logits[0, m.corr_pos, 20] = 2.0
        assert m(logits) == pytest.approx(-0.2)  # max(-0.4, -0.2)

    def test_same_labels_skips_clean_format_check(self):
        """When labels are the same, only corrupted-format is checked."""
        m = _make_metric()  # same IDs
        assert m.same_labels is True
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 1.0
        logits[0, m.corr_pos, m.corr_long_id] = 3.0
        # Should just return corrupted-format recovery
        expected = (1.0 - 3.0 - m.corr_val) / m.diff
        assert m(logits) == pytest.approx(expected)


# ===========================================================================
# PatchingMetric.compute_raw
# ===========================================================================


class TestComputeRaw:
    def test_returns_logit_diff_tensor(self):
        m = _make_metric()
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 7.0
        logits[0, m.corr_pos, m.corr_long_id] = 2.0
        raw = m.compute_raw(logits)
        assert isinstance(raw, torch.Tensor)
        assert raw.item() == pytest.approx(5.0)

    def test_gradient_flows(self):
        """compute_raw preserves gradient for attribution patching."""
        m = _make_metric()
        logits_data = _make_logits()
        logits_data[0, m.corr_pos, m.corr_short_id] = 7.0
        logits_data[0, m.corr_pos, m.corr_long_id] = 2.0
        logits = logits_data.requires_grad_(True)
        raw = m.compute_raw(logits)
        raw.backward()
        assert logits.grad is not None
        assert logits.grad[0, m.corr_pos, m.corr_short_id] != 0
        assert logits.grad[0, m.corr_pos, m.corr_long_id] != 0

    def test_same_labels_uses_corr_tokens(self):
        """Same labels: compute_raw uses corrupted-format tokens."""
        m = _make_metric()  # same IDs
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 4.0
        logits[0, m.corr_pos, m.corr_long_id] = 1.0
        assert m.compute_raw(logits).item() == pytest.approx(3.0)

    def test_different_labels_takes_max(self):
        """Different labels: compute_raw takes max of both formats."""
        m = _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=30, corr_long_id=40,
        )
        logits = _make_logits()
        # Corrupted format: 30-40 = 1-3 = -2
        logits[0, m.corr_pos, 30] = 1.0
        logits[0, m.corr_pos, 40] = 3.0
        # Clean format: 10-20 = 5-1 = 4
        logits[0, m.corr_pos, 10] = 5.0
        logits[0, m.corr_pos, 20] = 1.0
        # max(-2, 4) = 4
        assert m.compute_raw(logits).item() == pytest.approx(4.0)

    def test_different_labels_corr_format_wins(self):
        """When corrupted format has higher logit diff, use that."""
        m = _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=30, corr_long_id=40,
        )
        logits = _make_logits()
        # Corrupted format: 30-40 = 6-1 = 5
        logits[0, m.corr_pos, 30] = 6.0
        logits[0, m.corr_pos, 40] = 1.0
        # Clean format: 10-20 = 2-1 = 1
        logits[0, m.corr_pos, 10] = 2.0
        logits[0, m.corr_pos, 20] = 1.0
        # max(5, 1) = 5
        assert m.compute_raw(logits).item() == pytest.approx(5.0)

    def test_different_labels_gradient_flows_through_max(self):
        """Gradient flows through the dominant format in compute_raw."""
        m = _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=30, corr_long_id=40,
        )
        logits_data = _make_logits()
        # Clean format wins: 10-20 = 5-1 = 4 > corr format -2
        logits_data[0, m.corr_pos, 30] = 1.0
        logits_data[0, m.corr_pos, 40] = 3.0
        logits_data[0, m.corr_pos, 10] = 5.0
        logits_data[0, m.corr_pos, 20] = 1.0
        logits = logits_data.requires_grad_(True)
        raw = m.compute_raw(logits)
        raw.backward()
        # Gradient should flow through clean-format tokens (the max path)
        assert logits.grad[0, m.corr_pos, 10] != 0  # clean_short
        assert logits.grad[0, m.corr_pos, 20] != 0  # clean_long
        # Corrupted-format tokens should have zero gradient (not the max)
        assert logits.grad[0, m.corr_pos, 30] == 0
        assert logits.grad[0, m.corr_pos, 40] == 0


# ===========================================================================
# Recovery at correct position
# ===========================================================================


class TestPositionUsage:
    """Metric reads logits at corr_pos, not other positions."""

    def test_uses_corr_pos(self):
        m = _make_metric(corr_pos=3)
        logits = _make_logits()
        # Set values at corr_pos=3
        logits[0, 3, m.corr_short_id] = 5.0
        logits[0, 3, m.corr_long_id] = 2.0  # raw = 3 = clean_val → recovery = 1
        assert m(logits) == pytest.approx(1.0)

    def test_ignores_other_positions(self):
        m = _make_metric(corr_pos=3)
        logits = _make_logits()
        # Set strong signal at wrong position
        logits[0, 7, m.corr_short_id] = 100.0
        # corr_pos=3 still has zeros → raw = 0, recovery = (0 - (-2))/5 = 0.4
        assert m(logits) == pytest.approx(0.4)


# ===========================================================================
# Label format scenarios from real data
# ===========================================================================


@dataclass
class MockPreferenceItem:
    """Mock preference item for testing."""
    short_term_label: str
    long_term_label: str
    choice: str
    response: str = ""
    prompt_text: str = ""
    sample_idx: int = 0
    time_horizon: dict = None
    choice_prob: float = 0.8
    alt_prob: float = 0.2


class TestLabelFormats:
    """Test that different label format combinations work correctly."""

    LABEL_PAIRS = [
        ("(1)", "(2)"),
        ("a)", "b)"),
        ("OPTION_ONE:", "OPTION_TWO:"),
        ("Option A:", "Option B:"),
        ("[I]", "[II]"),
        ("Choice 1:", "Choice 2:"),
        ("(A)", "(B)"),
    ]

    @pytest.mark.parametrize("short,long", LABEL_PAIRS)
    def test_same_format_pair(self, short, long):
        """Same-format clean/corrupted → same_labels=True."""
        m = _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=10, corr_long_id=20,
        )
        assert m.same_labels is True

    def test_cross_format_pair(self):
        """Different-format clean/corrupted → same_labels=False."""
        m = _make_metric(
            clean_short_id=10, clean_long_id=20,
            corr_short_id=30, corr_long_id=40,
        )
        assert m.same_labels is False

    def test_swapped_label_assignment(self):
        """Label assignment can swap: OPTION_TWO can be short_term."""
        clean = MockPreferenceItem(
            short_term_label="OPTION_TWO:",
            long_term_label="OPTION_ONE:",
            choice="short_term",
        )
        corrupted = MockPreferenceItem(
            short_term_label="OPTION_ONE:",
            long_term_label="OPTION_TWO:",
            choice="long_term",
        )
        # Verify labels differ despite same format family
        assert clean.short_term_label != corrupted.short_term_label


# ===========================================================================
# Normalization edge cases
# ===========================================================================


class TestNormalizationEdgeCases:
    def test_very_small_diff(self):
        """Near-zero diff returns 0 to avoid division explosion."""
        m = _make_metric(clean_val=1.0, corr_val=1.0 - 1e-10)
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 100.0
        assert m(logits) == 0.0

    def test_large_diff(self):
        m = _make_metric(clean_val=100.0, corr_val=-100.0)
        logits = _make_logits()
        logits[0, m.corr_pos, m.corr_short_id] = 50.0
        logits[0, m.corr_pos, m.corr_long_id] = 0.0
        # raw=50, recovery = (50 - (-100)) / 200 = 0.75
        assert m(logits) == pytest.approx(0.75)

    def test_negative_clean_val(self):
        """Clean can have negative val if model weakly prefers short."""
        m = _make_metric(clean_val=-1.0, corr_val=-5.0)  # diff = 4
        logits = _make_logits()
        # raw = -3 → (-3 - (-5))/4 = 0.5
        logits[0, m.corr_pos, m.corr_short_id] = 0.0
        logits[0, m.corr_pos, m.corr_long_id] = 3.0
        assert m(logits) == pytest.approx(0.5)


# ===========================================================================
# Pair building (prompt pairs)
# ===========================================================================


class TestPairBuilding:
    """Test that prompt pairs use correct labels."""

    def test_pairs_use_own_labels(self):
        clean = MockPreferenceItem(
            short_term_label="OPTION_TWO:",
            long_term_label="OPTION_ONE:",
            choice="short_term",
        )
        corrupted = MockPreferenceItem(
            short_term_label="[II]",
            long_term_label="[I]",
            choice="long_term",
        )
        assert clean.short_term_label != corrupted.short_term_label
        assert clean.long_term_label != corrupted.long_term_label

    def test_clean_is_short_term_choice(self):
        clean = MockPreferenceItem(
            short_term_label="a)", long_term_label="b)", choice="short_term"
        )
        assert clean.choice == "short_term"

    def test_corrupted_is_long_term_choice(self):
        corrupted = MockPreferenceItem(
            short_term_label="a)", long_term_label="b)", choice="long_term"
        )
        assert corrupted.choice == "long_term"


# ===========================================================================
# build_prompt_pairs same_labels filtering
# ===========================================================================


def _make_pref_item(sample_idx, short_label, long_label, choice, prompt="prompt"):
    """Shorthand for creating a PreferenceSample for pair-building tests."""
    from src.common.types import PreferenceSample
    response = f"I select: {short_label if choice == 'short_term' else long_label}"
    return PreferenceSample(
        sample_idx=sample_idx,
        time_horizon=None,
        short_term_label=short_label,
        long_term_label=long_label,
        short_term_reward=100.0,
        long_term_reward=200.0,
        short_term_time=1.0,
        long_term_time=12.0,
        choice=choice,
        choice_prob=0.8,
        alt_prob=0.2,
        response_text=response,
        prompt_text=prompt,
    )


def _make_pref_data(items):
    from src.preference import PreferenceDataset
    return PreferenceDataset(
        prompt_dataset_id="test", model="test", preferences=items
    )


class TestBuildPromptPairsSameLabels:
    """Test same_labels parameter of build_prompt_pairs."""

    def test_same_labels_true_filters_mismatched(self):
        """same_labels=True excludes pairs with different label formats."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(0, "a)", "b)", "short_term"),
            _make_pref_item(1, "(1)", "(2)", "short_term"),
            _make_pref_item(10, "a)", "b)", "long_term"),
            _make_pref_item(11, "X:", "Y:", "long_term"),
        ]
        pref_data = _make_pref_data(items)
        pairs = build_prompt_pairs(pref_data, max_pairs=10, same_labels=True)
        # Only the a)/b) pair should match
        assert len(pairs) == 1
        _, _, clean, corrupted = pairs[0]
        assert clean.short_term_label == corrupted.short_term_label == "a)"
        assert clean.long_term_label == corrupted.long_term_label == "b)"

    def test_same_labels_false_allows_mismatched(self):
        """same_labels=False pairs by position regardless of labels."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(0, "a)", "b)", "short_term"),
            _make_pref_item(10, "X:", "Y:", "long_term"),
        ]
        pref_data = _make_pref_data(items)
        pairs = build_prompt_pairs(pref_data, max_pairs=10, same_labels=False)
        assert len(pairs) == 1
        _, _, clean, corrupted = pairs[0]
        assert clean.short_term_label != corrupted.short_term_label

    def test_same_labels_multiple_groups(self):
        """Pairs are drawn from multiple label groups."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(0, "a)", "b)", "short_term"),
            _make_pref_item(1, "(1)", "(2)", "short_term"),
            _make_pref_item(10, "a)", "b)", "long_term"),
            _make_pref_item(11, "(1)", "(2)", "long_term"),
        ]
        pref_data = _make_pref_data(items)
        pairs = build_prompt_pairs(pref_data, max_pairs=10, same_labels=True)
        assert len(pairs) == 2
        labels_seen = set()
        for _, _, clean, corrupted in pairs:
            assert clean.short_term_label == corrupted.short_term_label
            assert clean.long_term_label == corrupted.long_term_label
            labels_seen.add(clean.short_term_label)
        assert labels_seen == {"a)", "(1)"}

    def test_same_labels_respects_max_pairs(self):
        """max_pairs caps the total even with multiple groups."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(i, "a)", "b)", "short_term") for i in range(5)
        ] + [
            _make_pref_item(i + 10, "a)", "b)", "long_term") for i in range(5)
        ]
        pref_data = _make_pref_data(items)
        pairs = build_prompt_pairs(pref_data, max_pairs=2, same_labels=True)
        assert len(pairs) == 2

    def test_same_labels_no_matching_groups(self):
        """Returns empty when no label groups have both choices."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(0, "a)", "b)", "short_term"),
            _make_pref_item(10, "X:", "Y:", "long_term"),
        ]
        pref_data = _make_pref_data(items)
        pairs = build_prompt_pairs(pref_data, max_pairs=10, same_labels=True)
        assert len(pairs) == 0

    def test_default_is_same_labels_true(self):
        """Default behavior filters to same labels."""
        from src.preference import build_prompt_pairs
        items = [
            _make_pref_item(0, "a)", "b)", "short_term"),
            _make_pref_item(10, "X:", "Y:", "long_term"),
        ]
        pref_data = _make_pref_data(items)
        # No same_labels arg → should default to True
        pairs = build_prompt_pairs(pref_data, max_pairs=10)
        assert len(pairs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
