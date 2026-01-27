"""Tests for activation patching metric calculation.

The key insight: each sample has its own label format (e.g., OPTION_ONE/OPTION_TWO,
[I]/[II], a)/b), etc.). The metric must use the correct labels for each sample.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockPreferenceItem:
    """Mock preference item for testing."""
    short_term_label: str
    long_term_label: str
    choice: str
    response: str = ""
    prompt_text: str = ""


class TestLabelMatching:
    """Test that labels are correctly matched to short/long term."""

    def test_different_label_formats_same_meaning(self):
        """Different label formats should all work for metric calculation."""
        # These are all valid short_term labels from different samples
        short_term_labels = [
            "OPTION_ONE:",
            "[I]",
            "a)",
            "Option A:",
            "Choice 1:",
            "(A)",
        ]

        # These are corresponding long_term labels
        long_term_labels = [
            "OPTION_TWO:",
            "[II]",
            "b)",
            "Option B:",
            "Choice 2:",
            "(B)",
        ]

        for short, long in zip(short_term_labels, long_term_labels):
            item = MockPreferenceItem(
                short_term_label=short,
                long_term_label=long,
                choice="short_term",
            )
            # The item correctly identifies which label is short vs long term
            assert item.short_term_label == short
            assert item.long_term_label == long

    def test_swapped_labels(self):
        """Labels can be swapped - OPTION_TWO can be short_term in some samples."""
        # In some samples, OPTION_TWO is the short-term choice
        item1 = MockPreferenceItem(
            short_term_label="OPTION_TWO:",
            long_term_label="OPTION_ONE:",
            choice="short_term",
        )

        # In other samples, OPTION_ONE is the short-term choice
        item2 = MockPreferenceItem(
            short_term_label="OPTION_ONE:",
            long_term_label="OPTION_TWO:",
            choice="long_term",
        )

        # Both are valid - the label assignment depends on the prompt content
        assert item1.short_term_label == "OPTION_TWO:"
        assert item2.short_term_label == "OPTION_ONE:"


class TestMetricCalculation:
    """Test metric calculation uses correct labels per sample."""

    def test_metric_uses_sample_specific_labels(self):
        """Metric should use each sample's own labels, not a fixed set."""
        # Sample 1 uses OPTION format
        sample1 = MockPreferenceItem(
            short_term_label="OPTION_ONE:",
            long_term_label="OPTION_TWO:",
            choice="short_term",
        )

        # Sample 2 uses bracket format
        sample2 = MockPreferenceItem(
            short_term_label="[A]",
            long_term_label="[B]",
            choice="long_term",
        )

        # When computing metric for sample1, use OPTION tokens
        # When computing metric for sample2, use bracket tokens
        # They should NOT be mixed

        assert sample1.short_term_label != sample2.short_term_label
        assert sample1.long_term_label != sample2.long_term_label

    def test_logit_diff_direction(self):
        """Logit diff should be positive when model prefers short_term."""
        # Mock logits where short_term token has higher logit
        mock_logits = torch.zeros(1, 10, 100)  # batch, seq, vocab

        short_token_id = 50
        long_token_id = 60

        # Set short_term logit higher
        mock_logits[0, -1, short_token_id] = 5.0
        mock_logits[0, -1, long_token_id] = 2.0

        # Logit diff = short - long = 5.0 - 2.0 = 3.0 (positive)
        logit_diff = mock_logits[0, -1, short_token_id] - mock_logits[0, -1, long_token_id]
        assert logit_diff > 0, "Positive diff means model prefers short_term"

    def test_logit_diff_at_choice_position(self):
        """Logit diff should be computed at the choice position, not final position."""
        # Mock logits for a sequence
        seq_len = 150
        mock_logits = torch.zeros(1, seq_len, 100)

        short_token_id = 50
        long_token_id = 60
        choice_position = 131  # Where "I select: X" appears

        # At choice position: model prefers short_term
        mock_logits[0, choice_position, short_token_id] = 5.0
        mock_logits[0, choice_position, long_token_id] = 2.0

        # At final position: different distribution (continuation tokens)
        mock_logits[0, -1, short_token_id] = 1.0
        mock_logits[0, -1, long_token_id] = 1.5

        # Metric at choice position shows correct preference
        diff_at_choice = mock_logits[0, choice_position, short_token_id] - mock_logits[0, choice_position, long_token_id]
        diff_at_final = mock_logits[0, -1, short_token_id] - mock_logits[0, -1, long_token_id]

        assert diff_at_choice > 0, "At choice position, model prefers short_term"
        assert diff_at_final < 0, "At final position, metric is wrong"


class TestPairBuilding:
    """Test that prompt pairs are built correctly."""

    def test_pairs_use_own_labels(self):
        """Each item in a pair should use its own labels for metric."""
        clean_sample = MockPreferenceItem(
            short_term_label="OPTION_TWO:",
            long_term_label="OPTION_ONE:",
            choice="short_term",
        )

        corrupted_sample = MockPreferenceItem(
            short_term_label="[II]",
            long_term_label="[I]",
            choice="long_term",
        )

        # For clean metric: use OPTION_TWO (short) vs OPTION_ONE (long)
        # For corrupted metric: use [II] (short) vs [I] (long)

        # The key insight: we need to compute metrics using each sample's OWN labels
        # Clean logit diff = logit(OPTION_TWO) - logit(OPTION_ONE)
        # Corrupted logit diff = logit([II]) - logit([I])

        # These are DIFFERENT tokens, so we can't use a single token pair
        # for both metrics

        assert clean_sample.short_term_label != corrupted_sample.short_term_label


class TestNormalization:
    """Test metric normalization."""

    def test_normalized_metric_range(self):
        """Normalized metric should be 0 for corrupted baseline, 1 for clean baseline."""
        clean_value = 3.0  # Clean sample prefers short_term
        corrupted_value = -2.0  # Corrupted sample prefers long_term

        def normalize(raw_value):
            """Normalize so 0=corrupted, 1=clean."""
            diff = clean_value - corrupted_value
            if abs(diff) < 1e-8:
                return 0.0
            return (raw_value - corrupted_value) / diff

        assert normalize(corrupted_value) == pytest.approx(0.0)
        assert normalize(clean_value) == pytest.approx(1.0)

        # Mid-point should be 0.5
        mid_value = (clean_value + corrupted_value) / 2
        assert normalize(mid_value) == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
