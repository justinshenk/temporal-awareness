"""Tests for lookahead planning datasets.

These tests verify dataset integrity BEFORE we spend GPU hours extracting
activations. Every dataset must pass these checks.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.lookahead.utils.types import TaskType, PlanningExample
from src.lookahead.datasets.rhyme import (
    generate_rhyme_dataset,
    generate_minimal_rhyme_pairs,
    save_dataset,
    load_dataset,
    RHYME_SETS,
)
from src.lookahead.datasets.acrostic import (
    generate_acrostic_dataset,
    generate_acrostic_minimal_pairs,
    ACROSTIC_TARGETS,
)
from src.lookahead.datasets.code_return import (
    generate_code_return_dataset,
    TYPED_FUNCTIONS,
    UNTYPED_FUNCTIONS,
)


# ═══════════════════════════════════════════════════════════════════════
# RHYME DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestRhymeDataset:
    def test_generation_produces_examples(self):
        examples = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=True)
        assert len(examples) > 0, "Should generate at least 1 example"
    
    def test_all_examples_have_required_fields(self):
        examples = generate_rhyme_dataset(n_per_rhyme_set=2)
        for ex in examples:
            assert ex.task_type == TaskType.RHYME
            assert isinstance(ex.prompt, str) and len(ex.prompt) > 0
            assert isinstance(ex.example_id, str) and len(ex.example_id) > 0
            assert isinstance(ex.metadata, dict)
    
    def test_non_control_examples_have_targets(self):
        examples = generate_rhyme_dataset(n_per_rhyme_set=2)
        for ex in examples:
            if not ex.metadata.get("is_control"):
                assert ex.target_value, f"Non-control example {ex.example_id} missing target"
                assert ex.metadata.get("anchor_word"), "Missing anchor word"
                assert ex.metadata.get("all_valid_rhymes"), "Missing rhyme list"
    
    def test_control_examples_flagged(self):
        examples = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=True)
        controls = [e for e in examples if e.metadata.get("is_control")]
        non_controls = [e for e in examples if not e.metadata.get("is_control")]
        assert len(controls) > 0, "Should have control examples"
        assert len(non_controls) > len(controls), "Should have more non-controls than controls"
    
    def test_unique_ids(self):
        examples = generate_rhyme_dataset(n_per_rhyme_set=3)
        ids = [e.example_id for e in examples]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {len(ids)} total, {len(set(ids))} unique"
    
    def test_prompts_contain_anchor(self):
        """Non-control prompts should contain the anchor word."""
        examples = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=False)
        for ex in examples:
            anchor = ex.metadata.get("anchor_word", "")
            assert anchor in ex.prompt, f"Prompt doesn't contain anchor '{anchor}': {ex.prompt[:80]}"
    
    def test_target_is_valid_rhyme(self):
        """Target value should be in the valid rhymes list."""
        examples = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=False)
        for ex in examples:
            if ex.target_value:
                valid = ex.metadata.get("all_valid_rhymes", [])
                assert ex.target_value in valid, (
                    f"Target '{ex.target_value}' not in valid rhymes {valid}"
                )
    
    def test_rhyme_sets_have_valid_rhymes(self):
        """Sanity: all rhyme sets should have at least 3 valid rhymes."""
        for anchor, common, uncommon in RHYME_SETS:
            total = len(common) + len(uncommon)
            assert total >= 3, f"Rhyme set '{anchor}' has only {total} rhymes"
    
    def test_save_and_load_roundtrip(self):
        """Dataset should survive save/load without corruption."""
        examples = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_rhyme.json"
            save_dataset(examples, path)
            loaded = load_dataset(path)
        
        assert len(loaded) == len(examples)
        for orig, loaded_ex in zip(examples, loaded):
            assert orig.task_type == loaded_ex.task_type
            assert orig.prompt == loaded_ex.prompt
            assert orig.target_value == loaded_ex.target_value
            assert orig.example_id == loaded_ex.example_id
    
    def test_minimal_pairs_differ_in_anchor(self):
        """Minimal pairs should have different anchor words."""
        pairs = generate_minimal_rhyme_pairs(n_pairs=5)
        assert len(pairs) > 0
        
        for ex_a, ex_b in pairs:
            anchor_a = ex_a.metadata.get("anchor_word")
            anchor_b = ex_b.metadata.get("anchor_word")
            assert anchor_a != anchor_b, "Minimal pair should have different anchors"
    
    def test_deterministic_generation(self):
        """Same seed should produce same dataset."""
        ex1 = generate_rhyme_dataset(n_per_rhyme_set=2, seed=42)
        ex2 = generate_rhyme_dataset(n_per_rhyme_set=2, seed=42)
        assert len(ex1) == len(ex2)
        for a, b in zip(ex1, ex2):
            assert a.prompt == b.prompt
            assert a.target_value == b.target_value


# ═══════════════════════════════════════════════════════════════════════
# ACROSTIC DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestAcrosticDataset:
    def test_generation_produces_examples(self):
        examples = generate_acrostic_dataset(word_lengths=[3, 4])
        assert len(examples) > 0
    
    def test_progressive_reveal_series(self):
        """Progressive reveal should generate one example per reveal stage."""
        examples = generate_acrostic_dataset(
            word_lengths=[4], include_progressive=True
        )
        
        # Group by word
        by_word = {}
        for ex in examples:
            word = ex.metadata["full_word"]
            by_word.setdefault(word, []).append(ex)
        
        for word, exs in by_word.items():
            n_revealed_values = sorted(ex.metadata["n_revealed"] for ex in exs)
            expected = list(range(len(word)))
            assert n_revealed_values == expected, (
                f"Word '{word}': expected reveals {expected}, got {n_revealed_values}"
            )
    
    def test_next_letter_is_correct(self):
        """The target value should be the correct next letter."""
        examples = generate_acrostic_dataset(word_lengths=[3, 4, 5])
        for ex in examples:
            word = ex.metadata["full_word"]
            n_revealed = ex.metadata["n_revealed"]
            expected_letter = word[n_revealed]
            assert ex.target_value == expected_letter, (
                f"Word '{word}', {n_revealed} revealed: "
                f"expected '{expected_letter}', got '{ex.target_value}'"
            )
    
    def test_prompts_contain_instruction(self):
        examples = generate_acrostic_dataset(word_lengths=[4])
        for ex in examples:
            word = ex.metadata["full_word"]
            assert word in ex.prompt, f"Prompt should mention target word '{word}'"
            assert "acrostic" in ex.prompt.lower(), "Prompt should mention 'acrostic'"
    
    def test_unique_ids(self):
        examples = generate_acrostic_dataset(word_lengths=[3, 4, 5])
        ids = [e.example_id for e in examples]
        assert len(ids) == len(set(ids)), "Duplicate IDs"


# ═══════════════════════════════════════════════════════════════════════
# CODE RETURN TYPE DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCodeReturnDataset:
    def test_generation_produces_examples(self):
        examples = generate_code_return_dataset()
        assert len(examples) > 0
    
    def test_typed_functions_have_annotation(self):
        """Typed function prompts should contain '->'."""
        examples = generate_code_return_dataset(include_untyped=False, include_contrastive=False)
        for ex in examples:
            if ex.metadata.get("has_type_annotation"):
                assert "->" in ex.prompt, f"Typed function missing '->': {ex.prompt[:60]}"
    
    def test_return_types_are_valid(self):
        """All return types should be standard Python types."""
        valid_types = {"int", "str", "bool", "list", "float", "None", "dict", "tuple", "set"}
        examples = generate_code_return_dataset()
        for ex in examples:
            assert ex.target_value in valid_types, (
                f"Unknown return type: {ex.target_value}"
            )
    
    def test_contrastive_pairs_differ_in_type(self):
        """Contrastive pairs should have different return types."""
        examples = generate_code_return_dataset(include_contrastive=True)
        contrastive = [e for e in examples if e.metadata.get("is_contrastive")]
        
        # Group by pair ID
        by_pair = {}
        for ex in contrastive:
            pair_id = ex.metadata["contrastive_pair"]
            by_pair.setdefault(pair_id, []).append(ex)
        
        for pair_id, pair_exs in by_pair.items():
            types = set(ex.target_value for ex in pair_exs)
            assert len(types) >= 2, f"Contrastive pair {pair_id} has same types: {types}"
    
    def test_prompts_are_valid_python_starts(self):
        """Prompts should be syntactically valid Python function starts."""
        examples = generate_code_return_dataset(include_contrastive=False)
        for ex in examples:
            assert ex.prompt.strip().startswith("def "), (
                f"Prompt doesn't start with 'def': {ex.prompt[:60]}"
            )
            assert ":" in ex.prompt, f"Missing colon: {ex.prompt[:60]}"
    
    def test_unique_ids(self):
        examples = generate_code_return_dataset()
        ids = [e.example_id for e in examples]
        assert len(ids) == len(set(ids)), "Duplicate IDs"


# ═══════════════════════════════════════════════════════════════════════
# CROSS-TASK TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCrossTask:
    def test_all_tasks_use_correct_type(self):
        rhyme = generate_rhyme_dataset(n_per_rhyme_set=1, include_controls=False)
        acrostic = generate_acrostic_dataset(word_lengths=[3])
        code = generate_code_return_dataset(include_untyped=False, include_contrastive=False)
        
        for ex in rhyme:
            assert ex.task_type == TaskType.RHYME
        for ex in acrostic:
            assert ex.task_type == TaskType.ACROSTIC
        for ex in code:
            assert ex.task_type == TaskType.CODE_RETURN
    
    def test_no_id_collision_across_tasks(self):
        """IDs should be unique even across task types."""
        rhyme = generate_rhyme_dataset(n_per_rhyme_set=2, include_controls=False)
        acrostic = generate_acrostic_dataset(word_lengths=[3, 4])
        code = generate_code_return_dataset()
        
        all_ids = (
            [e.example_id for e in rhyme]
            + [e.example_id for e in acrostic]
            + [e.example_id for e in code]
        )
        assert len(all_ids) == len(set(all_ids)), "Cross-task ID collision"
    
    def test_no_empty_prompts(self):
        """No task should produce empty prompts."""
        all_examples = (
            generate_rhyme_dataset(n_per_rhyme_set=1)
            + generate_acrostic_dataset(word_lengths=[3])
            + generate_code_return_dataset()
        )
        for ex in all_examples:
            assert len(ex.prompt.strip()) > 5, f"Prompt too short: '{ex.prompt}'"
