"""Behavioral validation: verify the model actually performs the task.

THIS MUST RUN BEFORE ANY PROBING.

If the model can't produce rhymes, probing for "rhyme commitment" is 
meaningless — you'd be probing noise from a model doing something 
entirely unrelated to your task.

For each task type, we:
1. Generate completions from the model
2. Score whether the completion matches the task constraint
3. Filter examples to ONLY those where the model succeeds
4. Report behavioral accuracy — this is a key table in the paper

A hostile reviewer will ask: "How do you know the model is planning
and not just doing statistical continuation?" Behavioral validation
is step 1 of answering that.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from ..utils.types import PlanningExample, TaskType

logger = logging.getLogger(__name__)


@dataclass
class BehavioralResult:
    """Result of behavioral validation for a single example."""
    example_id: str
    task_type: TaskType
    prompt: str
    completion: str
    target_value: str
    
    # Task-specific scores
    task_success: bool  # did the model satisfy the constraint?
    target_match: bool  # did the model produce the SPECIFIC target we're probing for?
    any_valid_match: bool  # did the model produce ANY valid target?
    
    # Details
    detected_value: str = ""  # what the model actually produced
    score_details: dict = field(default_factory=dict)


def run_behavioral_validation(
    model,
    examples: list[PlanningExample],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    n_samples: int = 1,
) -> list[BehavioralResult]:
    """Generate completions and validate task performance.
    
    Args:
        model: TransformerLens HookedTransformer
        examples: Planning examples to validate
        max_new_tokens: Max tokens to generate
        temperature: 0 = greedy (deterministic)
        n_samples: Number of samples per example (>1 for stochastic)
        
    Returns:
        List of BehavioralResult
    """
    results = []
    
    for example in examples:
        if example.metadata.get("is_control", False):
            continue
        
        tokens = model.to_tokens(example.prompt, prepend_bos=True)
        
        with torch.no_grad():
            output_ids = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-10),
                stop_at_eos=True,
                verbose=False,
            )
        
        completion = model.to_string(output_ids[0, tokens.shape[1]:])
        
        # Score based on task type
        if example.task_type == TaskType.RHYME:
            result = _score_rhyme(example, completion)
        elif example.task_type == TaskType.ACROSTIC:
            result = _score_acrostic(example, completion)
        elif example.task_type == TaskType.CODE_RETURN:
            result = _score_code_return(example, completion)
        else:
            result = BehavioralResult(
                example_id=example.example_id,
                task_type=example.task_type,
                prompt=example.prompt,
                completion=completion,
                target_value=example.target_value,
                task_success=False,
                target_match=False,
                any_valid_match=False,
            )
        
        results.append(result)
    
    return results


def _score_rhyme(example: PlanningExample, completion: str) -> BehavioralResult:
    """Score whether completion rhymes with the anchor word."""
    anchor = example.metadata.get("anchor_word", "")
    valid_rhymes = example.metadata.get("all_valid_rhymes", [])
    target = example.target_value
    
    # Extract last word of completion (likely the rhyme)
    words = re.findall(r"[a-zA-Z]+", completion.lower())
    
    # Check all words in completion against valid rhymes
    detected = ""
    any_valid = False
    target_found = False
    
    for word in words:
        if word in [r.lower() for r in valid_rhymes]:
            any_valid = True
            detected = word
            if word == target.lower():
                target_found = True
            break  # take first rhyme match
    
    # Also check with simple phonetic heuristic: same ending
    if not any_valid and anchor:
        suffix_len = min(3, len(anchor))
        anchor_suffix = anchor[-suffix_len:].lower()
        for word in words:
            if len(word) >= suffix_len and word[-suffix_len:] == anchor_suffix and word != anchor.lower():
                any_valid = True
                detected = word
                break
    
    return BehavioralResult(
        example_id=example.example_id,
        task_type=TaskType.RHYME,
        prompt=example.prompt,
        completion=completion,
        target_value=target,
        task_success=any_valid,
        target_match=target_found,
        any_valid_match=any_valid,
        detected_value=detected,
        score_details={
            "anchor": anchor,
            "valid_rhymes_checked": len(valid_rhymes),
            "completion_words": words[:10],
        },
    )


def _score_acrostic(example: PlanningExample, completion: str) -> BehavioralResult:
    """Score whether completion starts with the correct letter."""
    target_letter = example.target_value.upper()
    full_word = example.metadata.get("full_word", "")
    
    # The completion should be the next line starting with target_letter
    lines = completion.strip().split("\n")
    first_line = lines[0].strip() if lines else ""
    
    # Get first alphabetic character
    first_char = ""
    for c in first_line:
        if c.isalpha():
            first_char = c.upper()
            break
    
    target_found = first_char == target_letter
    
    return BehavioralResult(
        example_id=example.example_id,
        task_type=TaskType.ACROSTIC,
        prompt=example.prompt,
        completion=completion,
        target_value=target_letter,
        task_success=target_found,
        target_match=target_found,
        any_valid_match=target_found,
        detected_value=first_char,
        score_details={
            "full_word": full_word,
            "first_line": first_line[:80],
            "n_revealed": example.metadata.get("n_revealed", 0),
        },
    )


def _score_code_return(example: PlanningExample, completion: str) -> BehavioralResult:
    """Score whether generated code body is consistent with return type."""
    target_type = example.target_value
    
    # Heuristic: check if the completion contains a return statement
    # consistent with the declared type
    has_return = "return " in completion
    
    type_consistent = False
    detected_type = "unknown"
    
    if has_return:
        # Extract what follows 'return'
        return_match = re.search(r"return\s+(.+?)(?:\n|$)", completion)
        if return_match:
            return_expr = return_match.group(1).strip()
            detected_type = _infer_return_type(return_expr)
            type_consistent = detected_type == target_type
    elif target_type == "None":
        # Functions returning None often don't have explicit return
        type_consistent = True
        detected_type = "None"
    
    return BehavioralResult(
        example_id=example.example_id,
        task_type=TaskType.CODE_RETURN,
        prompt=example.prompt,
        completion=completion,
        target_value=target_type,
        task_success=type_consistent,
        target_match=type_consistent,
        any_valid_match=type_consistent,
        detected_value=detected_type,
        score_details={
            "has_return": has_return,
            "completion_preview": completion[:100],
        },
    )


def _infer_return_type(expr: str) -> str:
    """Heuristic: infer type from a return expression."""
    expr = expr.strip().rstrip(";")
    
    if expr.startswith('"') or expr.startswith("'") or expr.startswith("f'") or expr.startswith('f"'):
        return "str"
    if expr in ("True", "False") or expr.startswith("not ") or " == " in expr or " != " in expr or " in " in expr:
        return "bool"
    if expr.startswith("[") or ".split(" in expr:
        return "list"
    if expr.startswith("{"):
        return "dict"
    if expr.startswith("("):
        return "tuple"
    if "." in expr and not expr.startswith("self."):
        try:
            float(expr)
            return "float"
        except ValueError:
            pass
    if expr == "None":
        return "None"
    try:
        int(expr)
        return "int"
    except ValueError:
        pass
    # len(), sum(), count() etc. return int
    if any(expr.startswith(f"{fn}(") for fn in ["len", "sum", "count", "int", "abs", "max", "min"]):
        return "int"
    if any(expr.startswith(f"{fn}(") for fn in ["float", "round"]):
        return "float"
    if any(expr.startswith(f"{fn}(") for fn in ["str", "format"]):
        return "str"
    if any(expr.startswith(f"{fn}(") for fn in ["list", "sorted"]):
        return "list"
    if any(expr.startswith(f"{fn}(") for fn in ["bool"]):
        return "bool"
    
    # Arithmetic expressions → likely int or float
    if re.match(r"^[\w\s\+\-\*/\%\(\)]+$", expr):
        if "/" in expr and "//" not in expr:
            return "float"
        return "int"
    
    return "unknown"


def filter_to_successful(
    examples: list[PlanningExample],
    behavioral_results: list[BehavioralResult],
    require_exact_target: bool = False,
) -> tuple[list[PlanningExample], dict]:
    """Filter examples to only those where the model succeeds at the task.
    
    Args:
        examples: Full example list
        behavioral_results: Results from run_behavioral_validation
        require_exact_target: If True, require exact target match, not just any valid
        
    Returns:
        (filtered_examples, stats_dict)
    """
    result_by_id = {r.example_id: r for r in behavioral_results}
    
    filtered = []
    stats = {
        "total": 0,
        "task_success": 0,
        "target_match": 0,
        "any_valid": 0,
        "controls_kept": 0,
    }
    
    for ex in examples:
        if ex.metadata.get("is_control", False):
            filtered.append(ex)
            stats["controls_kept"] += 1
            continue
        
        stats["total"] += 1
        br = result_by_id.get(ex.example_id)
        
        if br is None:
            continue
        
        if br.task_success:
            stats["task_success"] += 1
        if br.target_match:
            stats["target_match"] += 1
        if br.any_valid_match:
            stats["any_valid"] += 1
        
        # Apply filter
        if require_exact_target:
            if br.target_match:
                filtered.append(ex)
        else:
            if br.any_valid_match:
                filtered.append(ex)
    
    stats["kept"] = len(filtered) - stats["controls_kept"]
    stats["behavioral_accuracy"] = stats["task_success"] / max(1, stats["total"])
    stats["exact_match_rate"] = stats["target_match"] / max(1, stats["total"])
    
    return filtered, stats


def compute_behavioral_summary(results: list[BehavioralResult]) -> dict:
    """Compute summary statistics for behavioral validation.
    
    This becomes Table 1 in the paper.
    """
    by_task = {}
    for r in results:
        task = r.task_type.value
        if task not in by_task:
            by_task[task] = {"total": 0, "success": 0, "exact": 0, "any_valid": 0}
        by_task[task]["total"] += 1
        by_task[task]["success"] += int(r.task_success)
        by_task[task]["exact"] += int(r.target_match)
        by_task[task]["any_valid"] += int(r.any_valid_match)
    
    summary = {}
    for task, counts in by_task.items():
        n = counts["total"]
        summary[task] = {
            "n": n,
            "task_accuracy": counts["success"] / max(1, n),
            "exact_match": counts["exact"] / max(1, n),
            "any_valid": counts["any_valid"] / max(1, n),
        }
    
    return summary
