"""Code return type dataset for planning detection.

Tests whether models commit to a function's return type before generating
the function body. The hypothesis: when a model generates a function with
a type annotation (e.g., `def foo() -> int:`), the return type commitment
should appear in activations *at or before* the `->` token, and should
persist through the function body.

This is interesting because:
1. Return type is syntactically declared early but semantically constrains
   everything that follows (the body must return that type)
2. We can test both annotated (explicit type) and unannotated (implicit type)
   functions
3. The "commitment" here has a clear behavioral consequence: the generated
   body should be type-consistent
"""

from __future__ import annotations

import hashlib

from ..utils.types import PlanningExample, TaskType


# Function signatures with known return types
# Each entry: (signature, return_type, body_hint)
# body_hint helps us know what a natural completion looks like
TYPED_FUNCTIONS = [
    # int-returning functions
    ("def add(a: int, b: int) -> int:", "int",
     "return a + b"),
    ("def count_items(items: list) -> int:", "int",
     "return len(items)"),
    ("def fibonacci(n: int) -> int:", "int",
     "if n <= 1: return n"),
    ("def max_value(a: int, b: int) -> int:", "int",
     "return a if a > b else b"),
    ("def string_length(s: str) -> int:", "int",
     "return len(s)"),
    
    # str-returning functions
    ("def greet(name: str) -> str:", "str",
     'return f"Hello, {name}!"'),
    ("def reverse_string(s: str) -> str:", "str",
     "return s[::-1]"),
    ("def uppercase(text: str) -> str:", "str",
     "return text.upper()"),
    ("def join_words(words: list) -> str:", "str",
     'return " ".join(words)'),
    ("def format_date(year: int, month: int, day: int) -> str:", "str",
     'return f"{year}-{month:02d}-{day:02d}"'),
    
    # bool-returning functions
    ("def is_even(n: int) -> bool:", "bool",
     "return n % 2 == 0"),
    ("def is_empty(s: str) -> bool:", "bool",
     "return len(s) == 0"),
    ("def contains(items: list, target) -> bool:", "bool",
     "return target in items"),
    ("def is_palindrome(s: str) -> bool:", "bool",
     "return s == s[::-1]"),
    
    # list-returning functions
    ("def get_evens(numbers: list) -> list:", "list",
     "return [n for n in numbers if n % 2 == 0]"),
    ("def flatten(nested: list) -> list:", "list",
     "return [x for sublist in nested for x in sublist]"),
    ("def unique(items: list) -> list:", "list",
     "return list(set(items))"),
    
    # float-returning functions
    ("def average(numbers: list) -> float:", "float",
     "return sum(numbers) / len(numbers)"),
    ("def circle_area(radius: float) -> float:", "float",
     "return 3.14159 * radius ** 2"),
    
    # None-returning functions
    ("def print_message(msg: str) -> None:", "None",
     "print(msg)"),
    ("def log_error(error: str) -> None:", "None",
     'print(f"ERROR: {error}")'),
]

# Untyped versions — same function names, no type annotations
# The model must infer return type from the function name/body
# EXPANDED: 24 functions (4 per type) for reliable probing
UNTYPED_FUNCTIONS = [
    # int returns (4)
    ("def add(a, b):", "int", "return a + b"),
    ("def count_words(text):", "int", "return len(text.split())"),
    ("def factorial(n):", "int", "if n <= 1: return 1"),
    ("def find_max(numbers):", "int", "return max(numbers)"),
    # str returns (4)
    ("def greet(name):", "str", 'return "Hello, " + name'),
    ("def to_upper(text):", "str", "return text.upper()"),
    ("def remove_spaces(s):", "str", 'return s.replace(" ", "")'),
    ("def first_name(full_name):", "str", "return full_name.split()[0]"),
    # bool returns (4)
    ("def is_even(n):", "bool", "return n % 2 == 0"),
    ("def is_positive(x):", "bool", "return x > 0"),
    ("def has_duplicates(items):", "bool", "return len(items) != len(set(items))"),
    ("def starts_with_vowel(word):", "bool", "return word[0].lower() in 'aeiou'"),
    # list returns (4)
    ("def get_evens(numbers):", "list", "return [n for n in numbers if n % 2 == 0]"),
    ("def split_words(text):", "list", "return text.split()"),
    ("def remove_empty(items):", "list", "return [x for x in items if x]"),
    ("def get_keys(d):", "list", "return list(d.keys())"),
    # float returns (4)
    ("def average(numbers):", "float", "return sum(numbers) / len(numbers)"),
    ("def to_celsius(f):", "float", "return (f - 32) * 5 / 9"),
    ("def percentage(part, total):", "float", "return part / total * 100"),
    ("def distance(x1, y1, x2, y2):", "float", "return ((x2-x1)**2 + (y2-y1)**2)**0.5"),
    # None returns (4)
    ("def print_message(msg):", "None", "print(msg)"),
    ("def log_info(message):", "None", "print('[INFO]', message)"),
    ("def clear_list(items):", "None", "items.clear()"),
    ("def set_value(d, key, val):", "None", "d[key] = val"),
]


def generate_code_return_dataset(
    include_untyped: bool = True,
    include_contrastive: bool = True,
    seed: int = 42,
) -> list[PlanningExample]:
    """Generate code return type planning examples.
    
    For typed functions: the return type is explicit in the signature.
    The question is whether the model "commits" to the return type
    at the annotation token or earlier (from function name).
    
    For untyped functions: the return type is implicit.
    The question is whether we can detect return type commitment
    purely from the function name and parameter names.
    
    Args:
        include_untyped: Whether to include untyped function variants
        include_contrastive: Whether to include contrastive pairs
        seed: Random seed
        
    Returns:
        List of PlanningExample
    """
    examples = []
    
    # Typed functions
    for idx, (sig, ret_type, body) in enumerate(TYPED_FUNCTIONS):
        # Prompt: just the signature, model generates the body
        prompt = sig + "\n    "
        
        ex = PlanningExample(
            task_type=TaskType.CODE_RETURN,
            prompt=prompt,
            target_value=ret_type,
            target_token_positions=[],
            metadata={
                "signature": sig,
                "expected_body": body,
                "has_type_annotation": True,
                "return_type": ret_type,
                "is_control": False,
            },
            example_id=hashlib.md5(
                f"code_typed_{idx}".encode()
            ).hexdigest()[:12],
        )
        examples.append(ex)
    
    # Untyped functions
    if include_untyped:
        for idx, (sig, ret_type, body) in enumerate(UNTYPED_FUNCTIONS):
            prompt = sig + "\n    "
            
            ex = PlanningExample(
                task_type=TaskType.CODE_RETURN,
                prompt=prompt,
                target_value=ret_type,
                target_token_positions=[],
                metadata={
                    "signature": sig,
                    "expected_body": body,
                    "has_type_annotation": False,
                    "return_type": ret_type,
                    "is_control": False,
                },
                example_id=hashlib.md5(
                    f"code_untyped_{idx}".encode()
                ).hexdigest()[:12],
            )
            examples.append(ex)
    
    # Contrastive pairs: same function structure, different return types
    if include_contrastive:
        contrastive = _build_code_contrastive_pairs()
        examples.extend(contrastive)
    
    return examples


def _build_code_contrastive_pairs() -> list[PlanningExample]:
    """Build contrastive pairs where only the return type differs.
    
    E.g.: 
      def process(data) -> int: \n     # returns int
      def process(data) -> str: \n     # returns str
    
    Same function name, same params, different type annotation.
    """
    examples = []
    
    pairs = [
        # (function_name, params, type_a, type_b)
        ("process", "data", "int", "str"),
        ("transform", "value", "str", "list"),
        ("compute", "x, y", "int", "float"),
        ("validate", "input_data", "bool", "str"),
        ("convert", "raw", "int", "str"),
        ("extract", "source", "list", "str"),
        ("calculate", "values", "float", "int"),
    ]
    
    for idx, (name, params, type_a, type_b) in enumerate(pairs):
        sig_a = f"def {name}({params}) -> {type_a}:"
        sig_b = f"def {name}({params}) -> {type_b}:"
        
        for type_val, sig, variant in [(type_a, sig_a, "a"), (type_b, sig_b, "b")]:
            ex = PlanningExample(
                task_type=TaskType.CODE_RETURN,
                prompt=sig + "\n    ",
                target_value=type_val,
                target_token_positions=[],
                metadata={
                    "signature": sig,
                    "has_type_annotation": True,
                    "return_type": type_val,
                    "is_contrastive": True,
                    "contrastive_pair": f"{name}_{type_a}_vs_{type_b}",
                    "is_control": False,
                },
                example_id=hashlib.md5(
                    f"code_contrastive_{idx}_{variant}".encode()
                ).hexdigest()[:12],
            )
            examples.append(ex)
    
    return examples
