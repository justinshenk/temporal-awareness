"""
Generate formatting variations for temporal scope datasets.

This script creates variations of temporal scope datasets with different
option key formats (e.g., (A)/(B), (1)/(2), [A]/[B]) to test whether
model circuits respond to semantic content vs token format.

Usage:
------
    python scripts/data/generate_variation.py <input_file> --imm <label> --lt <label> [--randomize true/false]

Examples:
---------
    # Standard (A)/(B) format, no randomization
    python scripts/data/generate_variation.py data/raw/temporal_scope/temporal_scope_explicit_expanded_500.json --imm "(A)" --lt "(B)" --randomize false

    # Numeric format (1)/(2), no randomization
    python scripts/data/generate_variation.py \\
        data/raw/temporal_scope/temporal_scope_explicit_expanded_500.json \\
        --imm "(1)" --lt "(2)" --randomize false

    # Randomized (A)/(B) - labels flip on even indices
    python scripts/data/generate_variation.py \\
        data/raw/temporal_scope/temporal_scope_explicit_expanded_500.json \\
        --imm "(A)" --lt "(B)" --randomize true

Output:
-------
    Creates a new JSON file in data/raw/temporal_scope_variation/ with naming format:
    <original_name>+format_<imm_label>_randomized_<true/false>.json

    Example: temporal_scope_explicit_expanded_500+format_(1)_randomized_false.json

Notes:
------
    - The script automatically strips existing labels from the data before applying new ones
    - Randomization is useful for testing if circuits respond to position vs content
    - Output directory is created automatically if it doesn't exist
"""

import argparse
import json
import re
import sys
from pathlib import Path

def generate_variation(input_file: str, imm_label: str, lt_label: str, is_randomized: bool):
    """
    Generates a variation of the temporal scope dataset with custom labels.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output directory: data/raw/temporal_scope_variation
    # Assumes script is located in scripts/data/ or similar depth relative to project root
    # Using resolve() to handle execution from any directory
    script_path = Path(__file__).resolve()
    # Go up 3 levels: scripts/data/script.py -> scripts/data -> scripts -> root
    project_root = script_path.parents[2] 
    
    output_dir = project_root / "data" / "raw" / "temporal_scope_variation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    
    # Regex to strip common existing labels like (A), [A], (1), [1] from start of string
    # Matches: start of line, optional whitespace, open bracket/paren, alphanumeric, close bracket/paren, whitespace
    label_pattern = re.compile(r'^\s*[\(\[][A-Za-z0-9]+[\)\]]\s+')

    for i, item in enumerate(pairs):
        # 1. Clean the text (remove existing labels)
        raw_imm = item.get("immediate", "")
        raw_lt = item.get("long_term", "")
        
        clean_imm = label_pattern.sub("", raw_imm)
        clean_lt = label_pattern.sub("", raw_lt)

        # 2. Assign new labels
        # Standard assignment
        apply_imm_label = imm_label
        apply_lt_label = lt_label
        
        if is_randomized:
            # "randomized meaning every even index has: (B) for immediate (A) for long term"
            # Even index (0, 2...)
            if i % 2 == 0:
                apply_imm_label = lt_label
                apply_lt_label = imm_label
            # Odd index (1, 3...): Stay standard (A for immediate, B for long term)
            else:
                apply_imm_label = imm_label
                apply_lt_label = lt_label

        # Apply formatting: "{Label} {Text}"
        item["immediate"] = f"{apply_imm_label} {clean_imm}"
        item["long_term"] = f"{apply_lt_label} {clean_lt}"

    # 3. Construct filename
    # Format: name+format_<Starting><Char/Chars><Closing>_randomized_<true/false>.json
    
    # Get original filename without extension or previous suffixes
    original_stem = input_path.stem.split('+')[0]
    
    # Extract format part from immediate label (e.g., "[A]" -> "[A]")
    label_format_str = imm_label.strip()
    
    random_str = "true" if is_randomized else "false"
    new_filename = f"{original_stem}+format_{label_format_str}_randomized_{random_str}.json"
    
    output_path = output_dir / new_filename

    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated variation.")
    print(f"Input: {input_path.name}")
    print(f"Output: {output_path}")
    print(f"Settings: Immediate='{imm_label}', LongTerm='{lt_label}', Randomized={is_randomized}")

def main():
    parser = argparse.ArgumentParser(description="Generate formatting variations for temporal dataset.")
    
    parser.add_argument("input_file", 
                        help="Path to the input JSON file")
    
    parser.add_argument("--imm", required=True, 
                        help="Label for immediate option (e.g. '[A]' or '(1)')")
    
    parser.add_argument("--lt", required=True, 
                        help="Label for long term option (e.g. '[B]' or '(2)')")
    
    parser.add_argument("--randomize", type=str, choices=['true', 'false'], default='true',
                        help="Whether to flip labels on even indices (default: true)")

    args = parser.parse_args()
    
    # Convert string argument to boolean
    is_randomized = args.randomize.lower() == 'true'

    generate_variation(args.input_file, args.imm, args.lt, is_randomized)

if __name__ == "__main__":
    main()