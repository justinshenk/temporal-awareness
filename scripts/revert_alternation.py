"""
Script: Revert Alternation (Normalize AB)

Description:
This script processes JSON files in `data/raw/temporal_scope_AB_randomized`.
It creates NEW files where the randomization/alternation is removed.

Logic:
- Reads every pair.
- Strips existing (A)/(B) labels from both 'immediate' and 'long_term' fields.
- Enforces a fixed pattern:
    - Immediate option -> Always (A)
    - Long-term option -> Always (B)

Output:
Creates new files in `data/raw/temporal_scope_AB` (folder will be created if missing).

Usage:
  Process all files:    python scripts/revert_alternation.py
  Process one file:     python scripts/revert_alternation.py temporal_scope_implicit_expanded_1000.json
"""
import json
import re
import os
import sys


def process_file(input_path, output_path, label_pattern):
    """Process a single JSON file: normalize A/B labels."""
    print(f"Processing: {os.path.basename(input_path)}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data.get('pairs', [])

    for item in pairs:
        immediate_raw = item.get('immediate', "")
        long_term_raw = item.get('long_term', "")

        imm_text = label_pattern.sub('', immediate_raw)
        lt_text = label_pattern.sub('', long_term_raw)

        item['immediate'] = f"(A) {imm_text}"
        item['long_term'] = f"(B) {lt_text}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  -> Saved to: {output_path} ({len(pairs)} pairs)")


def revert_to_invariant_AB(target_file=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'data', 'raw', 'temporal_scope_AB_randomized')
    output_dir = os.path.join(base_dir, 'data', 'raw', 'temporal_scope')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    label_pattern = re.compile(r'^\s*\([AB]\)\s+')

    if target_file:
        input_path = os.path.join(input_dir, target_file)
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return
        new_filename = target_file
        output_path = os.path.join(output_dir, new_filename)
        process_file(input_path, output_path, label_pattern)
        print("\nDone. Processed 1 file.")
    else:
        files_processed = 0
        for filename in os.listdir(input_dir):
            if not filename.endswith('.json'):
                continue
            input_path = os.path.join(input_dir, filename)
            new_filename = filename
            output_path = os.path.join(output_dir, new_filename)
            try:
                process_file(input_path, output_path, label_pattern)
                files_processed += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        print(f"\nDone. Processed {files_processed} files.")

    print(f"Output files are located in: {output_dir}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    revert_to_invariant_AB(target)
