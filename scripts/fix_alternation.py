"""
Script: Fix Alternation for Explicit Temporal Scope Dataset

Description:
This script processes `data/raw/temporal_scope_....json` to enforce an alternating 
pattern for option labels (A) and (B). This is done to mitigate position bias in model evaluations.

Logic:
- Odd-numbered items (Index 0, 2, ...): Immediate option gets (A), Long-term gets (B).
- Even-numbered items (Index 1, 3, ...): Immediate option gets (B), Long-term gets (A).

Result:
The dataset will have a 50/50 split of immediate rewards being assigned to A vs B.
"""
import json
import re
import os

def fix_explicit_alternation():
    # Path to the specific json file
    file_path = os.path.join(os.path.dirname(__file__), '../data/raw/temporal_scope_implicit_backup_300.json')
    file_path = os.path.abspath(file_path)

    print(f"Processing file: {file_path}")

    # Load existing data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    pairs = data.get('pairs', [])
    
    # Pattern to match existing labels like " (A) ", "(A) ", " (B) ", etc.
    # It looks for optional whitespace, parentheses, A or B, parentheses, and trailing spaces.
    label_pattern = re.compile(r'^\s*\([AB]\)\s+')

    for i, item in enumerate(pairs):
        # 1. Clean the text (remove existing labels)
        # We assume the structure is " (Label) Text Content"
        immediate_raw = item['immediate']
        long_term_raw = item['long_term']

        # Remove the label prefix if it exists to get the clean text
        imm_text = label_pattern.sub('', immediate_raw)
        lt_text = label_pattern.sub('', long_term_raw)

        # 2. Assign new labels based on parity
        # i starts at 0 (1st pair). 
        # Requirement: 1st pair (Odd number) -> A is Immediate.
        #              2nd pair (Even number) -> B is Immediate.
        
        if i % 2 == 0:
            # Even index (0, 2, 4...) corresponds to Odd numbered pairs (1st, 3rd...)
            # Immediate gets (A)
            item['immediate'] = f" (A) {imm_text}"
            item['long_term'] = f" (B) {lt_text}"
        else:
            # Odd index (1, 3, 5...) corresponds to Even numbered pairs (2nd, 4th...)
            # Immediate gets (B)
            item['immediate'] = f" (B) {imm_text}"
            item['long_term'] = f" (A) {lt_text}"

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        # indent=2 matches the visual style of your snippet
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully processed {len(pairs)} pairs with alternating A/B assignment.")

if __name__ == "__main__":
    fix_explicit_alternation()