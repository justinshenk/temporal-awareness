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
Creates new files in `data/raw/temporal_scope_AB_invariant` (folder will be created if missing).
"""
import json
import re
import os

def revert_to_invariant_AB():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'data', 'raw', 'temporal_scope_AB_randomized')
    output_dir = os.path.join(base_dir, 'data', 'raw', 'temporal_scope_AB_invariant')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Regex to strip existing labels like "(A) ", " (B) ", etc.
    label_pattern = re.compile(r'^\s*\([AB]\)\s+')

    # 2. Iterate over all JSON files in the input folder
    files_processed = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_dir, filename)
        
        # Construct new filename (append _invariant to avoid confusion)
        new_filename = filename.replace('.json', '_invariant.json')
        output_path = os.path.join(output_dir, new_filename)

        print(f"Processing: {filename}...")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pairs = data.get('pairs', [])

            # 3. Process pairs
            for item in pairs:
                # Get raw content (handling potential missing keys gracefully)
                immediate_raw = item.get('immediate', "")
                long_term_raw = item.get('long_term', "")

                # clean text (remove whatever label was there before)
                imm_text = label_pattern.sub('', immediate_raw)
                lt_text = label_pattern.sub('', long_term_raw)

                # ENFORCE FIXED ASSIGNMENT
                # Immediate is always (A)
                # Long-term is always (B)
                item['immediate'] = f"(A) {imm_text}"
                item['long_term'] = f"(B) {lt_text}"

            # 4. Save to NEW file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            files_processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nDone. Processed {files_processed} files.")
    print(f"New invariant files are located in: {output_dir}")

if __name__ == "__main__":
    revert_to_invariant_AB()