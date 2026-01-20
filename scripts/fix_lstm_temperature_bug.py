#!/usr/bin/env python3
"""
Fix the temperature sampling bug in 06_01_lstm_text_train.ipynb

The issue: np.log(preds) returns -inf when preds contains zeros or very small values.
This causes NaN in the probability distribution, making the model immediately predict
the story delimiter '|', resulting in empty generated text.

The fix: Add a small epsilon to prevent log(0).
"""

import json
import sys
from pathlib import Path

def fix_notebook():
    notebook_path = Path(__file__).parent.parent / "v1/notebooks/06_01_lstm_text_train.ipynb"
    
    print(f"Reading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the sample_with_temp function
    fixed_count = 0
    
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = cell.get('source', [])
        if not isinstance(source, list):
            continue
            
        source_str = ''.join(source)
            
        # Check if this cell contains the sample_with_temp function
        if 'def sample_with_temp' in source_str and 'np.log(preds)' in source_str:
            print("Found sample_with_temp function, applying fix...")
            
            # Look for the problematic line
            new_source = []
            for line in source:
                # Match the exact line format (with or without trailing newline)
                if 'preds = np.log(preds) / temperature' in line and 'epsilon' not in source_str:
                    # Add comprehensive comment block and the epsilon fix
                    new_source.append("    # ═══════════════════════════════════════════════════════════════════\\n")
                    new_source.append("    # BUG FIX: Numerical stability for temperature sampling\\n")
                    new_source.append("    # ═══════════════════════════════════════════════════════════════════\\n")
                    new_source.append("    # PROBLEM: When the model becomes more confident after training,\\n")
                    new_source.append("    #          some probabilities become extremely small (near 0).\\n")
                    new_source.append("    #          np.log(0) = -inf, which when divided by temperature\\n")
                    new_source.append("    #          causes NaN values in the probability distribution.\\n")
                    new_source.append("    #\\n")
                    new_source.append("    # SYMPTOM: Empty text generation at later epochs (50, 100, etc.)\\n")
                    new_source.append("    #          while Epoch 0 works fine (uniform distribution = no zeros).\\n")
                    new_source.append("    #          Lower temperatures (0.2, 0.5) fail more often than 1.0\\n")
                    new_source.append("    #          because dividing -inf by smaller values amplifies the issue.\\n")
                    new_source.append("    #\\n")
                    new_source.append("    # SOLUTION: Add a small epsilon (1e-7) to prevent log(0).\\n")
                    new_source.append("    #           This is a standard practice in numerical computing.\\n")
                    new_source.append("    # ═══════════════════════════════════════════════════════════════════\\n")
                    new_source.append("    epsilon = 1e-7\\n")
                    new_source.append("    preds = np.log(preds + epsilon) / temperature\\n")
                    fixed_count += 1
                    print(f"  Replaced line: {line.strip()}")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            if fixed_count > 0:
                print("✓ Fixed sample_with_temp function with comprehensive comments")
    
    if fixed_count == 0:
        # Check if already fixed
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source_str = ''.join(cell.get('source', []))
            if 'epsilon = 1e-7' in source_str and 'sample_with_temp' in source_str:
                print("✓ The fix has already been applied.")
                return True
        
        print("WARNING: Could not find the line to fix.")
        return False
    
    # Write the fixed notebook
    print(f"Writing fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Successfully fixed {fixed_count} occurrence(s)")
    print("\nThe fix adds epsilon to prevent log(0):")
    print("  OLD: preds = np.log(preds) / temperature")
    print("  NEW: preds = np.log(preds + epsilon) / temperature")
    print("\nYou may need to restart the kernel and re-run the notebook.")
    return True

if __name__ == "__main__":
    success = fix_notebook()
    sys.exit(0 if success else 1)
