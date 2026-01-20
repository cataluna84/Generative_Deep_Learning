#!/usr/bin/env python3
"""
Add comprehensive comments to the temperature sampling fix in 06_01_lstm_text_train.ipynb
"""

import json
import sys
from pathlib import Path

def add_comments():
    notebook_path = Path(__file__).parent.parent / "v1/notebooks/06_01_lstm_text_train.ipynb"
    
    print(f"Reading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixed_count = 0
    
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = cell.get('source', [])
        if not isinstance(source, list):
            continue
            
        source_str = ''.join(source)
            
        # Check if this cell contains the sample_with_temp function with the fix
        if 'def sample_with_temp' in source_str and 'epsilon = 1e-7' in source_str:
            # Check if comprehensive comments already exist
            if 'BUG FIX: Numerical stability' in source_str:
                print("✓ Comprehensive comments already exist.")
                return True
                
            print("Found sample_with_temp function with fix, adding comments...")
            
            new_source = []
            for line in source:
                # Find the epsilon line and add comments before it
                if 'epsilon = 1e-7' in line:
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
                    fixed_count += 1
                # Skip the old simple comment if it exists
                if 'Add epsilon to prevent log(0)' in line:
                    continue
                new_source.append(line)
            
            cell['source'] = new_source
            if fixed_count > 0:
                print("✓ Added comprehensive comments")
    
    if fixed_count == 0:
        print("WARNING: Could not find the location to add comments.")
        return False
    
    # Write the updated notebook
    print(f"Writing updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Successfully added comprehensive comments")
    return True

if __name__ == "__main__":
    success = add_comments()
    sys.exit(0 if success else 1)
