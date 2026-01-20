#!/usr/bin/env python3
"""Fix data file path in LSTM notebook.

The notebook is in v1/notebooks/ and the data is in v1/data/,
so the path should be ../data/aesop/data.txt not ./data/aesop/data.txt
"""

import json

NOTEBOOK_PATH = "v1/notebooks/06_01_lstm_text_train.ipynb"

def fix_data_path():
    """Fix the data file path for correct relative location."""
    
    # Read the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the data loading cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if './data/aesop/data.txt' in source:
                # Fix the path
                new_source = []
                for line in cell['source']:
                    if './data/aesop/data.txt' in line:
                        new_source.append(line.replace('./data/aesop/data.txt', '../data/aesop/data.txt'))
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                # Clear any error outputs
                cell['outputs'] = []
                cell['execution_count'] = None
                print("✓ Fixed data file path: ./data/aesop/data.txt -> ../data/aesop/data.txt")
                break
    
    # Write back
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Saved {NOTEBOOK_PATH}")

if __name__ == "__main__":
    fix_data_path()
