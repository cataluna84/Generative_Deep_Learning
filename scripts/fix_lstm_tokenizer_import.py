#!/usr/bin/env python3
"""Fix Keras 3.0+ import for Tokenizer in LSTM notebook.

In Keras 3.0+, keras.preprocessing.text has been moved to 
tensorflow.keras.preprocessing.text.
"""

import json
import sys

NOTEBOOK_PATH = "v1/notebooks/06_01_lstm_text_train.ipynb"

def fix_tokenizer_import():
    """Fix the Tokenizer import for Keras 3.0+ compatibility."""
    
    # Read the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the imports cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from keras.preprocessing.text import Tokenizer' in source:
                # Fix the import
                new_source = []
                for line in cell['source']:
                    if 'from keras.preprocessing.text import Tokenizer' in line:
                        new_source.append("from tensorflow.keras.preprocessing.text import Tokenizer\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                # Clear any error outputs
                cell['outputs'] = []
                cell['execution_count'] = None
                print("✓ Fixed Tokenizer import")
                break
    
    # Write back
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Saved {NOTEBOOK_PATH}")

if __name__ == "__main__":
    fix_tokenizer_import()
