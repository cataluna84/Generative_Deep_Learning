#!/usr/bin/env python3
"""Remove the numerical stability epsilon from sample_with_temp function."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "v1/notebooks/06_01_lstm_text_train.ipynb"

# The original code without epsilon
ORIGINAL_SAMPLE_CODE = '''def sample_with_temp(preds, temperature=1.0):
    """Sample an index from a probability array using temperature scaling.
    
    Args:
        preds: Probability distribution over vocabulary
        temperature: Controls randomness (lower = more deterministic)
        
    Returns:
        Sampled token index
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)'''

def update_sample_function():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with sample_with_temp function
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def sample_with_temp' in source:
                # Replace with original code (without epsilon)
                new_source = ORIGINAL_SAMPLE_CODE.split('\n')
                # Reconstruct with proper line endings
                formatted_source = []
                for i, line in enumerate(new_source):
                    if i < len(new_source) - 1:
                        formatted_source.append(line + '\n')
                    else:
                        formatted_source.append(line)
                
                # Check if generate_text is in the same cell
                if 'def generate_text' in source:
                    # Find generate_text function and append it
                    lines = source.split('\n')
                    in_generate_text = False
                    generate_text_lines = []
                    for line in lines:
                        if 'def generate_text' in line:
                            in_generate_text = True
                        if in_generate_text:
                            generate_text_lines.append(line)
                    
                    # Add blank lines and generate_text function
                    formatted_source.append('\n')
                    formatted_source.append('\n')
                    for i, line in enumerate(generate_text_lines):
                        if i < len(generate_text_lines) - 1:
                            formatted_source.append(line + '\n')
                        else:
                            formatted_source.append(line)
                
                cell['source'] = formatted_source
                print("Removed epsilon fix from sample_with_temp function.")
                print("Restored original code without numerical stability fix.")
                break
    
    # Save the notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated notebook saved to: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_sample_function()
