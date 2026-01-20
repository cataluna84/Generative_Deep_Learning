#!/usr/bin/env python3
"""Add Run 009 to the Master Experiment Log."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "v1/notebooks/06_01_lstm_text_train.ipynb"

# Run 009 data
NEW_ROW = "| 009 | 2026-01-20 | [View](https://wandb.ai/cataluna84/generative-deep-learning/runs/zf3g2y4u) | 1000 | 10240 | word | 0.001→0.00013 | 2.5386 | Removed epsilon fix from sample_with_temp |"

def update_experiment_log():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## Master Experiment Log' in source:
                if '| 009 |' in source:
                    print("Run 009 already exists in the experiment log.")
                    return
                
                # Find the last row (Run 008) and add the new row after it
                new_source = []
                for line in cell['source']:
                    if '| 008 |' in line:
                        # Add newline if not present
                        if not line.endswith('\\n",\n') and not line.endswith('\\n"'):
                            new_source.append(line.rstrip('"') + '\\n",\n')
                        else:
                            new_source.append(line.replace('|"', '|\\n",\n'))
                        # Add new row
                        new_source.append(f'    "{NEW_ROW}"')
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                print("Added Run 009 to the experiment log.")
                break
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_experiment_log()
