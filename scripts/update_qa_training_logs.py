#!/usr/bin/env python3
"""
Script to add persistent training logs to the QA training notebook.

This adds tqdm.write() calls for:
1. Every 10 batches: Batch X | Train: X.XXXX | Test: X.XXXX
2. Every epoch: Epoch summary with avg losses
"""

import json

NOTEBOOK_PATH = "v1/notebooks/06_02_qa_train.ipynb"


def main():
    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    training_loop_updated = False

    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # Find the training loop cell and add logging
        if 'batch_pbar.set_postfix' in source and 'Batch {i:3d}' not in source:
            new_source = []
            i = 0
            while i < len(cell['source']):
                line = cell['source'][i]
                
                # After the batch_pbar.set_postfix block, add persistent logging every 10 batches
                if "        })\n" in line and i > 0 and "'lr': f'{current_lr:.2e}'" in cell['source'][i-1]:
                    new_source.append(line)
                    new_source.append("        \n")
                    new_source.append("        # Log every 10 batches for persistent history\n")
                    new_source.append("        if i % 10 == 0:\n")
                    new_source.append('            tqdm.write(f"  Batch {i:3d} | Train: {training_loss[0]:.4f} | Test: {test_loss[0]:.4f}")\n')
                    i += 1
                    continue
                
                # After epoch_pbar.set_postfix block, add epoch summary log
                if "    })\n" in line and i > 0 and "'lr': f'{current_lr:.2e}'" in cell['source'][i-1] and "epoch_pbar" in source:
                    new_source.append(line)
                    new_source.append("    \n")
                    new_source.append("    # Log epoch summary for persistent history\n")
                    new_source.append('    tqdm.write(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.2e}")\n')
                    i += 1
                    continue
                
                new_source.append(line)
                i += 1
            
            cell['source'] = new_source
            training_loop_updated = True
            print("✓ Added persistent batch and epoch logging")

    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)

    if training_loop_updated:
        print("\n✓ Notebook updated successfully!")
        return 0
    else:
        print("\n⚠ Could not find the training loop to update (may already have logging)")
        return 1


if __name__ == "__main__":
    exit(main())
