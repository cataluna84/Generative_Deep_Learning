#!/usr/bin/env python3
"""
Script to add tqdm progress bars to the QA training notebook.

This script modifies 06_02_qa_train.ipynb to:
1. Add tqdm.notebook import
2. Replace epoch loop with trange
3. Add batch progress bar with running metrics
4. Update postfix with running train/test loss
5. Clean up verbose print statements
"""

import json

NOTEBOOK_PATH = "v1/notebooks/06_02_qa_train.ipynb"


def main():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    import_updated = False
    training_loop_updated = False

    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # 1. Add tqdm import after numpy import
        if 'import numpy as np' in source and 'from tqdm' not in source:
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'import numpy as np' in line:
                    new_source.append("from tqdm.notebook import tqdm, trange\n")
            cell['source'] = new_source
            import_updated = True
            print("✓ Added tqdm import")

        # 2. Update training loop with tqdm
        if 'for epoch in range(start_epoch, start_epoch + EPOCHS + 1):' in source:
            new_source = []
            i = 0
            while i < len(cell['source']):
                line = cell['source'][i]
                
                # Replace "Starting training" print with tqdm description
                if 'print(f"\\nStarting training for {EPOCHS} epochs...")' in line:
                    # Skip this line and the next separator line
                    i += 1
                    if i < len(cell['source']) and '"=" * 60' in cell['source'][i]:
                        i += 1
                    continue
                
                # Replace epoch loop with trange
                if 'for epoch in range(start_epoch, start_epoch + EPOCHS + 1):' in line:
                    new_source.append("epoch_pbar = trange(start_epoch, start_epoch + EPOCHS + 1, desc='Training', unit='epoch')\n")
                    new_source.append("\n")
                    new_source.append("for epoch in epoch_pbar:\n")
                    i += 1
                    continue
                
                # Remove old epoch print
                if 'print(f"\\nEpoch {epoch}/{start_epoch + EPOCHS}")' in line:
                    i += 1
                    continue
                
                # Add tqdm wrapper to batch loop
                if 'for i, batch in enumerate(training_data()):' in line:
                    new_source.append("    batch_pbar = tqdm(enumerate(training_data()), desc=f'Epoch {epoch}', leave=False)\n")
                    new_source.append("    \n")
                    new_source.append("    for i, batch in batch_pbar:\n")
                    i += 1
                    continue
                
                # Replace batch print with postfix update
                if '# Print progress every 10 batches' in line:
                    # Skip the print block
                    i += 1
                    if i < len(cell['source']) and 'if i % 10 == 0:' in cell['source'][i]:
                        i += 1
                        if i < len(cell['source']) and 'print(f"  Batch {i}:' in cell['source'][i]:
                            i += 1
                    # Add postfix update instead
                    new_source.append("        # Update batch progress bar\n")
                    new_source.append("        batch_pbar.set_postfix({\n")
                    new_source.append("            'train': f'{training_loss[0]:.4f}',\n")
                    new_source.append("            'test': f'{test_loss[0]:.4f}',\n")
                    new_source.append("            'lr': f'{current_lr:.2e}'\n")
                    new_source.append("        })\n")
                    continue
                
                # Remove epoch complete print statement
                if 'print(f"  Epoch {epoch} complete: Avg Train Loss:' in line:
                    # Replace with epoch_pbar postfix update
                    new_source.append("    # Update epoch progress bar with summary\n")
                    new_source.append("    epoch_pbar.set_postfix({\n")
                    new_source.append("        'train': f'{avg_train_loss:.4f}',\n")
                    new_source.append("        'test': f'{avg_test_loss:.4f}',\n")
                    new_source.append("        'best': f'{best_loss:.4f}',\n")
                    new_source.append("        'lr': f'{current_lr:.2e}'\n")
                    new_source.append("    })\n")
                    i += 1
                    continue
                
                # Update early stopping message to use tqdm.write
                if 'print(f"  ⚠ No improvement for' in line:
                    new_source.append('        tqdm.write(f"  ⚠ No improvement for {no_improve_count}/{LR_PATIENCE} epochs (LR), {early_stop_count}/{EARLY_STOP_PATIENCE} epochs (ES)")\n')
                    i += 1
                    continue
                
                # Update LR reduction message to use tqdm.write
                if 'print(f"  → Reducing LR to' in line:
                    new_source.append('            tqdm.write(f"  → Reducing LR to {current_lr:.2e}")\n')
                    i += 1
                    continue
                
                # Update early stopping triggered messages to use tqdm.write  
                if 'print(f"\\n✓ Early stopping triggered' in line:
                    new_source.append('        tqdm.write(f"\\n✓ Early stopping triggered after {epoch} epochs")\n')
                    i += 1
                    continue
                
                if 'print(f"  Best test loss: {best_loss:.4f}")' in line:
                    new_source.append('        tqdm.write(f"  Best test loss: {best_loss:.4f}")\n')
                    i += 1
                    continue
                
                # Update final "Training complete" message to use tqdm.write
                if 'print("\\n" + "=" * 60)' in line:
                    new_source.append('tqdm.write("\\n" + "=" * 60)\n')
                    i += 1
                    continue
                
                if 'print("Training complete!")' in line:
                    new_source.append('tqdm.write("Training complete!")\n')
                    i += 1
                    continue
                
                if 'print("=" * 60)' in line and i == len(cell['source']) - 1:
                    new_source.append('tqdm.write("=" * 60)\n')
                    i += 1
                    continue
                
                new_source.append(line)
                i += 1
            
            cell['source'] = new_source
            training_loop_updated = True
            print("✓ Updated training loop with tqdm progress bars")

    # Save the notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)

    if import_updated and training_loop_updated:
        print("\n✓ Notebook updated successfully!")
        return 0
    else:
        print("\n⚠ Some updates may not have been applied:")
        print(f"  - Import: {'Updated' if import_updated else 'Not updated (may already have tqdm)'}")
        print(f"  - Training loop: {'Updated' if training_loop_updated else 'Not updated'}")
        return 1


if __name__ == "__main__":
    exit(main())
