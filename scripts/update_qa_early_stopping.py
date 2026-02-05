#!/usr/bin/env python3
"""
Script to update the QA training notebook with early stopping and min_delta.

This script modifies 06_02_qa_train.ipynb to add:
1. MIN_DELTA configuration parameter
2. EARLY_STOP_PATIENCE configuration parameter
3. Updated ReduceLROnPlateau logic with min_delta threshold
4. Early stopping with best weights saving/restoration
5. Debug output for monitoring no-improvement epochs
"""

import json

NOTEBOOK_PATH = "v1/notebooks/06_02_qa_train.ipynb"

def main():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    config_updated = False
    training_loop_updated = False

    # Find and update the cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Update configuration cell - add MIN_DELTA and EARLY_STOP_PATIENCE
            if 'LR_PATIENCE = 5' in source and 'MIN_DELTA' not in source:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if 'MIN_LR = 1e-7' in line:
                        new_source.append("MIN_DELTA = 0.001        # Minimum improvement to count as progress\n")
                        new_source.append("\n")
                        new_source.append("# Early Stopping parameters\n")
                        new_source.append("EARLY_STOP_PATIENCE = 10 # Epochs with no improvement before stopping\n")
                cell['source'] = new_source
                config_updated = True
                print("✓ Updated configuration cell with MIN_DELTA and EARLY_STOP_PATIENCE")

            # Update training loop - add early stopping logic
            if 'ReduceLROnPlateau equivalent' in source and 'early_stop_count' not in source:
                new_source = []
                in_reduce_lr_section = False
                skip_next_reduce_lr_comment = False
                
                i = 0
                while i < len(cell['source']):
                    line = cell['source'][i]
                    
                    # Add early_stop_count after no_improve_count initialization
                    if line == "no_improve_count = 0\n":
                        new_source.append(line)
                        new_source.append("early_stop_count = 0\n")
                        i += 1
                        continue
                    
                    # Replace the ReduceLROnPlateau comment and condition
                    if "# ReduceLROnPlateau equivalent\n" in line:
                        new_source.append("    # ReduceLROnPlateau equivalent with min_delta\n")
                        i += 1
                        continue
                    
                    # Update condition to use MIN_DELTA
                    if "if avg_test_loss < best_loss:\n" in line:
                        new_source.append("    if avg_test_loss < (best_loss - MIN_DELTA):\n")
                        i += 1
                        continue
                    
                    # After best_loss update and no_improve_count reset, add early_stop_count reset and best weights save
                    if "        no_improve_count = 0\n" in line and i > 0 and "best_loss = avg_test_loss" in cell['source'][i-1]:
                        new_source.append(line)
                        new_source.append("        early_stop_count = 0\n")
                        new_source.append("        # Save best weights\n")
                        new_source.append("        total_model.save_weights(\n")
                        new_source.append("            os.path.join(RUN_FOLDER, 'weights/best_weights.weights.h5')\n")
                        new_source.append("        )\n")
                        i += 1
                        continue
                    
                    # After no_improve_count increment, add early_stop_count and debug message
                    if "        no_improve_count += 1\n" in line:
                        new_source.append(line)
                        new_source.append("        early_stop_count += 1\n")
                        new_source.append('        print(f"  ⚠ No improvement for {no_improve_count}/{LR_PATIENCE} epochs (LR), {early_stop_count}/{EARLY_STOP_PATIENCE} epochs (ES)")\n')
                        i += 1
                        continue
                    
                    # After the final no_improve_count = 0 (after LR reduction), add early stopping check
                    if "            no_improve_count = 0\n" in line and i > 0 and "Reducing LR" in cell['source'][i-1]:
                        new_source.append(line)
                        new_source.append("    \n")
                        new_source.append("    # Early Stopping check\n")
                        new_source.append("    if early_stop_count >= EARLY_STOP_PATIENCE:\n")
                        new_source.append('        print(f"\\n✓ Early stopping triggered after {epoch} epochs")\n')
                        new_source.append('        print(f"  Best test loss: {best_loss:.4f}")\n')
                        new_source.append("        # Load best weights\n")
                        new_source.append("        total_model.load_weights(\n")
                        new_source.append("            os.path.join(RUN_FOLDER, 'weights/best_weights.weights.h5')\n")
                        new_source.append("        )\n")
                        new_source.append("        break\n")
                        i += 1
                        continue
                    
                    new_source.append(line)
                    i += 1
                
                cell['source'] = new_source
                training_loop_updated = True
                print("✓ Updated training loop with early stopping and min_delta logic")

    # Save the notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)

    if config_updated and training_loop_updated:
        print("\n✓ Notebook updated successfully!")
        return 0
    else:
        print("\n⚠ Some updates may not have been applied:")
        print(f"  - Configuration: {'Updated' if config_updated else 'Not updated'}")
        print(f"  - Training loop: {'Updated' if training_loop_updated else 'Not updated'}")
        return 1


if __name__ == "__main__":
    exit(main())
