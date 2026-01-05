#!/usr/bin/env python3
"""
Fix VAE Faces Training Notebook Early Stopping Configuration.

Updates the early stopping min_delta from 1e-4 to 0.5 and adds comprehensive
comments explaining the callback configuration rationale.

Problem: The original min_delta of 1e-4 is far too small for VAE losses
in the 200-340 range, making it effectively zero. This causes early stopping
to trigger based on normal training fluctuations rather than true stagnation.

Solution: Use min_delta=0.5 which represents a meaningful improvement
(~0.15-0.25% of typical loss values).
"""

import json
import re


def fix_early_stopping(notebook_path: str) -> None:
    """
    Update the early stopping configuration in the notebook.
    
    Args:
        notebook_path: Path to the notebook file.
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update the training callbacks cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = ''.join(source) if isinstance(source, list) else source
            
            # Find the cell with the extra_callbacks definition
            if 'get_early_stopping(monitor=\'loss\'' in source_text:
                # Build the new source with comprehensive comments
                new_source = [
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# TRAINING CALLBACKS CONFIGURATION\n",
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# - WandbMetricsLogger: Logs all metrics to Weights & Biases\n",
                    "# - get_lr_scheduler: Reduces LR when loss plateaus (patience=5 epochs)\n",
                    "# - get_early_stopping: Stops training when loss stops improving\n",
                    "#   * patience=20: Wait 20 epochs without improvement before stopping\n",
                    "#   * min_delta=0.5: Minimum improvement required to count as progress.\n",
                    "#     NOTE: For VAE losses typically in the 200-300 range, a min_delta\n",
                    "#     of 1e-4 is effectively zero (too small to detect meaningful change).\n",
                    "#     Using 0.5 (~0.2% of loss) ensures only significant improvements\n",
                    "#     are recognized, preventing premature early stopping.\n",
                    "# - LRLogger: Logs current learning rate each epoch for debugging\n",
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "\n",
                    "# Compile with optimal LR\n",
                    "vae.compile(optimal_lr, R_LOSS_FACTOR)\n",
                    "\n",
                    "# Define extra callbacks for W&B tracking and training optimization\n",
                    "extra_callbacks = [\n",
                    "    WandbMetricsLogger(),\n",
                    "    get_lr_scheduler(monitor='loss', patience=5, factor=0.5),\n",
                    "    get_early_stopping(monitor='loss', patience=20, min_delta=0.5),\n",
                    "    LRLogger(),\n",
                    "]\n",
                    "\n",
                    "print(f\"Training with optimal LR: {optimal_lr:.6f}\")\n",
                    "print(f\"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}\")\n",
                    "print(f\"Steps per epoch: {NUM_IMAGES // BATCH_SIZE}\")"
                ]
                
                cell['source'] = new_source
                print("✓ Updated early stopping configuration:")
                print("  - Changed min_delta from 1e-4 to 0.5")
                print("  - Added comprehensive callback documentation")
                break
    else:
        print("✗ Could not find the extra_callbacks cell")
        return
    
    # Write the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n✓ Notebook saved: {notebook_path}")


if __name__ == "__main__":
    notebook_path = "v1/notebooks/03_05_vae_faces_train.ipynb"
    fix_early_stopping(notebook_path)
