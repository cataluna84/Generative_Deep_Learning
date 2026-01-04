#!/usr/bin/env python3
"""
Script to fix LRFinder cell in 03_03_vae_digits_train.ipynb notebook.
Uses keras.ops instead of keras.backend for Keras 3.0 compatibility.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "v1", "notebooks", "03_03_vae_digits_train.ipynb"
)


def main():
    """Fix LRFinder cell in the notebook."""
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Find and fix the LRFinder cell
            if 'LRFINDER - FIND OPTIMAL LEARNING RATE' in source:
                cell['source'] = [
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "# LRFINDER - FIND OPTIMAL LEARNING RATE\n",
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "# Clone the model and run LRFinder to find optimal learning rate.\n",
                    "# The sampling function is now serializable, enabling model cloning.\n",
                    "\n",
                    "import tensorflow as tf\n",
                    "from keras.optimizers import Adam\n",
                    "import keras.ops as ops  # Keras 3.0+ uses ops instead of backend\n",
                    "\n",
                    "# Define reconstruction loss for LRFinder (same as VAE uses)\n",
                    "def vae_r_loss(y_true, y_pred):\n",
                    "    \"\"\"Weighted reconstruction loss (MSE) for LRFinder.\"\"\"\n",
                    "    return R_LOSS_FACTOR * ops.mean(ops.square(y_true - y_pred), axis=[1, 2, 3])\n",
                    "\n",
                    "# Clone model for LRFinder\n",
                    "print(\"Cloning model for LRFinder...\")\n",
                    "lr_model = tf.keras.models.clone_model(vae.model)\n",
                    "lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))\n",
                    "print(\"✓ Model cloned successfully\")\n",
                    "\n",
                    "# Run LRFinder\n",
                    "lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)\n",
                    "lr_model.fit(x_train, x_train, epochs=2, batch_size=BATCH_SIZE, callbacks=[lr_finder], verbose=0)\n",
                    "\n",
                    "# Plot and get optimal LR\n",
                    "lr_finder.plot_loss()\n",
                    "OPTIMAL_LR = lr_finder.get_optimal_lr()\n",
                    "print(f\"\\nOptimal Learning Rate: {OPTIMAL_LR:.2e}\")\n",
                ]
                cell['outputs'] = []  # Clear old outputs
                cell['execution_count'] = None
                print("✅ Fixed LRFinder cell to use keras.ops")
                break

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"✅ Notebook saved: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
