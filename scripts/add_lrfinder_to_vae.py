#!/usr/bin/env python3
"""
Script to add LRFinder to 03_03_vae_digits_train.ipynb notebook.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "v1", "notebooks", "03_03_vae_digits_train.ipynb"
)


def main():
    """Apply LRFinder changes to the notebook."""
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    new_cells = []

    for i, cell in enumerate(cells):
        # Find and update imports cell
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Update imports to add LRFinder
            if 'from utils.callbacks import get_lr_scheduler' in source and 'LRFinder' not in source:
                cell['source'] = [line.replace(
                    'from utils.callbacks import get_lr_scheduler, get_early_stopping, LRLogger',
                    'from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger'
                ) for line in cell['source']]
                print("✅ Added LRFinder to imports")
            
            # Replace the training hyperparameters cell
            if 'VAE models cannot use LRFinder' in source:
                cell['source'] = [
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "# TRAINING HYPERPARAMETERS\n",
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "\n",
                    "LEARNING_RATE = 0.0005\n",
                    "R_LOSS_FACTOR = 1000  # Reconstruction loss multiplier\n",
                ]
                print("✅ Removed old LRFinder exception comment")
                
                # Insert LRFinder cell after this one
                new_cells.append(cell)
                
                # Add LRFinder section header
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## LRFinder (Optimal Learning Rate)"]
                })
                
                # Add LRFinder cell
                new_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# ═══════════════════════════════════════════════════════════════════════════════\n",
                        "# LRFINDER - FIND OPTIMAL LEARNING RATE\n",
                        "# ═══════════════════════════════════════════════════════════════════════════════\n",
                        "# Clone the model and run LRFinder to find optimal learning rate.\n",
                        "# The sampling function is now serializable, enabling model cloning.\n",
                        "\n",
                        "import tensorflow as tf\n",
                        "from keras.optimizers import Adam\n",
                        "import keras.backend as K\n",
                        "\n",
                        "# Define reconstruction loss for LRFinder (same as VAE uses)\n",
                        "def vae_r_loss(y_true, y_pred):\n",
                        "    return R_LOSS_FACTOR * K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])\n",
                        "\n",
                        "# Clone model for LRFinder\n",
                        "print(\"Cloning model for LRFinder...\")\n",
                        "lr_model = tf.keras.models.clone_model(vae.model)\n",
                        "lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))\n",
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
                })
                print("✅ Added LRFinder section")
                continue  # Skip adding this cell again
        
        new_cells.append(cell)

    # Update notebook cells
    nb['cells'] = new_cells

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Notebook saved: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
