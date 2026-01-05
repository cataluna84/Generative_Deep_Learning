#!/usr/bin/env python3
"""
Fix keras.backend.mean/square error in VAE Faces notebook.

In Keras 3.x, K.mean and K.square are replaced with keras.ops.mean/square
"""

import json

NOTEBOOK_PATH = "v1/notebooks/03_05_vae_faces_train.ipynb"

with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

# Fix 1: Update imports to use keras.ops
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "import keras.backend as K" in source:
        print(f"Found Imports cell at index {i}")
        
        new_source = [
            "import sys\n",
            "sys.path.insert(0, '../..')\n",
            "sys.path.insert(0, '..')\n",
            "\n",
            "import os\n",
            "from glob import glob\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# TensorFlow/Keras\n",
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
            "from keras.optimizers import Adam\n",
            "import keras.ops as ops  # Keras 3.x: use ops instead of backend\n",
            "\n",
            "# Local imports\n",
            "from src.models.VAE import VariationalAutoencoder\n",
            "\n",
            "# W&B and utilities\n",
            "import wandb\n",
            "from wandb.integration.keras import WandbMetricsLogger\n",
            "from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger\n",
            "from utils.gpu_utils import (\n",
            "    find_optimal_batch_size,\n",
            "    calculate_adjusted_epochs,\n",
            "    get_gpu_vram_gb,\n",
            "    print_training_config\n",
            ")"
        ]
        nb["cells"][i]["source"] = new_source
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print("  ✓ Fixed: keras.backend -> keras.ops")
        break

# Fix 2: Update LRFinder cell to use keras.ops
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "K.mean(K.square" in source:
        print(f"Found LRFinder cell at index {i}")
        
        new_source = [
            "# LRFinder: Uses dynamically determined BATCH_SIZE via data generator\n",
            "NUM_LR_STEPS = 100\n",
            "print(f\"Running LRFinder with BATCH_SIZE={BATCH_SIZE} for {NUM_LR_STEPS} steps...\")\n",
            "\n",
            "# For VAE with custom Lambda layers, save weights instead of cloning\n",
            "initial_weights = vae.model.get_weights()\n",
            "\n",
            "# VAE reconstruction loss (using keras.ops for Keras 3.x compatibility)\n",
            "def vae_r_loss(y_true, y_pred):\n",
            '    """Reconstruction loss for VAE."""\n',
            "    return R_LOSS_FACTOR * ops.mean(ops.square(y_true - y_pred), axis=[1, 2, 3])\n",
            "\n",
            "vae.model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))\n",
            "\n",
            "# Run LRFinder with data_flow to use the global BATCH_SIZE\n",
            "lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-2, steps=NUM_LR_STEPS)\n",
            "vae.model.fit(\n",
            "    data_flow,\n",
            "    epochs=4,\n",
            "    steps_per_epoch=NUM_LR_STEPS,\n",
            "    callbacks=[lr_finder],\n",
            "    verbose=0\n",
            ")\n",
            "\n",
            "# Plot and get optimal LR\n",
            "lr_finder.plot_loss()\n",
            "optimal_lr = lr_finder.get_optimal_lr()\n",
            "print(f\"\\nOptimal learning rate: {optimal_lr:.6f}\")\n",
            "\n",
            "# Update W&B config\n",
            "wandb.config.update({\"learning_rate\": optimal_lr})\n",
            "\n",
            "# Restore original weights after LRFinder\n",
            "vae.model.set_weights(initial_weights)\n",
            "tf.keras.backend.clear_session()"
        ]
        nb["cells"][i]["source"] = new_source
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print("  ✓ Fixed: K.mean/K.square -> ops.mean/ops.square")
        break

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("\n✓ Keras 3.x compatibility fixed!")
