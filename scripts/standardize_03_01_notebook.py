#!/usr/bin/env python3
"""
Script to standardize 03_01_autoencoder_train.ipynb.

This notebook is partially standardized - adds dynamic scaling and fixes structure.

Usage:
    uv run python scripts/standardize_03_01_notebook.py
"""

import json
import os


def create_standardized_notebook():
    """Create a fully standardized autoencoder notebook."""
    
    cells = []
    
    # =========================================================================
    # Cell 0: Header Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Autoencoder Training\n",
            "\n",
            "This notebook trains an Autoencoder on the MNIST dataset to learn a 2D latent\n",
            "representation of handwritten digits.\n",
            "\n",
            "**Standards Applied:**\n",
            "- ✅ GPU memory growth enabled\n",
            "- ✅ Global configuration block with dynamic batch/epoch scaling\n",
            "- ✅ W&B integration for experiment tracking\n",
            "- ✅ LRFinder for optimal learning rate detection\n",
            "- ✅ Full callback stack (WandbMetricsLogger, LRScheduler, EarlyStopping, LRLogger)\n",
            "- ✅ Enhanced training visualizations\n",
            "- ✅ Kernel restart cell (commented out)"
        ]
    })
    
    # =========================================================================
    # Cell 1: GPU Setup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## GPU Setup"]
    })
    
    # =========================================================================
    # Cell 2: GPU Memory Growth Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GPU MEMORY GROWTH\n",
            "# Enable memory growth to prevent TensorFlow from allocating all GPU memory\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "import tensorflow as tf\n",
            "\n",
            "gpus = tf.config.list_physical_devices('GPU')\n",
            "if gpus:\n",
            "    for gpu in gpus:\n",
            "        tf.config.experimental.set_memory_growth(gpu, True)\n",
            "    print(f\"GPU(s) available: {[gpu.name for gpu in gpus]}\")\n",
            "else:\n",
            "    print(\"WARNING: No GPU detected, running on CPU\")"
        ]
    })
    
    # =========================================================================
    # Cell 3: Imports Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Imports"]
    })
    
    # =========================================================================
    # Cell 4: Imports Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Standard library imports\n",
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# TensorFlow/Keras imports\n",
            "import keras\n",
            "import keras.ops as K\n",
            "from keras.optimizers import Adam\n",
            "\n",
            "# Path setup for project utilities\n",
            "sys.path.insert(0, '../..')    # For project root utils/\n",
            "sys.path.insert(0, '..')       # For v1/src modules\n",
            "\n",
            "# Project utilities\n",
            "from src.utils.loaders import load_mnist\n",
            "from src.models.AE import Autoencoder\n",
            "from utils.wandb_utils import init_wandb, get_model_checkpoint\n",
            "from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger\n",
            "from utils.gpu_utils import (\n",
            "    get_optimal_batch_size,\n",
            "    calculate_adjusted_epochs,\n",
            "    get_gpu_vram_gb,\n",
            "    print_training_config\n",
            ")\n",
            "\n",
            "# W&B\n",
            "import wandb\n",
            "from wandb.integration.keras import WandbMetricsLogger"
        ]
    })
    
    # =========================================================================
    # Cell 5: Global Configuration Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Global Configuration"]
    })
    
    # =========================================================================
    # Cell 6: Global Configuration Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GLOBAL CONFIGURATION\n",
            "# All hyperparameters and settings are defined here for easy modification\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "# Reference values (original notebook settings for epoch scaling)\n",
            "REFERENCE_BATCH_SIZE = 32   # Original book value\n",
            "REFERENCE_EPOCHS = 200      # Original book value\n",
            "\n",
            "# Auto-detect GPU VRAM or override manually\n",
            "TARGET_VRAM_GB = None  # Set to 6, 8, 12, etc. to override detection\n",
            "GPU_VRAM_GB = TARGET_VRAM_GB if TARGET_VRAM_GB else get_gpu_vram_gb()\n",
            "\n",
            "# Calculate optimal settings dynamically based on GPU VRAM\n",
            "BATCH_SIZE = get_optimal_batch_size('ae', vram_gb=GPU_VRAM_GB)\n",
            "EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)\n",
            "\n",
            "# Ensure minimum epochs for meaningful training\n",
            "EPOCHS = max(EPOCHS, 100)\n",
            "\n",
            "# Model configuration\n",
            "INPUT_DIM = (28, 28, 1)\n",
            "Z_DIM = 2  # Latent space dimension\n",
            "ENCODER_FILTERS = [32, 64, 64, 64]\n",
            "ENCODER_KERNELS = [3, 3, 3, 3]\n",
            "ENCODER_STRIDES = [1, 2, 2, 1]\n",
            "DECODER_FILTERS = [64, 64, 32, 1]\n",
            "DECODER_KERNELS = [3, 3, 3, 3]\n",
            "DECODER_STRIDES = [1, 2, 2, 1]\n",
            "\n",
            "# Training configuration\n",
            "PRINT_EVERY_N_BATCHES = 100\n",
            "INITIAL_EPOCH = 0\n",
            "LEARNING_RATE = \"auto\"  # Will be set by LRFinder\n",
            "\n",
            "# W&B configuration\n",
            "MODEL_TYPE = \"autoencoder\"\n",
            "DATASET_NAME = \"digits\"\n",
            "OPTIMIZER_NAME = \"adam\"\n",
            "\n",
            "# Run folder configuration\n",
            "SECTION = 'vae'\n",
            "RUN_ID = '0001'\n",
            "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATASET_NAME}'\n",
            "\n",
            "# Create run directories\n",
            "if not os.path.exists(RUN_FOLDER):\n",
            "    os.makedirs(RUN_FOLDER, exist_ok=True)\n",
            "    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))\n",
            "    os.makedirs(os.path.join(RUN_FOLDER, 'images'))\n",
            "    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))\n",
            "\n",
            "MODE = 'build'  # 'build' for new model, 'load' to continue training\n",
            "\n",
            "# Print configuration summary\n",
            "print_training_config(\n",
            "    MODEL_TYPE, BATCH_SIZE, EPOCHS,\n",
            "    REFERENCE_BATCH_SIZE, REFERENCE_EPOCHS, GPU_VRAM_GB\n",
            ")"
        ]
    })
    
    # =========================================================================
    # Cell 7: Load Data Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Data"]
    })
    
    # =========================================================================
    # Cell 8: Load Data Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load MNIST dataset\n",
            "(x_train, y_train), (x_test, y_test) = load_mnist()\n",
            "\n",
            "print(f\"Training samples: {x_train.shape[0]}\")\n",
            "print(f\"Test samples: {x_test.shape[0]}\")\n",
            "print(f\"Image shape: {x_train.shape[1:]}\")"
        ]
    })
    
    # =========================================================================
    # Cell 9: W&B Init Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## W&B Initialization"]
    })
    
    # =========================================================================
    # Cell 10: W&B Init Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize W&B for experiment tracking\n",
            "run = init_wandb(\n",
            "    name=\"03_01_autoencoder\",\n",
            "    config={\n",
            "        \"model\": MODEL_TYPE,\n",
            "        \"dataset\": DATASET_NAME,\n",
            "        \"z_dim\": Z_DIM,\n",
            "        \"encoder_filters\": ENCODER_FILTERS,\n",
            "        \"learning_rate\": LEARNING_RATE,  # Will be updated after LRFinder\n",
            "        \"batch_size\": BATCH_SIZE,\n",
            "        \"epochs\": EPOCHS,\n",
            "        \"optimizer\": OPTIMIZER_NAME,\n",
            "        \"gpu_vram_gb\": GPU_VRAM_GB,\n",
            "    }\n",
            ")"
        ]
    })
    
    # =========================================================================
    # Cell 11: Model Architecture Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model Architecture"]
    })
    
    # =========================================================================
    # Cell 12: Model Architecture Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Build the Autoencoder model\n",
            "AE = Autoencoder(\n",
            "    input_dim=INPUT_DIM,\n",
            "    encoder_conv_filters=ENCODER_FILTERS,\n",
            "    encoder_conv_kernel_size=ENCODER_KERNELS,\n",
            "    encoder_conv_strides=ENCODER_STRIDES,\n",
            "    decoder_conv_t_filters=DECODER_FILTERS,\n",
            "    decoder_conv_t_kernel_size=DECODER_KERNELS,\n",
            "    decoder_conv_t_strides=DECODER_STRIDES,\n",
            "    z_dim=Z_DIM\n",
            ")\n",
            "\n",
            "if MODE == 'build':\n",
            "    AE.save(RUN_FOLDER)\n",
            "else:\n",
            "    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.weights.h5'))"
        ]
    })
    
    # =========================================================================
    # Cell 13: Encoder Summary Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Encoder summary\n",
            "AE.encoder.summary()"
        ]
    })
    
    # =========================================================================
    # Cell 14: Decoder Summary Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Decoder summary\n",
            "AE.decoder.summary()"
        ]
    })
    
    # =========================================================================
    # Cell 15: LRFinder Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Find Optimal Learning Rate"]
    })
    
    # =========================================================================
    # Cell 16: LRFinder Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clone the model to avoid pre-training the actual model\n",
            "lr_model = tf.keras.models.clone_model(AE.model)\n",
            "\n",
            "# Define reconstruction loss locally\n",
            "def r_loss_lr(y_true, y_pred):\n",
            "    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])\n",
            "\n",
            "# Compile with small initial LR\n",
            "lr_model.compile(loss=r_loss_lr, optimizer=Adam(learning_rate=1e-6))\n",
            "\n",
            "# Run LRFinder\n",
            "lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)\n",
            "lr_model.fit(\n",
            "    x_train, x_train,\n",
            "    batch_size=BATCH_SIZE,\n",
            "    epochs=2,\n",
            "    callbacks=[lr_finder],\n",
            "    verbose=0\n",
            ")\n",
            "\n",
            "# Visualize and get optimal LR\n",
            "lr_finder.plot_loss()\n",
            "LEARNING_RATE = lr_finder.get_optimal_lr()\n",
            "\n",
            "# Update W&B config\n",
            "wandb.config.update({\"learning_rate\": LEARNING_RATE})\n",
            "print(f\"\\nOptimal learning rate: {LEARNING_RATE:.2e}\")"
        ]
    })
    
    # =========================================================================
    # Cell 17: Train Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Train Autoencoder"]
    })
    
    # =========================================================================
    # Cell 18: Compile Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compile with optimal learning rate\n",
            "AE.compile(LEARNING_RATE)"
        ]
    })
    
    # =========================================================================
    # Cell 19: Train Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define full callback stack\n",
            "callbacks = [\n",
            "    WandbMetricsLogger(),\n",
            "    get_lr_scheduler(monitor='loss', patience=5),\n",
            "    get_early_stopping(monitor='loss', patience=10),\n",
            "    LRLogger(),\n",
            "]\n",
            "\n",
            "# Train the model\n",
            "# lr_decay=1 disables built-in step decay in favor of ReduceLROnPlateau\n",
            "AE.train(\n",
            "    x_train,\n",
            "    batch_size=BATCH_SIZE,\n",
            "    epochs=EPOCHS,\n",
            "    run_folder=RUN_FOLDER,\n",
            "    print_every_n_batches=PRINT_EVERY_N_BATCHES,\n",
            "    initial_epoch=INITIAL_EPOCH,\n",
            "    lr_decay=1,\n",
            "    extra_callbacks=callbacks\n",
            ")"
        ]
    })
    
    # =========================================================================
    # Cell 20: Training Visualization Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Training Visualization"]
    })
    
    # =========================================================================
    # Cell 21: Training Visualization Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot training history\n",
            "history = AE.model.history.history\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Plot 1: Training Loss\n",
            "axes[0].plot(history['loss'], 'b-', linewidth=2)\n",
            "axes[0].set_xlabel('Epoch')\n",
            "axes[0].set_ylabel('Loss')\n",
            "axes[0].set_title('Training Loss Over Epochs')\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 2: Learning Rate (LOG SCALE)\n",
            "if 'learning_rate' in history:\n",
            "    lr_data = np.array(history['learning_rate'])\n",
            "    lr_min, lr_max = lr_data.min(), lr_data.max()\n",
            "    \n",
            "    if lr_max > lr_min * 1.1:  # LR changed significantly\n",
            "        axes[1].semilogy(lr_data, 'g-', linewidth=2)\n",
            "    else:\n",
            "        axes[1].plot(lr_data, 'g-', linewidth=2)\n",
            "        axes[1].set_ylim([lr_min * 0.5, lr_max * 1.5])\n",
            "    \n",
            "    axes[1].set_xlabel('Epoch')\n",
            "    axes[1].set_ylabel('Learning Rate')\n",
            "    axes[1].set_title('Learning Rate Schedule')\n",
            "    axes[1].grid(True, alpha=0.3)\n",
            "else:\n",
            "    axes[1].text(0.5, 0.5, 'LR not tracked', ha='center', va='center', fontsize=14)\n",
            "    axes[1].set_title('Learning Rate (Not Available)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Print summary\n",
            "print(f\"\\n{'='*60}\")\n",
            "print(\"TRAINING SUMMARY\")\n",
            "print(f\"{'='*60}\")\n",
            "print(f\"  Initial Loss    : {history['loss'][0]:.6f}\")\n",
            "print(f\"  Final Loss      : {history['loss'][-1]:.6f}\")\n",
            "print(f\"  Min Loss        : {min(history['loss']):.6f} (Epoch {history['loss'].index(min(history['loss'])) + 1})\")\n",
            "print(f\"  Improvement     : {((history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100):.1f}%\")\n",
            "print(f\"  Total Epochs    : {len(history['loss'])}\")\n",
            "if 'learning_rate' in history:\n",
            "    print(f\"  Final LR        : {history['learning_rate'][-1]:.2e}\")\n",
            "print(f\"{'='*60}\")"
        ]
    })
    
    # =========================================================================
    # Cell 22: Cleanup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Cleanup"]
    })
    
    # =========================================================================
    # Cell 23: W&B Finish Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Finish W&B run\n",
            "if wandb.run is not None:\n",
            "    wandb.finish()\n",
            "    print(\"W&B run finished successfully.\")"
        ]
    })
    
    # =========================================================================
    # Cell 24: Kernel Restart Code (commented out)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# CLEANUP: Restart kernel to fully release GPU memory\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# TensorFlow/CUDA does not release GPU memory within a running Python process.\n",
            "# Restarting the kernel is the only guaranteed way to free all GPU resources.\n",
            "#\n",
            "# NOTE: Only run this cell after all work is complete and saved.\n",
            "#       The kernel restart will clear all variables and outputs.\n",
            "\n",
            "# import IPython\n",
            "# print(\"Restarting kernel to release GPU memory...\")\n",
            "# IPython.Application.instance().kernel.do_shutdown(restart=True)"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Main function to run the standardization."""
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'v1', 'notebooks',
        '03_01_autoencoder_train.ipynb'
    )
    notebook_path = os.path.abspath(notebook_path)
    
    print(f"Standardizing notebook: {notebook_path}")
    
    # Create standardized notebook
    notebook = create_standardized_notebook()
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n{'='*60}")
    print("STANDARDIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cells: {len(notebook['cells'])}")
    print("\nStandards applied:")
    print("  ✅ GPU memory growth cell (first code cell)")
    print("  ✅ Clean imports with gpu_utils")
    print("  ✅ Dynamic batch/epoch scaling using 'ae' profile")
    print("  ✅ Model uses configuration variables")
    print("  ✅ LRFinder for optimal learning rate")
    print("  ✅ Full callback stack")
    print("  ✅ Training visualization with semilogy LR")
    print("  ✅ Kernel restart cell (commented out)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
