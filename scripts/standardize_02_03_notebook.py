#!/usr/bin/env python3
"""
Script to standardize 02_03_deep_learning_conv_neural_network.ipynb.

This CNN training notebook needs all 7 project standards applied.

Usage:
    uv run python scripts/standardize_02_03_notebook.py
"""

import json
import os


def create_standardized_notebook():
    """Create a fully standardized CNN notebook."""
    
    cells = []
    
    # =========================================================================
    # Cell 0: Header Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Your First Convolutional Neural Network\n",
            "\n",
            "This notebook implements a CNN on the CIFAR-10 dataset with BatchNormalization,\n",
            "LeakyReLU activations, and Dropout for regularization.\n",
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
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# TensorFlow/Keras imports\n",
            "from keras.layers import (\n",
            "    Input, Flatten, Dense, Conv2D,\n",
            "    BatchNormalization, LeakyReLU, Dropout, Activation\n",
            ")\n",
            "from keras.models import Model\n",
            "from keras.optimizers import Adam\n",
            "from keras.utils import to_categorical\n",
            "from keras.datasets import cifar10\n",
            "\n",
            "# Path setup for project utilities\n",
            "import sys\n",
            "sys.path.insert(0, '../..')    # For project root utils/\n",
            "\n",
            "# Project utilities\n",
            "from utils.wandb_utils import init_wandb, get_metrics_logger\n",
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
            "REFERENCE_EPOCHS = 10       # Original book value\n",
            "\n",
            "# Auto-detect GPU VRAM or override manually\n",
            "TARGET_VRAM_GB = None  # Set to 6, 8, 12, etc. to override detection\n",
            "GPU_VRAM_GB = TARGET_VRAM_GB if TARGET_VRAM_GB else get_gpu_vram_gb()\n",
            "\n",
            "# Calculate optimal settings dynamically based on GPU VRAM\n",
            "# Uses 'cifar10' profile for CIFAR-10 classification networks\n",
            "BATCH_SIZE = get_optimal_batch_size('cifar10', vram_gb=GPU_VRAM_GB)\n",
            "EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)\n",
            "\n",
            "# Ensure minimum epochs for meaningful training\n",
            "EPOCHS = max(EPOCHS, 10)\n",
            "\n",
            "# Learning rate (will be set by LRFinder)\n",
            "LEARNING_RATE = \"auto\"\n",
            "\n",
            "# Model configuration\n",
            "NUM_CLASSES = 10\n",
            "CONV_FILTERS = [32, 32, 64, 64]  # Filters for each conv layer\n",
            "DENSE_UNITS = 128                 # Dense layer size\n",
            "DROPOUT_RATE = 0.5\n",
            "\n",
            "# W&B configuration\n",
            "MODEL_TYPE = \"cnn\"\n",
            "DATASET_NAME = \"cifar10\"\n",
            "OPTIMIZER_NAME = \"adam\"\n",
            "LAYERS_DESC = \"4_conv_2_dense\"\n",
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
        "source": ["## Load and Preprocess Data"]
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
            "# Load CIFAR-10 dataset\n",
            "(x_train, y_train), (x_test, y_test) = cifar10.load_data()\n",
            "\n",
            "# Normalize pixel values to [0, 1]\n",
            "x_train = x_train.astype('float32') / 255.0\n",
            "x_test = x_test.astype('float32') / 255.0\n",
            "\n",
            "# One-hot encode labels\n",
            "y_train = to_categorical(y_train, NUM_CLASSES)\n",
            "y_test = to_categorical(y_test, NUM_CLASSES)\n",
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
            "    name=\"02_03_cnn\",\n",
            "    config={\n",
            "        \"model\": MODEL_TYPE,\n",
            "        \"dataset\": DATASET_NAME,\n",
            "        \"layers\": LAYERS_DESC,\n",
            "        \"conv_filters\": CONV_FILTERS,\n",
            "        \"dense_units\": DENSE_UNITS,\n",
            "        \"dropout_rate\": DROPOUT_RATE,\n",
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
        "source": [
            "## Model Architecture\n",
            "\n",
            "CNN with 4 convolutional layers, BatchNormalization, LeakyReLU, and Dropout."
        ]
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
            "# Build the CNN model using Keras Functional API\n",
            "input_layer = Input((32, 32, 3))\n",
            "\n",
            "# Conv Block 1: 32 filters, stride 1\n",
            "x = Conv2D(filters=CONV_FILTERS[0], kernel_size=3, strides=1, padding='same')(input_layer)\n",
            "x = BatchNormalization()(x)\n",
            "x = LeakyReLU()(x)\n",
            "\n",
            "# Conv Block 2: 32 filters, stride 2 (downsampling)\n",
            "x = Conv2D(filters=CONV_FILTERS[1], kernel_size=3, strides=2, padding='same')(x)\n",
            "x = BatchNormalization()(x)\n",
            "x = LeakyReLU()(x)\n",
            "\n",
            "# Conv Block 3: 64 filters, stride 1\n",
            "x = Conv2D(filters=CONV_FILTERS[2], kernel_size=3, strides=1, padding='same')(x)\n",
            "x = BatchNormalization()(x)\n",
            "x = LeakyReLU()(x)\n",
            "\n",
            "# Conv Block 4: 64 filters, stride 2 (downsampling)\n",
            "x = Conv2D(filters=CONV_FILTERS[3], kernel_size=3, strides=2, padding='same')(x)\n",
            "x = BatchNormalization()(x)\n",
            "x = LeakyReLU()(x)\n",
            "\n",
            "# Flatten and Dense layers\n",
            "x = Flatten()(x)\n",
            "x = Dense(DENSE_UNITS)(x)\n",
            "x = BatchNormalization()(x)\n",
            "x = LeakyReLU()(x)\n",
            "x = Dropout(rate=DROPOUT_RATE)(x)\n",
            "\n",
            "# Output layer\n",
            "x = Dense(NUM_CLASSES)(x)\n",
            "output_layer = Activation('softmax')(x)\n",
            "\n",
            "model = Model(input_layer, output_layer)\n",
            "model.summary()"
        ]
    })
    
    # =========================================================================
    # Cell 13: LRFinder Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Find Optimal Learning Rate"]
    })
    
    # =========================================================================
    # Cell 14: LRFinder Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clone model for LR finding (to avoid affecting the main model)\n",
            "lr_model = tf.keras.models.clone_model(model)\n",
            "\n",
            "# Compile with a very small initial learning rate\n",
            "lr_model.compile(\n",
            "    loss='categorical_crossentropy',\n",
            "    optimizer=Adam(learning_rate=1e-6),\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "# Run LRFinder\n",
            "lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)\n",
            "lr_model.fit(\n",
            "    x_train, y_train,\n",
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
            "# Update W&B config with optimal learning rate\n",
            "wandb.config.update({\"learning_rate\": LEARNING_RATE})\n",
            "print(f\"\\nOptimal learning rate: {LEARNING_RATE:.2e}\")"
        ]
    })
    
    # =========================================================================
    # Cell 15: Train Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Train Model"]
    })
    
    # =========================================================================
    # Cell 16: Train Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compile model with optimal learning rate\n",
            "model.compile(\n",
            "    loss='categorical_crossentropy',\n",
            "    optimizer=Adam(learning_rate=LEARNING_RATE),\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "# Define full callback stack\n",
            "callbacks = [\n",
            "    WandbMetricsLogger(),                              # W&B logging\n",
            "    get_lr_scheduler(monitor='val_loss', patience=2),  # Reduce LR on plateau\n",
            "    get_early_stopping(monitor='val_loss', patience=5),# Stop if no improvement\n",
            "    LRLogger(),                                        # Log learning rate\n",
            "]\n",
            "\n",
            "# Train the model\n",
            "history = model.fit(\n",
            "    x_train,\n",
            "    y_train,\n",
            "    batch_size=BATCH_SIZE,\n",
            "    epochs=EPOCHS,\n",
            "    shuffle=True,\n",
            "    validation_data=(x_test, y_test),\n",
            "    callbacks=callbacks\n",
            ")"
        ]
    })
    
    # =========================================================================
    # Cell 17: Training Visualization Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Training Visualization"]
    })
    
    # =========================================================================
    # Cell 18: Training Visualization Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot training history\n",
            "history_dict = history.history\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "# Plot 1: Training & Validation Loss\n",
            "axes[0].plot(history_dict['loss'], 'b-', linewidth=2, label='Train')\n",
            "axes[0].plot(history_dict['val_loss'], 'r-', linewidth=2, label='Validation')\n",
            "axes[0].set_xlabel('Epoch')\n",
            "axes[0].set_ylabel('Loss')\n",
            "axes[0].set_title('Loss Over Epochs')\n",
            "axes[0].legend()\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 2: Training & Validation Accuracy\n",
            "axes[1].plot(history_dict['accuracy'], 'b-', linewidth=2, label='Train')\n",
            "axes[1].plot(history_dict['val_accuracy'], 'r-', linewidth=2, label='Validation')\n",
            "axes[1].set_xlabel('Epoch')\n",
            "axes[1].set_ylabel('Accuracy')\n",
            "axes[1].set_title('Accuracy Over Epochs')\n",
            "axes[1].legend()\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 3: Learning Rate (LOG SCALE)\n",
            "if 'learning_rate' in history_dict:\n",
            "    axes[2].semilogy(history_dict['learning_rate'], 'g-', linewidth=2)\n",
            "    axes[2].set_xlabel('Epoch')\n",
            "    axes[2].set_ylabel('Learning Rate (log scale)')\n",
            "    axes[2].set_title('Learning Rate Schedule')\n",
            "    axes[2].grid(True, which='both', alpha=0.3)\n",
            "else:\n",
            "    axes[2].text(0.5, 0.5, 'LR not tracked', ha='center', va='center', fontsize=14)\n",
            "    axes[2].set_title('Learning Rate (Not Available)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Print summary\n",
            "print(f\"\\n{'='*60}\")\n",
            "print(\"TRAINING SUMMARY\")\n",
            "print(f\"{'='*60}\")\n",
            "print(f\"  Initial Loss    : {history_dict['loss'][0]:.6f}\")\n",
            "print(f\"  Final Loss      : {history_dict['loss'][-1]:.6f}\")\n",
            "print(f\"  Min Loss        : {min(history_dict['loss']):.6f} (Epoch {history_dict['loss'].index(min(history_dict['loss'])) + 1})\")\n",
            "print(f\"  Final Accuracy  : {history_dict['accuracy'][-1]:.4f}\")\n",
            "print(f\"  Final Val Acc   : {history_dict['val_accuracy'][-1]:.4f}\")\n",
            "print(f\"  Total Epochs    : {len(history_dict['loss'])}\")\n",
            "if 'learning_rate' in history_dict:\n",
            "    print(f\"  Final LR        : {history_dict['learning_rate'][-1]:.2e}\")\n",
            "print(f\"{'='*60}\")"
        ]
    })
    
    # =========================================================================
    # Cell 19: Evaluation Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model Evaluation"]
    })
    
    # =========================================================================
    # Cell 20: Evaluation Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate on test set\n",
            "test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)\n",
            "print(f\"\\nTest accuracy: {test_acc:.4f}\")"
        ]
    })
    
    # =========================================================================
    # Cell 21: Analysis Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Analysis"]
    })
    
    # =========================================================================
    # Cell 22: Predictions Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Class names for CIFAR-10\n",
            "CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',\n",
            "                    'dog', 'frog', 'horse', 'ship', 'truck'])\n",
            "\n",
            "# Make predictions\n",
            "preds = model.predict(x_test)\n",
            "preds_single = CLASSES[np.argmax(preds, axis=-1)]\n",
            "actual_single = CLASSES[np.argmax(y_test, axis=-1)]"
        ]
    })
    
    # =========================================================================
    # Cell 23: Visualization Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize some predictions\n",
            "n_to_show = 10\n",
            "indices = np.random.choice(range(len(x_test)), n_to_show)\n",
            "\n",
            "fig = plt.figure(figsize=(15, 3))\n",
            "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
            "\n",
            "for i, idx in enumerate(indices):\n",
            "    img = x_test[idx]\n",
            "    ax = fig.add_subplot(1, n_to_show, i + 1)\n",
            "    ax.axis('off')\n",
            "    ax.text(0.5, -0.35, f'pred: {preds_single[idx]}', fontsize=10, ha='center', transform=ax.transAxes)\n",
            "    ax.text(0.5, -0.7, f'actual: {actual_single[idx]}', fontsize=10, ha='center', transform=ax.transAxes)\n",
            "    ax.imshow(img)\n",
            "\n",
            "plt.show()"
        ]
    })
    
    # =========================================================================
    # Cell 24: Cleanup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Cleanup"]
    })
    
    # =========================================================================
    # Cell 25: W&B Finish Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Finish W&B run\n",
            "wandb.finish()\n",
            "print(\"W&B run finished successfully.\")"
        ]
    })
    
    # =========================================================================
    # Cell 26: Kernel Restart Code (commented out)
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
        '02_03_deep_learning_conv_neural_network.ipynb'
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
    print("  ✅ Clean imports with all required modules")
    print("  ✅ Dynamic batch/epoch scaling from gpu_utils")
    print("  ✅ Proper section headers (## Title Case)")
    print("  ✅ Model uses configuration variables")
    print("  ✅ LRFinder for optimal learning rate")
    print("  ✅ Full callback stack (WandbMetricsLogger, LRScheduler, EarlyStopping, LRLogger)")
    print("  ✅ Training visualization with semilogy LR")
    print("  ✅ Kernel restart cell (commented out)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
