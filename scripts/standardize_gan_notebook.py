#!/usr/bin/env python3
"""
GAN Notebook Standardization Script.

This script generates a fully standardized version of the GAN training notebook
(04_01_gan_camel_train.ipynb) with:
- PEP 8 compliant code formatting
- Comprehensive documentation and comments
- Dynamic batch size and epoch scaling based on GPU VRAM
- W&B integration for experiment tracking
- Step decay learning rate scheduler
- Enhanced post-training visualizations

Usage:
    uv run python v1/scripts/standardize_gan_notebook.py
    
Output:
    v1/notebooks/04_01_gan_camel_train.ipynb (overwrites existing)

Author: Antigravity AI
Created: 2026-01-02
"""

import json
import os

# =============================================================================
# NOTEBOOK CELL DEFINITIONS
# =============================================================================
# Each cell is defined as a dictionary with 'type' and 'source' keys.
# Markdown cells use 'markdown', code cells use 'code'.


def create_markdown_cell(source: str) -> dict:
    """Create a markdown cell for the notebook."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n')
    }


def create_code_cell(source: str) -> dict:
    """Create a code cell for the notebook."""
    # Split source into lines, preserving line endings
    lines = source.split('\n')
    # Add newlines to all but the last line
    source_lines = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # Add last line without newline if not empty
        source_lines.append(lines[-1])
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }


# =============================================================================
# CELL DEFINITIONS
# =============================================================================

CELLS = []

# -----------------------------------------------------------------------------
# CELL 0: Title
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""# GAN Training - Camel Dataset

This notebook trains a Generative Adversarial Network (GAN) on the Camel
dataset (Quick, Draw!) to generate hand-drawn camel sketches.

## Features

- **Dynamic Configuration**: Batch size and epochs adjust based on GPU VRAM
- **W&B Integration**: Full experiment tracking with Weights & Biases
- **LR Scheduling**: Step decay learning rate for stable training
- **Enhanced Visualization**: Loss, accuracy, and LR history plots

## Architecture

- **Discriminator**: 4-layer CNN with strided convolutions
- **Generator**: 4-layer deconvolution network with upsampling

## References

- Goodfellow et al. "Generative Adversarial Networks" (2014)
- Chapter 4 of "Generative Deep Learning" book"""))

# -----------------------------------------------------------------------------
# CELL 1: GPU Setup
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## GPU Memory Setup

Configure TensorFlow to use memory growth, preventing OOM errors."""))

CELLS.append(create_code_cell("""# =============================================================================
# GPU MEMORY CONFIGURATION
# =============================================================================
# Enable memory growth to prevent TensorFlow from allocating all GPU memory
# at once. This must be done BEFORE any other TensorFlow operations.

import tensorflow as tf

# Get list of available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Enable memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("⚠ WARNING: No GPU detected, running on CPU")"""))

# -----------------------------------------------------------------------------
# CELL 2: Imports
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Imports

Import all required modules including project utilities."""))

CELLS.append(create_code_cell("""# =============================================================================
# IMPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
# Add parent directory to path for importing project modules
# This allows imports from both v1/src and project root utils/
import sys
sys.path.insert(0, '..')      # For v1/src modules
sys.path.insert(0, '../..')   # For project root utils/

# -----------------------------------------------------------------------------
# Standard Library
# -----------------------------------------------------------------------------
import os

# -----------------------------------------------------------------------------
# Third-Party Libraries
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import wandb

# -----------------------------------------------------------------------------
# Project Modules
# -----------------------------------------------------------------------------
from src.models.GAN import GAN
from src.utils.loaders import load_safari

# GPU utilities for dynamic batch size and epoch scaling
# Located in project root: utils/gpu_utils.py
from utils.gpu_utils import (
    get_optimal_batch_size,
    calculate_adjusted_epochs,
    get_gpu_vram_gb,
    print_training_config
)"""))

# -----------------------------------------------------------------------------
# CELL 3: Global Configuration
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Global Configuration

Central configuration cell with all training hyperparameters.
Batch size and epochs are automatically optimized based on GPU VRAM."""))

CELLS.append(create_code_cell("""# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Run Identification
# -----------------------------------------------------------------------------
SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'camel'
RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'

# -----------------------------------------------------------------------------
# Create Run Directories
# -----------------------------------------------------------------------------
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))

# -----------------------------------------------------------------------------
# Reference Training Configuration
# These are the original notebook values used as baseline
# -----------------------------------------------------------------------------
REFERENCE_BATCH_SIZE = 256
REFERENCE_EPOCHS = 6000

# -----------------------------------------------------------------------------
# Dynamic Training Configuration
# Automatically optimized based on available GPU VRAM
# -----------------------------------------------------------------------------

# VRAM Override: Set to specific value (e.g., 8) or None for auto-detection
# Use this when you want to force a specific configuration
TARGET_VRAM_GB = None  # Options: None, 6, 8, 12, 16, 24

# Detect GPU VRAM or use manual override
GPU_VRAM_GB = TARGET_VRAM_GB if TARGET_VRAM_GB else get_gpu_vram_gb()

# Calculate optimal batch size for detected VRAM
# Larger VRAM = larger batch size = faster training
BATCH_SIZE = get_optimal_batch_size('gan', vram_gb=GPU_VRAM_GB)

# Scale epochs to maintain equivalent total training updates
# Formula: reference_epochs × (reference_batch / actual_batch)
EPOCHS = calculate_adjusted_epochs(
    REFERENCE_EPOCHS,
    REFERENCE_BATCH_SIZE,
    BATCH_SIZE
)

# Adjust checkpoint frequency proportionally
PRINT_EVERY_N_BATCHES = max(50 * REFERENCE_BATCH_SIZE // BATCH_SIZE, 10)

# -----------------------------------------------------------------------------
# Learning Rate Scheduler Configuration
# Step decay: Reduce LR at fixed intervals for stable late-stage training
# -----------------------------------------------------------------------------
LR_DECAY_FACTOR = 0.5  # Multiply LR by this factor at each decay point
LR_DECAY_EPOCHS = EPOCHS // 4  # Decay 4 times during training

# -----------------------------------------------------------------------------
# Training Mode
# -----------------------------------------------------------------------------
MODE = 'build'  # Options: 'build' (new training), 'load' (resume from weights)

# -----------------------------------------------------------------------------
# Print Configuration Summary
# -----------------------------------------------------------------------------
print_training_config(
    'gan',
    BATCH_SIZE,
    EPOCHS,
    REFERENCE_BATCH_SIZE,
    REFERENCE_EPOCHS,
    GPU_VRAM_GB
)
print(f"LR Decay: ×{LR_DECAY_FACTOR} every {LR_DECAY_EPOCHS} epochs")
print(f"Checkpoints: Every {PRINT_EVERY_N_BATCHES} epochs")"""))

# -----------------------------------------------------------------------------
# CELL 4: Data Loading
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Data Loading

Load the Camel dataset from Quick, Draw! collection."""))

CELLS.append(create_code_cell("""# =============================================================================
# DATA LOADING
# =============================================================================
# Load the camel dataset from the Quick, Draw! collection
# Images are 28x28 grayscale drawings

(x_train, y_train) = load_safari(DATA_NAME)

# Print dataset information
print(f"Dataset: {DATA_NAME}")
print(f"Training samples: {x_train.shape[0]:,}")
print(f"Image dimensions: {x_train.shape[1:]}")
print(f"Data type: {x_train.dtype}")
print(f"Value range: [{x_train.min():.2f}, {x_train.max():.2f}]")"""))

# -----------------------------------------------------------------------------
# CELL 5: Sample Visualization
# -----------------------------------------------------------------------------
CELLS.append(create_code_cell("""# =============================================================================
# SAMPLE VISUALIZATION
# =============================================================================
# Display a sample image from the dataset

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
fig.suptitle('Sample Camel Drawings', fontsize=14)

for i, ax in enumerate(axes):
    idx = np.random.randint(0, x_train.shape[0])
    ax.imshow(x_train[idx, :, :, 0], cmap='gray_r')
    ax.axis('off')
    ax.set_title(f'Sample {idx}')

plt.tight_layout()
plt.show()"""))

# -----------------------------------------------------------------------------
# CELL 6: Model Architecture
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Model Architecture

Build the GAN with discriminator and generator networks."""))

CELLS.append(create_code_cell("""# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Build the GAN with configurable discriminator and generator
#
# Discriminator Architecture:
#   Input (28, 28, 1) → Conv layers → Flatten → Dense(1, sigmoid)
#
# Generator Architecture:
#   Input (z_dim,) → Dense → Reshape → ConvTranspose layers → Output (28, 28, 1)

gan = GAN(
    # -------------------------------------------------------------------------
    # Input Configuration
    # -------------------------------------------------------------------------
    input_dim=(28, 28, 1),  # 28x28 grayscale images
    z_dim=100,              # Latent space dimension
    
    # -------------------------------------------------------------------------
    # Discriminator Configuration
    # 4 convolutional layers with increasing filters
    # -------------------------------------------------------------------------
    discriminator_conv_filters=[64, 64, 128, 128],
    discriminator_conv_kernel_size=[5, 5, 5, 5],
    discriminator_conv_strides=[2, 2, 2, 1],
    discriminator_batch_norm_momentum=None,  # No batch norm in discriminator
    discriminator_activation='relu',
    discriminator_dropout_rate=0.4,
    discriminator_learning_rate=0.0008,
    
    # -------------------------------------------------------------------------
    # Generator Configuration
    # 4 upsampling layers to generate 28x28 output
    # -------------------------------------------------------------------------
    generator_initial_dense_layer_size=(7, 7, 64),  # Start from 7x7
    generator_upsample=[2, 2, 1, 1],  # Upsample to 14x14, then 28x28
    generator_conv_filters=[128, 64, 64, 1],
    generator_conv_kernel_size=[5, 5, 5, 5],
    generator_conv_strides=[1, 1, 1, 1],
    generator_batch_norm_momentum=0.9,
    generator_activation='relu',
    generator_dropout_rate=None,  # No dropout in generator
    generator_learning_rate=0.0004,
    
    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    optimiser='rmsprop'  # RMSprop works well for GANs
)

print("✓ GAN model built successfully")"""))

# -----------------------------------------------------------------------------
# CELL 7: Model Summary
# -----------------------------------------------------------------------------
CELLS.append(create_code_cell("""# =============================================================================
# DISCRIMINATOR ARCHITECTURE
# =============================================================================
gan.discriminator.summary()"""))

CELLS.append(create_code_cell("""# =============================================================================
# GENERATOR ARCHITECTURE
# =============================================================================
gan.generator.summary()"""))

# -----------------------------------------------------------------------------
# CELL 8: W&B Initialization
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## W&B Initialization

Initialize Weights & Biases for experiment tracking."""))

CELLS.append(create_code_cell("""# =============================================================================
# W&B INITIALIZATION
# =============================================================================
# Initialize Weights & Biases for experiment tracking
# This logs all training metrics, sample images, and hyperparameters

wandb.init(
    project="generative-deep-learning",
    name=f"gan-{DATA_NAME}-bs{BATCH_SIZE}",
    config={
        # Model Configuration
        "model": "GAN",
        "dataset": DATA_NAME,
        "input_dim": gan.input_dim,
        "z_dim": gan.z_dim,
        
        # Training Configuration
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "reference_epochs": REFERENCE_EPOCHS,
        
        # Optimizer Configuration
        "discriminator_lr": gan.discriminator_learning_rate,
        "generator_lr": gan.generator_learning_rate,
        "optimizer": gan.optimiser,
        
        # LR Scheduler Configuration
        "lr_decay_factor": LR_DECAY_FACTOR,
        "lr_decay_epochs": LR_DECAY_EPOCHS,
        
        # Environment
        "gpu_vram_gb": GPU_VRAM_GB,
    }
)

print("✓ W&B initialized")
print(f"  Project: generative-deep-learning")
print(f"  Run: gan-{DATA_NAME}-bs{BATCH_SIZE}")
print(f"  URL: {wandb.run.url}")"""))

# -----------------------------------------------------------------------------
# CELL 9: Training
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Training

Train the GAN with W&B logging and LR scheduling enabled."""))

CELLS.append(create_code_cell("""# =============================================================================
# TRAINING
# =============================================================================
# Train the GAN with:
# - W&B logging for all metrics
# - Step decay LR scheduler
# - Periodic weight saving and sample generation

gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    use_wandb=True,           # Enable W&B logging
    lr_decay_factor=LR_DECAY_FACTOR,
    lr_decay_epochs=LR_DECAY_EPOCHS
)"""))

# -----------------------------------------------------------------------------
# CELL 10: Training Visualization
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Training Visualization

Generate comprehensive training plots showing loss, accuracy, and LR history."""))

CELLS.append(create_code_cell("""# =============================================================================
# TRAINING VISUALIZATION
# =============================================================================
# Creates a 3-panel figure showing:
# 1. Discriminator/Generator Loss vs Epoch
# 2. Discriminator/Generator Accuracy vs Epoch
# 3. Learning Rate Schedule (log scale)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -----------------------------------------------------------------------------
# Plot 1: D/G Loss vs Epoch
# -----------------------------------------------------------------------------
ax1 = axes[0]
ax1.plot(
    [x[0] for x in gan.d_losses],
    color='blue', linewidth=0.5, label='D Total'
)
ax1.plot(
    [x[1] for x in gan.d_losses],
    color='green', linewidth=0.25, alpha=0.6, label='D Real'
)
ax1.plot(
    [x[2] for x in gan.d_losses],
    color='red', linewidth=0.25, alpha=0.6, label='D Fake'
)
ax1.plot(
    [x[0] for x in gan.g_losses],
    color='orange', linewidth=0.5, label='G'
)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Discriminator / Generator Loss', fontsize=14)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 2)
ax1.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot 2: D/G Accuracy vs Epoch
# -----------------------------------------------------------------------------
ax2 = axes[1]
ax2.plot(
    [x[3] for x in gan.d_losses],
    color='blue', linewidth=0.5, label='D Total'
)
ax2.plot(
    [x[4] for x in gan.d_losses],
    color='green', linewidth=0.25, alpha=0.6, label='D Real'
)
ax2.plot(
    [x[5] for x in gan.d_losses],
    color='red', linewidth=0.25, alpha=0.6, label='D Fake'
)
ax2.plot(
    [x[1] for x in gan.g_losses],
    color='orange', linewidth=0.5, label='G'
)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Discriminator / Generator Accuracy', fontsize=14)
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot 3: Learning Rate vs Epoch (Log Scale)
# -----------------------------------------------------------------------------
ax3 = axes[2]
ax3.semilogy(
    gan.d_lr_history,
    color='blue', linewidth=1.5, label='Discriminator LR'
)
ax3.semilogy(
    gan.g_lr_history,
    color='orange', linewidth=1.5, label='Generator LR'
)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Finalize and Save
# -----------------------------------------------------------------------------
plt.tight_layout()
plt.savefig(
    os.path.join(RUN_FOLDER, 'training_summary.png'),
    dpi=200,
    bbox_inches='tight'
)
plt.show()

print(f"✓ Training summary saved to {RUN_FOLDER}/training_summary.png")"""))

# -----------------------------------------------------------------------------
# CELL 11: Training Summary
# -----------------------------------------------------------------------------
CELLS.append(create_code_cell("""# =============================================================================
# TRAINING SUMMARY
# =============================================================================
# Print final training metrics

print(f"\\n{'═' * 60}")
print("TRAINING COMPLETE")
print(f"{'═' * 60}")
print(f"  Epochs trained  : {gan.epoch}")
print(f"  Final D loss    : {gan.d_losses[-1][0]:.4f}")
print(f"  Final G loss    : {gan.g_losses[-1][0]:.4f}")
print(f"  Final D accuracy: {gan.d_losses[-1][3]:.4f}")
print(f"  Final G accuracy: {gan.g_losses[-1][1]:.4f}")
print(f"  Final D LR      : {gan.d_lr_history[-1]:.2e}")
print(f"  Final G LR      : {gan.g_lr_history[-1]:.2e}")
print(f"  Weights saved   : {RUN_FOLDER}/weights/")
print(f"{'═' * 60}")"""))

# -----------------------------------------------------------------------------
# CELL 12: W&B Cleanup
# -----------------------------------------------------------------------------
CELLS.append(create_markdown_cell("""## Cleanup

Finish W&B run and optionally restart kernel to release GPU memory."""))

CELLS.append(create_code_cell("""# =============================================================================
# W&B CLEANUP
# =============================================================================
# Finish the W&B run to ensure all data is synced

wandb.finish()
print("✓ W&B run finished and synced")"""))

# -----------------------------------------------------------------------------
# CELL 13: Kernel Restart
# -----------------------------------------------------------------------------
CELLS.append(create_code_cell("""# =============================================================================
# CLEANUP: Restart Kernel to Release GPU Memory
# =============================================================================
# TensorFlow/CUDA does not release GPU memory within a running Python process.
# Restarting the kernel is the only guaranteed way to free all GPU resources.
#
# ⚠️ WARNING: This will clear all variables and outputs!
# Only run after all work is complete and saved.

import IPython
print("Restarting kernel to release GPU memory...")
IPython.Application.instance().kernel.do_shutdown(restart=True)"""))


# =============================================================================
# NOTEBOOK GENERATION
# =============================================================================

def create_notebook() -> dict:
    """Create the complete notebook structure."""
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def main():
    """Generate the standardized notebook."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(script_dir, '..', 'v1', 'notebooks')
    output_path = os.path.join(notebook_dir, '04_01_gan_camel_train.ipynb')
    
    # Create notebook
    notebook = create_notebook()
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("=" * 60)
    print("GAN NOTEBOOK STANDARDIZATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total cells: {len(CELLS)}")
    print()
    print("Features:")
    print("  ✓ PEP 8 compliant code formatting")
    print("  ✓ Comprehensive documentation and comments")
    print("  ✓ Dynamic batch size and epoch scaling")
    print("  ✓ W&B integration for experiment tracking")
    print("  ✓ Step decay LR scheduler")
    print("  ✓ Enhanced training visualizations")
    print("  ✓ Kernel restart cell for GPU memory release")
    print("=" * 60)


if __name__ == "__main__":
    main()
