"""
GAN Notebook Cell Templates
============================

This file contains the cell templates for standardizing the GAN training notebook
(v1/notebooks/04_01_gan_camel_train.ipynb).

Copy each section to the corresponding cell in the notebook.

Author: Antigravity AI
Created: 2026-01-02
"""

# =============================================================================
# CELL 1: GPU MEMORY SETUP (FIRST CELL)
# =============================================================================
# Description: Configure TensorFlow to use memory growth for GPU. This prevents
# TensorFlow from allocating all GPU memory at once, which helps avoid OOM errors.
# Place this BEFORE any other TensorFlow imports.

CELL_1_GPU_SETUP = '''
import tensorflow as tf

# Enable memory growth to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("⚠ WARNING: No GPU detected, running on CPU")
'''


# =============================================================================
# CELL 2: IMPORTS
# =============================================================================
# Description: Import all required modules including the new gpu_utils module.

CELL_2_IMPORTS = '''
import sys
sys.path.insert(0, '..')

import os
import wandb
import matplotlib.pyplot as plt
import numpy as np

from src.models.GAN import GAN
from src.utils.loaders import load_safari
from utils.gpu_utils import (
    get_optimal_batch_size, 
    calculate_adjusted_epochs, 
    get_gpu_vram_gb,
    print_training_config
)
'''


# =============================================================================
# CELL 3: GLOBAL CONFIGURATION
# =============================================================================
# Description: Central configuration cell with all training hyperparameters.
# Dynamic batch size and epoch scaling based on available GPU VRAM.

CELL_3_GLOBAL_CONFIG = '''
# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Run identification
SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'camel'
RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'

# Create run directories if they don't exist
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))

# ───────────────────────────────────────────────────────────────────────────────
# DYNAMIC TRAINING CONFIGURATION
# These values are automatically optimized based on your GPU VRAM
# ───────────────────────────────────────────────────────────────────────────────

# Reference training config (original notebook values)
REFERENCE_BATCH_SIZE = 256
REFERENCE_EPOCHS = 6000

# VRAM override: Set to specific value (e.g., 8) or None for auto-detection
TARGET_VRAM_GB = None  # Change to 6, 8, 12, etc. to override auto-detection

# Auto-detect or use override
GPU_VRAM_GB = TARGET_VRAM_GB if TARGET_VRAM_GB else get_gpu_vram_gb()

# Calculate optimal batch size and scaled epochs
BATCH_SIZE = get_optimal_batch_size('gan', vram_gb=GPU_VRAM_GB)
EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)

# Adjust print frequency to maintain similar number of checkpoints
PRINT_EVERY_N_BATCHES = max(50 * REFERENCE_BATCH_SIZE // BATCH_SIZE, 10)

# ───────────────────────────────────────────────────────────────────────────────
# LR SCHEDULER CONFIGURATION
# Step decay: Reduce LR at fixed intervals for stable late-stage training
# ───────────────────────────────────────────────────────────────────────────────

LR_DECAY_FACTOR = 0.5  # Multiply LR by this factor at each decay point
LR_DECAY_EPOCHS = EPOCHS // 4  # Decay 4 times during training (0%, 25%, 50%, 75%)

# ───────────────────────────────────────────────────────────────────────────────
# TRAINING MODE
# ───────────────────────────────────────────────────────────────────────────────

mode = 'build'  # 'build' for new training, 'load' to resume from weights

# Print configuration summary
print_training_config(
    'gan', BATCH_SIZE, EPOCHS, 
    REFERENCE_BATCH_SIZE, REFERENCE_EPOCHS, 
    GPU_VRAM_GB
)
print(f"LR Decay: ×{LR_DECAY_FACTOR} every {LR_DECAY_EPOCHS} epochs")
print(f"Checkpoints: Every {PRINT_EVERY_N_BATCHES} epochs")
'''


# =============================================================================
# CELL: W&B INITIALIZATION (Before Training)
# =============================================================================
# Description: Initialize Weights & Biases experiment tracking.
# Place this after model creation and before training.

CELL_WANDB_INIT = '''
# ═══════════════════════════════════════════════════════════════════════════════
# W&B INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

wandb.init(
    project="generative-deep-learning",
    name=f"gan-{DATA_NAME}-bs{BATCH_SIZE}",
    config={
        # Model configuration
        "model": "GAN",
        "dataset": DATA_NAME,
        "input_dim": gan.input_dim,
        "z_dim": gan.z_dim,
        
        # Training configuration
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "reference_epochs": REFERENCE_EPOCHS,
        
        # Optimizer configuration
        "discriminator_lr": gan.discriminator_learning_rate,
        "generator_lr": gan.generator_learning_rate,
        "optimizer": gan.optimiser,
        
        # LR scheduler configuration
        "lr_decay_factor": LR_DECAY_FACTOR,
        "lr_decay_epochs": LR_DECAY_EPOCHS,
        
        # Environment
        "gpu_vram_gb": GPU_VRAM_GB,
    }
)

print("✓ W&B initialized - tracking experiment at: " + wandb.run.url)
'''


# =============================================================================
# CELL: TRAINING (Modified)
# =============================================================================
# Description: Train the GAN with W&B logging and LR scheduling enabled.

CELL_TRAINING = '''
# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    use_wandb=True,  # Enable W&B logging
    lr_decay_factor=LR_DECAY_FACTOR,
    lr_decay_epochs=LR_DECAY_EPOCHS
)
'''


# =============================================================================
# CELL: ENHANCED TRAINING PLOTS (After Training)
# =============================================================================
# Description: Generate comprehensive training visualization with 3 subplots:
# 1. D/G Loss over epochs
# 2. D/G Accuracy over epochs
# 3. Learning Rate schedule (new)

CELL_TRAINING_PLOTS = '''
# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING VISUALIZATION
# Creates a 3-panel figure showing loss, accuracy, and learning rate history
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ───────────────────────────────────────────────────────────────────────────────
# Plot 1: D/G Loss vs Epoch
# ───────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot([x[0] for x in gan.d_losses], color='blue', linewidth=0.5, label='D Total')
ax1.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25, alpha=0.6, label='D Real')
ax1.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25, alpha=0.6, label='D Fake')
ax1.plot([x[0] for x in gan.g_losses], color='orange', linewidth=0.5, label='G')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Discriminator / Generator Loss', fontsize=14)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 2)
ax1.grid(True, alpha=0.3)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 2: D/G Accuracy vs Epoch
# ───────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot([x[3] for x in gan.d_losses], color='blue', linewidth=0.5, label='D Total')
ax2.plot([x[4] for x in gan.d_losses], color='green', linewidth=0.25, alpha=0.6, label='D Real')
ax2.plot([x[5] for x in gan.d_losses], color='red', linewidth=0.25, alpha=0.6, label='D Fake')
ax2.plot([x[1] for x in gan.g_losses], color='orange', linewidth=0.5, label='G')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Discriminator / Generator Accuracy', fontsize=14)
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 3: Learning Rate vs Epoch (NEW - uses semilogy scale)
# ───────────────────────────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.semilogy(gan.d_lr_history, color='blue', linewidth=1.5, label='Discriminator LR')
ax3.semilogy(gan.g_lr_history, color='orange', linewidth=1.5, label='Generator LR')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Finalize and save
plt.tight_layout()
plt.savefig(os.path.join(RUN_FOLDER, 'training_summary.png'), dpi=200, bbox_inches='tight')
plt.show()

print(f"✓ Training summary saved to {RUN_FOLDER}/training_summary.png")
'''


# =============================================================================
# CELL: CLEANUP (End of Notebook)
# =============================================================================
# Description: Finish W&B run and optionally restart kernel to release GPU memory.

CELL_CLEANUP = '''
# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

# Finish W&B run
wandb.finish()
print("✓ W&B run finished")

# Print training summary
print(f"\\n{'═'*60}")
print("TRAINING COMPLETE")
print(f"{'═'*60}")
print(f"Epochs trained: {gan.epoch}")
print(f"Final D loss: {gan.d_losses[-1][0]:.4f}")
print(f"Final G loss: {gan.g_losses[-1][0]:.4f}")
print(f"Final D LR: {gan.d_lr_history[-1]:.2e}")
print(f"Final G LR: {gan.g_lr_history[-1]:.2e}")
print(f"Weights saved to: {RUN_FOLDER}/weights/")
print(f"{'═'*60}")
'''


# =============================================================================
# CELL: RESTART KERNEL (Final Cell - Release GPU Memory)
# =============================================================================
# Description: Restart kernel to release GPU memory. This is the ONLY
# guaranteed way to fully release TensorFlow/CUDA GPU memory.
# Run this cell ONLY after all work is complete and saved.

CELL_RESTART_KERNEL = '''
# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP: Restart kernel to fully release GPU memory
# ═══════════════════════════════════════════════════════════════════════════════
# TensorFlow/CUDA does not release GPU memory within a running Python process.
# Restarting the kernel is the only guaranteed way to free all GPU resources.
#
# ⚠️ WARNING: This will clear all variables and outputs!
# Only run after all work is complete and saved.

import IPython
print("Restarting kernel to release GPU memory...")
IPython.Application.instance().kernel.do_shutdown(restart=True)
'''


# =============================================================================
# PRINT ALL CELLS FOR EASY COPYING
# =============================================================================
if __name__ == "__main__":
    print("GAN Notebook Cell Templates")
    print("=" * 60)
    print("\nCopy each section to the corresponding cell in the notebook.\n")
    
    cells = [
        ("Cell 1: GPU Setup", CELL_1_GPU_SETUP),
        ("Cell 2: Imports", CELL_2_IMPORTS),
        ("Cell 3: Global Configuration", CELL_3_GLOBAL_CONFIG),
        ("Cell: W&B Initialization", CELL_WANDB_INIT),
        ("Cell: Training", CELL_TRAINING),
        ("Cell: Training Plots", CELL_TRAINING_PLOTS),
        ("Cell: Cleanup", CELL_CLEANUP),
        ("Cell: Restart Kernel", CELL_RESTART_KERNEL),
    ]
    
    for name, content in cells:
        print(f"\n{'─'*60}")
        print(f"# {name}")
        print(f"{'─'*60}")
        print(content)
