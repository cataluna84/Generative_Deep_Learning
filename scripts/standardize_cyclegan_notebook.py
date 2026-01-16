"""Standardize CycleGAN notebook cells.

This script updates the CycleGAN training notebook with:
- Standardized GPU memory growth cell
- Consolidated global configuration
- Run folder auto-increment
- W&B initialization
- Artifact upload cells
- Kernel restart cell
"""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NOTEBOOK_PATH = "v1/notebooks/05_01_cyclegan_train.ipynb"

# ═══════════════════════════════════════════════════════════════════════════════
# NEW CELL CONTENTS
# ═══════════════════════════════════════════════════════════════════════════════

GPU_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION
# Enable memory growth to prevent TensorFlow from allocating all GPU memory
# ═══════════════════════════════════════════════════════════════════════════════
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected, running on CPU")'''

IMPORTS_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# Standard library, TensorFlow/Keras, and project-specific imports
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import re
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import wandb

from src.models.cycleGAN import CycleGAN
from src.utils.loaders import DataLoader'''

CONFIG_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# All hyperparameters and paths defined here for easy modification
# ═══════════════════════════════════════════════════════════════════════════════
SECTION = 'paint'
DATASET_RUN_ID = '0001'
DATA_NAME = 'apple2orange'
EXPERIMENT_RUN_ID = None  # Set to None for auto-increment

# Model hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 200
LEARNING_RATE = 0.0002
PRINT_EVERY_N_BATCHES = 20

# Test files for visualization
TEST_A_FILE = 'n07740461_1470.jpg'
TEST_B_FILE = 'n07749192_4241.jpg'

# CycleGAN architecture
GENERATOR_TYPE = 'unet'
GEN_N_FILTERS = 32
DISC_N_FILTERS = 32
LAMBDA_VALIDATION = 1
LAMBDA_RECONSTR = 10
LAMBDA_ID = 2
BUFFER_MAX_LENGTH = 50'''

RUN_FOLDER_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# RUN FOLDER CONFIGURATION
# Auto-increment experiment ID for each new run
# ═══════════════════════════════════════════════════════════════════════════════
BASE_RUN_FOLDER = f'../run/{SECTION}/{DATASET_RUN_ID}_{DATA_NAME}'

def get_next_experiment_run_id(base_folder):
    """Determine next available 3-digit experiment run ID."""
    if not os.path.exists(base_folder):
        return '001'
    existing = [int(d) for d in os.listdir(base_folder) 
                if os.path.isdir(os.path.join(base_folder, d)) and re.match(r'^\\d{3}$', d)]
    return f'{max(existing) + 1:03d}' if existing else '001'

if EXPERIMENT_RUN_ID is None:
    EXPERIMENT_RUN_ID = get_next_experiment_run_id(BASE_RUN_FOLDER)
    print(f'Auto-generated EXPERIMENT_RUN_ID: {EXPERIMENT_RUN_ID}')

RUN_FOLDER = f'{BASE_RUN_FOLDER}/{EXPERIMENT_RUN_ID}'

# Create run folder structure
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))
    print(f'Created run folder: {RUN_FOLDER}')

mode = 'build'  # 'build' for new model, 'load' to continue training'''

WANDB_INIT_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHTS & BIASES INITIALIZATION
# Track experiments, metrics, and artifacts
# ═══════════════════════════════════════════════════════════════════════════════
wandb.init(
    project="generative-deep-learning",
    name=f"cyclegan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}",
    config={
        "model": "CycleGAN",
        "dataset": DATA_NAME,
        "dataset_run_id": DATASET_RUN_ID,
        "experiment_run_id": EXPERIMENT_RUN_ID,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "generator_type": GENERATOR_TYPE,
        "lambda_validation": LAMBDA_VALIDATION,
        "lambda_reconstr": LAMBDA_RECONSTR,
        "lambda_id": LAMBDA_ID,
    }
)'''

DATA_LOADER_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# Initialize data loader for image batches
# ═══════════════════════════════════════════════════════════════════════════════
data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))'''

MODEL_BUILD_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# Build CycleGAN with configured hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
gan = CycleGAN(
    input_dim=(IMAGE_SIZE, IMAGE_SIZE, 3),
    learning_rate=LEARNING_RATE,
    buffer_max_length=BUFFER_MAX_LENGTH,
    lambda_validation=LAMBDA_VALIDATION,
    lambda_reconstr=LAMBDA_RECONSTR,
    lambda_id=LAMBDA_ID,
    generator_type=GENERATOR_TYPE,
    gen_n_filters=GEN_N_FILTERS,
    disc_n_filters=DISC_N_FILTERS,
)

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.weights.h5'))'''

TRAIN_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# Train CycleGAN with W&B logging enabled
# ═══════════════════════════════════════════════════════════════════════════════
gan.train(
    data_loader,
    run_folder=RUN_FOLDER,
    epochs=EPOCHS,
    test_A_file=TEST_A_FILE,
    test_B_file=TEST_B_FILE,
    batch_size=BATCH_SIZE,
    sample_interval=PRINT_EVERY_N_BATCHES,
    wandb_log=True  # Enable W&B logging
)'''

ARTIFACT_UPLOAD_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD ARTIFACTS TO W&B
# Save model visualizations and generated images for experiment tracking
# ═══════════════════════════════════════════════════════════════════════════════
# Upload viz folder (model architecture diagrams)
viz_artifact = wandb.Artifact(
    name=f"cyclegan_viz_{DATA_NAME}_{EXPERIMENT_RUN_ID}",
    type="model_architecture"
)
viz_artifact.add_dir(os.path.join(RUN_FOLDER, "viz"))
wandb.log_artifact(viz_artifact)

# Upload images folder (generated samples)
images_artifact = wandb.Artifact(
    name=f"cyclegan_images_{DATA_NAME}_{EXPERIMENT_RUN_ID}",
    type="generated_images"
)
images_artifact.add_dir(os.path.join(RUN_FOLDER, "images"))
wandb.log_artifact(images_artifact)

print("Artifacts uploaded to W&B")'''

WANDB_FINISH_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# FINISH W&B RUN
# ═══════════════════════════════════════════════════════════════════════════════
wandb.finish()'''

KERNEL_RESTART_CELL = '''# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP: Restart kernel to fully release GPU memory
# TensorFlow/CUDA does not release GPU memory within a running Python process.
# Restarting the kernel is the only guaranteed way to free all GPU resources.
# ═══════════════════════════════════════════════════════════════════════════════
import IPython
print("Restarting kernel to release GPU memory...")
IPython.Application.instance().kernel.do_shutdown(restart=True)'''

EXPERIMENT_LOG_MD = '''## Master Experiment Log

| Run | Date | W&B URL | Epochs | Batch | D Loss | G Loss | Notes |
|-----|------|---------|--------|-------|--------|--------|-------|
| 001 | 2026-01-15 | [View](#) | 200 | 1 | - | - | Initial run |
'''

def main():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Build new cell list
    new_cells = []
    
    # Keep title and download instructions (first 2 cells)
    new_cells.append(nb.cells[0])  # Title
    new_cells.append(nb.cells[1])  # Download instructions
    
    # Add standardized cells
    new_cells.append(new_code_cell(GPU_CELL))
    new_cells.append(new_code_cell(IMPORTS_CELL))
    new_cells.append(new_code_cell(CONFIG_CELL))
    new_cells.append(new_code_cell(RUN_FOLDER_CELL))
    new_cells.append(new_code_cell(WANDB_INIT_CELL))
    
    # Data section
    new_cells.append(new_markdown_cell("## Data"))
    new_cells.append(new_code_cell(DATA_LOADER_CELL))
    
    # Architecture section
    new_cells.append(new_markdown_cell("## Model Architecture"))
    new_cells.append(new_code_cell(MODEL_BUILD_CELL))
    
    # Keep model summary cells (find and copy them)
    for cell in nb.cells:
        if cell.cell_type == 'code' and '.summary()' in cell.source:
            new_cells.append(cell)
    
    # Training section
    new_cells.append(new_markdown_cell("## Training"))
    new_cells.append(new_code_cell(TRAIN_CELL))
    
    # Loss visualization section
    new_cells.append(new_markdown_cell("## Loss Visualization"))
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'g_losses' in cell.source and 'plt.plot' in cell.source:
            new_cells.append(cell)
            break
    
    # Post-training section
    new_cells.append(new_markdown_cell("## Post-Training"))
    new_cells.append(new_code_cell(ARTIFACT_UPLOAD_CELL))
    new_cells.append(new_code_cell(WANDB_FINISH_CELL))
    
    # Experiment log
    new_cells.append(new_markdown_cell(EXPERIMENT_LOG_MD))
    
    # Kernel restart (final cell)
    new_cells.append(new_code_cell(KERNEL_RESTART_CELL))
    
    # Replace cells
    nb.cells = new_cells
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("Notebook standardized successfully!")
    print(f"Total cells: {len(nb.cells)}")

if __name__ == "__main__":
    main()
