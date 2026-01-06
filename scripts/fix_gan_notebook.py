#!/usr/bin/env python3
"""
Fix GAN Camel Training Notebook - Version 2.

This script properly restructures the notebook by:
1. Fixing import error (get_optimal_batch_size → find_optimal_batch_size)
2. Splitting global configuration into 3 cells
3. Moving dynamic batch size to after model build
4. Removing duplicate cells
"""

import json


def fix_gan_notebook(notebook_path: str) -> None:
    """Apply all fixes to the GAN notebook."""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    new_cells = []
    skip_next_global_config_note = False
    inserted_dynamic_batch = False
    
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        source = cell.get('source', [])
        source_text = ''.join(source) if isinstance(source, list) else source
        
        # Skip the duplicate imports cell (the one after Global Configuration markdown)
        if cell_type == 'code' and '# IMPORTS' in source_text:
            # Check if we already have an imports cell
            already_have_imports = any(
                '# IMPORTS' in ''.join(c.get('source', []))
                for c in new_cells
                if c['cell_type'] == 'code'
            )
            if already_have_imports:
                print("✓ Removing duplicate imports cell")
                continue
        
        # Fix the imports cell (first occurrence)
        if cell_type == 'code' and 'from utils.gpu_utils import' in source_text:
            if 'find_optimal_batch_size' not in source_text:
                print("✓ Fixing imports cell")
                cell['source'] = [
                    "# =============================================================================\n",
                    "# IMPORTS\n",
                    "# =============================================================================\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Path Configuration\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import sys\n",
                    "sys.path.insert(0, '..')      # For v1/src modules\n",
                    "sys.path.insert(0, '../..')   # For project root utils/\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Standard Library\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import os\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Third-Party Libraries\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import wandb\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Project Modules\n",
                    "# -----------------------------------------------------------------------------\n",
                    "from src.models.GAN import GAN\n",
                    "from src.utils.loaders import load_safari\n",
                    "\n",
                    "# GPU utilities for dynamic batch size and epoch scaling\n",
                    "from utils.gpu_utils import (\n",
                    "    find_optimal_batch_size,    # Dynamic OOM-based batch size finder\n",
                    "    calculate_adjusted_epochs,\n",
                    "    get_gpu_vram_gb,\n",
                    "    print_training_config\n",
                    ")"
                ]
                cell['outputs'] = []
                cell['execution_count'] = None
            new_cells.append(cell)
            continue
        
        # Replace the Global Configuration markdown with updated version
        if cell_type == 'markdown' and 'Global Configuration' in source_text and 'Central configuration' in source_text:
            cell['source'] = [
                "## Global Configuration - Part 1: Static Parameters\n",
                "\n",
                "Run identification and reference training values."
            ]
            new_cells.append(cell)
            skip_next_global_config_note = True
            continue
        
        # Replace the monolithic Global Configuration code cell
        if cell_type == 'code' and 'GLOBAL CONFIGURATION' in source_text and 'REFERENCE_BATCH_SIZE' in source_text:
            print("✓ Creating Static Configuration cell")
            cell['source'] = [
                "# ═══════════════════════════════════════════════════════════════════\n",
                "# GLOBAL CONFIGURATION - PART 1: STATIC PARAMETERS\n",
                "# ═══════════════════════════════════════════════════════════════════\n",
                "# These values define the experiment identity and reference baselines.\n",
                "# They do not depend on GPU detection or model building.\n",
                "\n",
                "# -----------------------------------------------------------------------------\n",
                "# Run Identification\n",
                "# -----------------------------------------------------------------------------\n",
                "SECTION = 'gan'\n",
                "RUN_ID = '0001'\n",
                "DATA_NAME = 'camel'\n",
                "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'\n",
                "\n",
                "# Create run directories\n",
                "if not os.path.exists(RUN_FOLDER):\n",
                "    os.makedirs(RUN_FOLDER)\n",
                "    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))\n",
                "    os.makedirs(os.path.join(RUN_FOLDER, 'images'))\n",
                "    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))\n",
                "\n",
                "# -----------------------------------------------------------------------------\n",
                "# Reference Training Configuration\n",
                "# These are the original notebook values used for epoch scaling\n",
                "# -----------------------------------------------------------------------------\n",
                "REFERENCE_BATCH_SIZE = 256\n",
                "REFERENCE_EPOCHS = 6000\n",
                "MODE = 'build'  # Options: 'build' (new training), 'load' (resume)\n",
                "\n",
                "print(f\"Run folder: {RUN_FOLDER}\")\n",
                "print(f\"Reference batch size: {REFERENCE_BATCH_SIZE}\")\n",
                "print(f\"Reference epochs: {REFERENCE_EPOCHS}\")"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None
            new_cells.append(cell)
            continue
        
        # After Model Architecture cell (with gan = GAN(...)), insert dynamic batch cells
        if cell_type == 'code' and 'gan = GAN(' in source_text and not inserted_dynamic_batch:
            new_cells.append(cell)  # Add the original model cell
            inserted_dynamic_batch = True
            
            print("✓ Adding Dynamic Batch Size section after model build")
            
            # Markdown header
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Global Configuration - Part 2: Dynamic Batch Size\n",
                    "\n",
                    "Find optimal batch size using OOM detection. Must run after model is built."
                ]
            })
            
            # Dynamic batch size code
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# GLOBAL CONFIGURATION - PART 2: DYNAMIC BATCH SIZE\n",
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# Uses binary search with OOM detection to find optimal batch size.\n",
                    "# Must run AFTER model is built so we can test actual GPU usage.\n",
                    "\n",
                    "GPU_VRAM_GB = get_gpu_vram_gb()\n",
                    "\n",
                    "# Find optimal batch size by testing generator memory usage\n",
                    "# The generator is typically the memory bottleneck in GANs\n",
                    "BATCH_SIZE = find_optimal_batch_size(\n",
                    "    model=gan.generator,\n",
                    "    input_shape=(gan.z_dim,),\n",
                    ")\n",
                    "\n",
                    "# Scale epochs to maintain equivalent training updates\n",
                    "# Formula: reference_epochs × (reference_batch / actual_batch)\n",
                    "EPOCHS = calculate_adjusted_epochs(\n",
                    "    REFERENCE_EPOCHS,\n",
                    "    REFERENCE_BATCH_SIZE,\n",
                    "    BATCH_SIZE\n",
                    ")\n",
                    "\n",
                    "# Adjust checkpoint frequency proportionally\n",
                    "PRINT_EVERY_N_BATCHES = max(50 * REFERENCE_BATCH_SIZE // BATCH_SIZE, 10)\n",
                    "\n",
                    "# Print configuration summary\n",
                    "print_training_config(\n",
                    "    batch_size=BATCH_SIZE,\n",
                    "    epochs=EPOCHS,\n",
                    "    model_params=gan.generator.count_params() + gan.discriminator.count_params(),\n",
                    "    reference_batch=REFERENCE_BATCH_SIZE,\n",
                    "    reference_epochs=REFERENCE_EPOCHS,\n",
                    "    vram_gb=GPU_VRAM_GB\n",
                    ")"
                ]
            })
            
            print("✓ Adding LR Scheduler section")
            
            # LR Scheduler markdown
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Global Configuration - Part 3: LR Scheduler\n",
                    "\n",
                    "Configure step decay learning rate for stable GAN training."
                ]
            })
            
            # LR Scheduler code
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# GLOBAL CONFIGURATION - PART 3: LR SCHEDULER\n",
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# Step decay: Reduce LR at fixed intervals for stable late-stage training\n",
                    "\n",
                    "LR_DECAY_FACTOR = 0.5  # Multiply LR by 0.5 at each decay point\n",
                    "LR_DECAY_EPOCHS = EPOCHS // 4  # Decay 4 times during training\n",
                    "\n",
                    "print(f\"LR Decay: ×{LR_DECAY_FACTOR} every {LR_DECAY_EPOCHS} epochs\")\n",
                    "print(f\"Checkpoints: Every {PRINT_EVERY_N_BATCHES} epochs\")"
                ]
            })
            continue
        
        # Keep all other cells unchanged
        new_cells.append(cell)
    
    notebook['cells'] = new_cells
    
    # Write the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n✓ Notebook saved: {notebook_path}")
    print(f"  Total cells: {len(new_cells)}")


if __name__ == "__main__":
    notebook_path = "v1/notebooks/04_01_gan_camel_train.ipynb"
    fix_gan_notebook(notebook_path)
