#!/usr/bin/env python3
"""
Script to add dynamic GPU scaling to 02_01_deep_learning_deep_neural_network.ipynb.

This script updates the notebook to use utils/gpu_utils.py for dynamic batch size
and epoch calculation based on available GPU VRAM.

Usage:
    uv run python scripts/add_dynamic_gpu_scaling.py

Changes made:
    1. Adds gpu_utils imports to the imports cell
    2. Adds a new Global Configuration cell with dynamic batch/epoch scaling
    3. Updates W&B config to include GPU_VRAM_GB
    4. Updates header to mention dynamic scaling
"""

import json
import os


def update_notebook():
    """Update the notebook with dynamic GPU scaling."""
    # Path to the notebook
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'v1', 'notebooks',
        '02_01_deep_learning_deep_neural_network.ipynb'
    )
    notebook_path = os.path.abspath(notebook_path)
    
    print(f"Updating notebook: {notebook_path}")
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Track changes
    changes_made = []
    
    # Find cell indices for specific cells
    imports_cell_idx = None
    data_section_idx = None
    num_classes_cell_idx = None
    wandb_init_cell_idx = None
    
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        
        if cell['cell_type'] == 'code':
            # Find imports cell (contains numpy and keras imports)
            if 'import numpy as np' in source and 'from keras' in source:
                imports_cell_idx = i
            
            # Find NUM_CLASSES definition cell
            if 'NUM_CLASSES = 10' in source and len(source.strip()) < 30:
                num_classes_cell_idx = i
            
            # Find W&B init cell
            if 'init_wandb(' in source:
                wandb_init_cell_idx = i
        
        elif cell['cell_type'] == 'markdown':
            # Find "# data" section
            if source.strip() == '# data':
                data_section_idx = i
    
    print(f"Found indices: imports={imports_cell_idx}, data={data_section_idx}, "
          f"num_classes={num_classes_cell_idx}, wandb_init={wandb_init_cell_idx}")
    
    # 1. Update imports cell to add gpu_utils
    if imports_cell_idx is not None:
        nb['cells'][imports_cell_idx]['source'] = [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from keras.layers import Input, Flatten, Dense, Conv2D\n",
            "from keras.models import Model\n",
            "from keras.optimizers import Adam\n",
            "from keras.utils import to_categorical\n",
            "from keras.datasets import cifar10\n",
            "\n",
            "# Path setup for project utilities\n",
            "import sys\n",
            "sys.path.insert(0, '../..')    # For project root utils/\n",
            "\n",
            "# W&B integration for experiment tracking\n",
            "from utils.wandb_utils import init_wandb, get_metrics_logger\n",
            "import wandb\n",
            "\n",
            "# GPU utilities for dynamic batch size and epoch scaling\n",
            "from utils.gpu_utils import (\n",
            "    get_optimal_batch_size,\n",
            "    calculate_adjusted_epochs,\n",
            "    get_gpu_vram_gb,\n",
            "    print_training_config\n",
            ")"
        ]
        nb['cells'][imports_cell_idx]['outputs'] = []
        nb['cells'][imports_cell_idx]['execution_count'] = None
        changes_made.append(f"Cell {imports_cell_idx}: Added gpu_utils imports")
    
    # 2. Replace NUM_CLASSES cell with full Global Configuration cell
    if num_classes_cell_idx is not None:
        nb['cells'][num_classes_cell_idx]['source'] = [
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
            "# Model configuration\n",
            "NUM_CLASSES = 10            # CIFAR-10 classes\n",
            "\n",
            "# Print configuration summary\n",
            "print_training_config(\n",
            "    'dense_network', BATCH_SIZE, EPOCHS,\n",
            "    REFERENCE_BATCH_SIZE, REFERENCE_EPOCHS, GPU_VRAM_GB\n",
            ")"
        ]
        nb['cells'][num_classes_cell_idx]['outputs'] = []
        nb['cells'][num_classes_cell_idx]['execution_count'] = None
        changes_made.append(f"Cell {num_classes_cell_idx}: Added full Global Configuration with dynamic scaling")
    
    # 3. Update "# data" markdown to "## Global Configuration"
    if data_section_idx is not None:
        nb['cells'][data_section_idx]['source'] = ["## Global Configuration"]
        changes_made.append(f"Cell {data_section_idx}: Updated section header")
    
    # 4. Update W&B init cell to include GPU_VRAM_GB
    if wandb_init_cell_idx is not None:
        current_source = ''.join(nb['cells'][wandb_init_cell_idx].get('source', []))
        if 'gpu_vram_gb' not in current_source:
            nb['cells'][wandb_init_cell_idx]['source'] = [
                "# Initialize W&B for experiment tracking\n",
                "run = init_wandb(\n",
                "    name='02_01_deep_neural_network',\n",
                "    config={\n",
                "        'model': 'dense_network',\n",
                "        'dataset': 'cifar10',\n",
                "        'num_classes': NUM_CLASSES,\n",
                "        'batch_size': BATCH_SIZE,\n",
                "        'epochs': EPOCHS,\n",
                "        'gpu_vram_gb': GPU_VRAM_GB,\n",
                "    }\n",
                ")"
            ]
            nb['cells'][wandb_init_cell_idx]['outputs'] = []
            nb['cells'][wandb_init_cell_idx]['execution_count'] = None
            changes_made.append(f"Cell {wandb_init_cell_idx}: Added gpu_vram_gb to W&B config")
    
    # 5. Update header cell (cell 0)
    if nb['cells'][0]['cell_type'] == 'markdown':
        source = ''.join(nb['cells'][0].get('source', []))
        if 'Dynamic batch size and epoch scaling' not in source:
            nb['cells'][0]['source'] = [
                "# Your first deep neural network\n",
                "\n",
                "This notebook implements a simple Multi-Layer Perceptron (MLP) on the CIFAR-10 dataset.\n",
                "\n",
                "**Standards Applied:**\n",
                "- GPU memory growth enabled\n",
                "- Global configuration block\n",
                "- **Dynamic batch size and epoch scaling** (using `utils/gpu_utils.py`)\n",
                "- W&B integration for experiment tracking\n",
                "- LRFinder for optimal learning rate detection\n",
                "- Full callback stack (WandbMetricsLogger, LRScheduler, EarlyStopping, LRLogger)"
            ]
            changes_made.append("Cell 0: Updated header with dynamic scaling mention")
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CHANGES MADE:")
    print("=" * 60)
    for change in changes_made:
        print(f"  - {change}")
    if not changes_made:
        print("  (No changes needed - already up to date)")
    print("=" * 60)
    print(f"\nNotebook saved: {notebook_path}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    update_notebook()
