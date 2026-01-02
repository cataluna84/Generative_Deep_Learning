#!/usr/bin/env python3
"""
Notebook Cell Updater.

This script updates specific cells in a Jupyter notebook without
regenerating the entire notebook from scratch.

Usage:
    uv run python v1/scripts/update_notebook_cell.py
"""

import json
import sys


def update_notebook_cell(notebook_path: str, cell_index: int, new_source: str):
    """
    Update a specific cell in a notebook.
    
    Args:
        notebook_path: Path to the .ipynb file
        cell_index: Index of the cell to update (0-based)
        new_source: New source code for the cell
    """
    # Read the notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Update the cell
    lines = new_source.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        source_lines.append(lines[-1])
    
    nb['cells'][cell_index]['source'] = source_lines
    nb['cells'][cell_index]['outputs'] = []  # Clear old outputs
    
    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"✓ Updated cell {cell_index} in {notebook_path}")


# =============================================================================
# FIXED IMPORTS CELL
# =============================================================================
# The fix: Added sys.path.insert(0, '../..') to access project root utils/

FIXED_IMPORTS_CELL = """# =============================================================================
# IMPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
# Add parent directories to path for importing project modules
# - '..' gives access to v1/src modules
# - '../..' gives access to project root utils/ directory
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
)"""


if __name__ == "__main__":
    notebook_path = "v1/notebooks/04_01_gan_camel_train.ipynb"
    
    # Cell 4 is the imports cell (index 4 in 0-based indexing)
    # Let's verify by reading the notebook first
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find the imports cell
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source'])
        if 'from utils.gpu_utils import' in source:
            print(f"Found imports cell at index {i}")
            update_notebook_cell(notebook_path, i, FIXED_IMPORTS_CELL)
            break
    else:
        print("Could not find imports cell!")
        sys.exit(1)
