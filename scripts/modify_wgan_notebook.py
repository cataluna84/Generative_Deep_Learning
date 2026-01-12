#!/usr/bin/env python3
"""
Notebook Modification Script for Run Folder Restructuring.

This script modifies the WGAN CIFAR training notebook to implement:
1. DATASET_RUN_ID and EXPERIMENT_RUN_ID configuration variables
2. Auto-increment logic for experiment run IDs
3. Updated W&B run naming format

Usage:
    uv run python scripts/modify_wgan_notebook.py

Author:
    Auto-generated for run folder restructuring implementation.
"""

import json
import os
import sys


def load_notebook(path: str) -> dict:
    """Load a Jupyter notebook from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook: dict, path: str) -> None:
    """Save a Jupyter notebook to file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"✓ Saved notebook: {path}")


def find_cell_by_content(
    cells: list,
    search_text: str,
    cell_type: str = 'code'
) -> int:
    """
    Find the index of a cell containing specific text.

    Args:
        cells: List of notebook cells
        search_text: Text to search for in cell source
        cell_type: Type of cell to search ('code' or 'markdown')

    Returns:
        Index of the cell, or -1 if not found
    """
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != cell_type:
            continue
        source = ''.join(cell.get('source', []))
        if search_text in source:
            return i
    return -1


def update_global_config_cell(cells: list) -> bool:
    """
    Update the Global Configuration cell with new run folder structure.

    Args:
        cells: List of notebook cells

    Returns:
        True if update was successful, False otherwise
    """
    # Find the cell with RUN_ID = '0002'
    idx = find_cell_by_content(cells, "RUN_ID = '0002'", 'code')
    if idx == -1:
        print("❌ Could not find Global Configuration cell")
        return False

    # New source for the Global Configuration cell
    new_source = [
        "# =============================================================================\n",
        "# GLOBAL CONFIGURATION\n",
        "# =============================================================================\n",
        "# These values define the experiment identity and output locations.\n",
        "\n",
        "# =============================================================================\n",
        "# DATASET AND EXPERIMENT CONFIGURATION\n",
        "# =============================================================================\n",
        "# SECTION: Parent folder for all experiments of this type (e.g., 'gan', 'vae')\n",
        "#\n",
        "# DATASET_RUN_ID: Unique identifier for the dataset (e.g., '0002' for horses)\n",
        "#                 This groups all experiments using the same dataset.\n",
        "#\n",
        "# DATA_NAME: Human-readable dataset name (e.g., 'horses', 'camel', 'celeba')\n",
        "#\n",
        "# EXPERIMENT_RUN_ID: Unique ID for THIS specific training experiment.\n",
        "#                    - Should match the run number in the Master Experiment Log\n",
        "#                      (see the 'Experiment Log' section at the end of this notebook)\n",
        "#                    - If empty or None, auto-generates the next available ID\n",
        "#                    - Format: 3-digit zero-padded string (e.g., '001', '002', '005')\n",
        "#\n",
        "# Example: DATASET_RUN_ID='0002', DATA_NAME='horses', EXPERIMENT_RUN_ID='005'\n",
        "#          -> Saves to: ../run/gan/0002_horses/005/\n",
        "# =============================================================================\n",
        "\n",
        "SECTION = 'gan'              # Parent folder for all GAN experiments\n",
        "DATASET_RUN_ID = '0002'      # Dataset identifier\n",
        "DATA_NAME = 'horses'         # Dataset name (CIFAR-10 class 7)\n",
        "EXPERIMENT_RUN_ID = None     # Set to None for auto-increment, or specify e.g., '006'\n",
        "\n",
        "# =============================================================================\n",
        "# AUTO-INCREMENT LOGIC FOR EXPERIMENT_RUN_ID\n",
        "# =============================================================================\n",
        "# If EXPERIMENT_RUN_ID is None or empty, automatically determines the next\n",
        "# available run ID by scanning existing folders. Falls back to '001' if no\n",
        "# prior runs exist.\n",
        "# =============================================================================\n",
        "\n",
        "# Base folder for this dataset (without experiment run ID)\n",
        "BASE_RUN_FOLDER = f'../run/{SECTION}/{DATASET_RUN_ID}_{DATA_NAME}'\n",
        "\n",
        "import re\n",
        "\n",
        "def get_next_experiment_run_id(base_folder):\n",
        "    \"\"\"\n",
        "    Determine the next available experiment run ID.\n",
        "    \n",
        "    Scans existing folders in base_folder for numeric subdirectories\n",
        "    and returns the next sequential ID as a 3-digit zero-padded string.\n",
        "    \n",
        "    Args:\n",
        "        base_folder: Path to dataset run folder (e.g., '../run/gan/0002_horses')\n",
        "        \n",
        "    Returns:\n",
        "        str: Next available run ID (e.g., '001', '002', '006')\n",
        "    \"\"\"\n",
        "    if not os.path.exists(base_folder):\n",
        "        return '001'\n",
        "    \n",
        "    # Find all numeric subdirectories (3-digit format)\n",
        "    existing_ids = []\n",
        "    for item in os.listdir(base_folder):\n",
        "        item_path = os.path.join(base_folder, item)\n",
        "        if os.path.isdir(item_path) and re.match(r'^\\d{3}$', item):\n",
        "            existing_ids.append(int(item))\n",
        "    \n",
        "    if not existing_ids:\n",
        "        return '001'\n",
        "    \n",
        "    # Return next sequential ID\n",
        "    next_id = max(existing_ids) + 1\n",
        "    return f'{next_id:03d}'\n",
        "\n",
        "# Auto-generate EXPERIMENT_RUN_ID if not specified\n",
        "if EXPERIMENT_RUN_ID is None or EXPERIMENT_RUN_ID == '':\n",
        "    EXPERIMENT_RUN_ID = get_next_experiment_run_id(BASE_RUN_FOLDER)\n",
        "    print(f'Auto-generated EXPERIMENT_RUN_ID: {EXPERIMENT_RUN_ID}')\n",
        "\n",
        "# Construct the full run folder path (includes experiment run ID)\n",
        "RUN_FOLDER = f'{BASE_RUN_FOLDER}/{EXPERIMENT_RUN_ID}'\n",
        "\n",
        "# =============================================================================\n",
        "# CREATE RUN FOLDER STRUCTURE\n",
        "# =============================================================================\n",
        "# Creates experiment-specific folders for outputs:\n",
        "# - viz: Training visualization plots (loss curves, etc.)\n",
        "# - images: Generated sample images at checkpoints\n",
        "# - weights: Model weight checkpoints\n",
        "# =============================================================================\n",
        "if not os.path.exists(RUN_FOLDER):\n",
        "    os.makedirs(RUN_FOLDER)\n",
        "    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))\n",
        "    os.makedirs(os.path.join(RUN_FOLDER, 'images'))\n",
        "    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))\n",
        "    print(f'✓ Created new run folder: {RUN_FOLDER}')\n",
        "elif os.listdir(RUN_FOLDER):\n",
        "    print(f'⚠️ WARNING: Run folder {RUN_FOLDER} already exists with files!')\n",
        "    print('  Artifacts may be overwritten. Consider using a new EXPERIMENT_RUN_ID.')\n",
        "else:\n",
        "    print(f'Run folder: {RUN_FOLDER}')\n",
        "\n",
        "# Training mode: 'build' creates new model, 'load' resumes from checkpoint\n",
        "MODE = 'build'\n",
        "\n",
        "print(f'Run folder: {RUN_FOLDER}')\n",
        "print(f'Mode: {MODE}')"
    ]

    # Update the cell
    cells[idx]['source'] = new_source
    # Clear outputs since the cell has changed
    cells[idx]['outputs'] = []
    cells[idx]['execution_count'] = None
    print(f"✓ Updated Global Configuration cell (index {idx})")
    return True


def update_wandb_run_name(cells: list) -> bool:
    """
    Update the W&B initialization to use new run name format.

    The new format includes both dataset and experiment IDs:
    wgan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}

    Args:
        cells: List of notebook cells

    Returns:
        True if update was successful, False otherwise
    """
    # Find the cell with wandb.init
    idx = find_cell_by_content(cells, "wandb.init", 'code')
    if idx == -1:
        print("❌ Could not find W&B init cell")
        return False

    # Get the source and update the run name line
    source = cells[idx].get('source', [])
    updated = False

    for i, line in enumerate(source):
        # Look for the name parameter in wandb.init
        if 'name=f"wgan_' in line and 'RUN_ID' in line:
            # Replace RUN_ID with DATASET_RUN_ID and add EXPERIMENT_RUN_ID
            new_line = line.replace(
                '{RUN_ID}"',
                '{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}"'
            )
            source[i] = new_line
            updated = True
            print(f"✓ Updated W&B run name format in cell {idx}")
            break

    if updated:
        cells[idx]['source'] = source
        return True

    print("⚠ Could not find RUN_ID in W&B init cell, skipping update")
    return False


def main():
    """Main entry point for notebook modification."""
    # Configuration
    notebook_path = "v1/notebooks/04_02_wgan_cifar_train.ipynb"

    # Check if notebook exists
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)

    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)
    cells = notebook.get('cells', [])

    print(f"Found {len(cells)} cells")

    # Apply modifications
    success = True

    # 1. Update Global Configuration cell
    if not update_global_config_cell(cells):
        success = False

    # 2. Update W&B run name format
    if not update_wandb_run_name(cells):
        # This is non-critical, continue
        pass

    # Save the notebook
    if success:
        save_notebook(notebook, notebook_path)
        print("\n✓ All modifications complete!")
    else:
        print("\n❌ Some modifications failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
