#!/usr/bin/env python3
"""
Notebook Update Script for W&B Run Name Format.

This script updates the W&B initialization in the WGAN notebook to use
the new run name format with DATASET_RUN_ID and EXPERIMENT_RUN_ID.

Usage:
    uv run python scripts/update_wandb_name.py

Author:
    Auto-generated for run folder restructuring implementation.
"""

import json
import os
import re
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


def update_wandb_name_in_notebook(notebook_path: str) -> bool:
    """
    Update W&B run name format in the notebook.

    Changes: name=f"wgan_{DATA_NAME}_{RUN_ID}"
    To:      name=f"wgan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}"

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if update was successful, False otherwise
    """
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the old pattern exists
    old_pattern = 'name=f"wgan_{DATA_NAME}_{RUN_ID}"'
    new_pattern = 'name=f"wgan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}"'

    if old_pattern not in content:
        print(f"⚠ Pattern not found: {old_pattern}")
        # Check if already updated
        if new_pattern in content:
            print("✓ W&B run name already uses new format")
            return True
        return False

    # Replace the pattern
    updated_content = content.replace(old_pattern, new_pattern)

    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print(f"✓ Updated W&B run name format")
    print(f"  Old: {old_pattern}")
    print(f"  New: {new_pattern}")
    return True


def add_comment_for_wandb_name(notebook_path: str) -> bool:
    """
    Add a comment explaining the new W&B run name format.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if update was successful, False otherwise
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the old comment pattern and add more context
    old_comment = '"# Config uses the global constants defined above - no duplication!\\n"'
    new_comment = (
        '"# Config uses the global constants defined above - no duplication!\\n",\n'
        '    "# W&B run name format: wgan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}\\n",\n'
        '    "# Example: wgan_horses_0002_005\\n"'
    )

    if old_comment not in content:
        print("⚠ Could not find W&B comment to update, skipping")
        return False

    # Only add if not already added
    if "Example: wgan_horses_0002_005" in content:
        print("✓ W&B comment already includes new format example")
        return True

    updated_content = content.replace(old_comment, new_comment)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print("✓ Added W&B run name format comment")
    return True


def main():
    """Main entry point."""
    notebook_path = "v1/notebooks/04_02_wgan_cifar_train.ipynb"

    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)

    print(f"Updating W&B run name in: {notebook_path}")

    # Update the run name format
    if not update_wandb_name_in_notebook(notebook_path):
        print("⚠ W&B run name update had issues")

    # Add explanatory comment
    add_comment_for_wandb_name(notebook_path)

    print("\n✓ W&B name update complete!")


if __name__ == "__main__":
    main()
