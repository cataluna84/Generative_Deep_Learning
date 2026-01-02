#!/usr/bin/env python3
"""
Add kernel restart cleanup cell to notebooks.
This script adds a final cell to each notebook that restarts the kernel
to fully release GPU memory.
"""

import json
import sys
from pathlib import Path

# Define the cleanup cell to add
CLEANUP_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Cleanup: Restart Kernel to Release GPU Memory"
    ]
}

CLEANUP_CODE_CELL = {
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
        "# Run this cell only after all work is complete and saved.\n",
        "\n",
        "import IPython\n",
        "print(\"Restarting kernel to release GPU memory...\")\n",
        "IPython.Application.instance().kernel.do_shutdown(restart=True)"
    ]
}


def add_cleanup_cell_to_notebook(notebook_path: Path) -> bool:
    """
    Add the cleanup cell to a notebook if not already present.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        True if cell was added, False if already present
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if cleanup cell already exists
    cells = notebook.get('cells', [])
    for cell in cells:
        source = ''.join(cell.get('source', []))
        if 'do_shutdown(restart=True)' in source:
            print(f"  ⏭️  Cleanup cell already exists in {notebook_path.name}")
            return False
    
    # Add the cleanup cells at the end
    cells.append(CLEANUP_CELL)
    cells.append(CLEANUP_CODE_CELL)
    notebook['cells'] = cells
    
    # Write the notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"  ✅ Added cleanup cell to {notebook_path.name}")
    return True


def main():
    """Main function to process all specified notebooks."""
    # List of notebooks to update
    notebooks = [
        "v1/notebooks/02_01_deep_learning_deep_neural_network.ipynb",
        "v1/notebooks/02_02_deep_learning_convolutions.ipynb",
        "v1/notebooks/02_03_deep_learning_conv_neural_network.ipynb",
        "v1/notebooks/03_01_autoencoder_train.ipynb",
        "v1/notebooks/03_02_autoencoder_analysis.ipynb",
        "v1/notebooks/03_03_vae_digits_train.ipynb",
        "v1/notebooks/03_04_vae_digits_analysis.ipynb",
        "v1/notebooks/03_05_vae_faces_train.ipynb",
        "v1/notebooks/03_06_vae_faces_analysis.ipynb",
    ]
    
    # Get the project root (assuming script is run from project root)
    project_root = Path(__file__).parent.parent
    
    print("Adding kernel restart cleanup cells to notebooks...")
    print("=" * 60)
    
    added_count = 0
    for notebook_rel_path in notebooks:
        notebook_path = project_root / notebook_rel_path
        if notebook_path.exists():
            if add_cleanup_cell_to_notebook(notebook_path):
                added_count += 1
        else:
            print(f"  ❌ Notebook not found: {notebook_rel_path}")
    
    print("=" * 60)
    print(f"Done! Added cleanup cells to {added_count} notebook(s).")


if __name__ == "__main__":
    main()
