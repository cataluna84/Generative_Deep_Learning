#!/usr/bin/env python3
"""
Update WGAN experiment logging cell with W&B URL and move to end of notebook.

This script:
1. Updates URL_HERE placeholders with the actual W&B run URL
2. Moves the experiment log cell to after the kernel restart/cleanup cell

Usage:
    python scripts/update_wgan_experiment_logging.py
"""

import json
import os


def update_experiment_logging():
    """Update W&B URL and move experiment log to end of notebook."""
    # Define paths
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "v1",
        "notebooks",
        "04_02_wgan_cifar_train.ipynb"
    )
    notebook_path = os.path.normpath(notebook_path)

    # W&B URL to insert
    wandb_url = "https://wandb.ai/cataluna84/generative-deep-learning/runs/x5ln97by"

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find and update the experiment logging cell
    experiment_cell_index = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source_text = "".join(cell["source"])
            if "## Experiment Log" in source_text:
                experiment_cell_index = i
                # Update URL_HERE placeholders
                cell["source"] = [
                    line.replace("URL_HERE", wandb_url)
                    for line in cell["source"]
                ]
                print(f"✓ Updated W&B URL placeholders with: {wandb_url}")
                break

    if experiment_cell_index is None:
        print("⚠ Experiment logging cell not found!")
        return False

    # Remove the cell from its current position
    experiment_cell = notebook["cells"].pop(experiment_cell_index)

    # Append it to the end of the notebook
    notebook["cells"].append(experiment_cell)
    print(f"✓ Moved experiment log cell from position {experiment_cell_index} to end")

    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved updated notebook: {notebook_path}")
    return True


if __name__ == "__main__":
    update_experiment_logging()
