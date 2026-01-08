#!/usr/bin/env python3
"""
Add define_wgan_charts import and call to notebook.

Usage:
    python scripts/add_define_wgan_charts.py
"""

import json
import os


def update_notebook():
    """Add define_wgan_charts to the notebook."""
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "v1",
        "notebooks",
        "04_02_wgan_cifar_train.ipynb"
    )
    notebook_path = os.path.normpath(notebook_path)

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find the import cell and add define_wgan_charts
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "from utils.wandb_utils import init_wandb" in source:
                # Update the import line
                new_source = []
                for line in cell["source"]:
                    if "from utils.wandb_utils import init_wandb" in line:
                        line = "from utils.wandb_utils import init_wandb, log_images, define_wgan_charts\n"
                    new_source.append(line)
                cell["source"] = new_source
                print(f"✓ Updated import in cell {i}")
                break

    # Find W&B init cell and add define_wgan_charts() call
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "run = init_wandb(" in source and "define_wgan_charts()" not in source:
                # Add definition after the print statements
                cell["source"].append("\n")
                cell["source"].append("# -----------------------------------------------------------------------------\n")
                cell["source"].append("# Configure W&B Chart Panels\n")
                cell["source"].append("# -----------------------------------------------------------------------------\n")
                cell["source"].append("# Define metric groupings for organized W&B visualization.\n")
                cell["source"].append("define_wgan_charts()\n")
                print(f"✓ Added define_wgan_charts() call to cell {i}")
                break

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved: {notebook_path}")


if __name__ == "__main__":
    update_notebook()
