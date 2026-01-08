#!/usr/bin/env python3
"""
Update the WGAN CIFAR notebook experiment log with latest run data.

This script adds a new entry to the Master Experiment Log table.

Usage:
    python scripts/update_wgan_experiment_log.py
"""

import json
import os


def update_experiment_log():
    """Update experiment log with the latest run data."""
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

    # Find the experiment log cell
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            if "## Experiment Log" in source and "Master Experiment Log" in source:
                # Find and update the placeholder row (Run 002)
                new_source = []
                for line in cell["source"]:
                    if "| 002 | - | - | - | - | - | - | - | - | *Placeholder* |" in line:
                        # Replace with actual run data
                        line = "| 002 | 2026-01-08 | [View](https://wandb.ai/cataluna84/generative-deep-learning/) | 512 | 6000 | 5e-5 | ✅ Stable | 3.70 | -113.3 | Verbose metrics test, FID improved to ~334 |\n"
                    new_source.append(line)
                cell["source"] = new_source
                print("✓ Updated experiment log entry for Run 002")
                break

    # Save the notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved notebook: {notebook_path}")


if __name__ == "__main__":
    update_experiment_log()
