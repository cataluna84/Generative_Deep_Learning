#!/usr/bin/env python3
"""
Fix the WGAN CIFAR notebook experiment log with correct W&B URL.

Usage:
    python scripts/fix_wgan_experiment_url.py
"""

import json
import os


def fix_experiment_url():
    """Fix experiment log with correct W&B URL."""
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
                # Find and fix the URL for Run 002
                new_source = []
                for line in cell["source"]:
                    if "| 002 |" in line and "wandb.ai" in line:
                        # Replace with correct URL
                        line = "| 002 | 2026-01-08 | [View](https://wandb.ai/cataluna84/generative-deep-learning/runs/g0wvrotx) | 512 | 6000 | 5e-5 | ✅ Stable | 3.70 | -113.3 | Verbose metrics test, FID ~334 |\n"
                    new_source.append(line)
                cell["source"] = new_source
                print("✓ Fixed W&B URL for Run 002")
                break

    # Save the notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved notebook: {notebook_path}")


if __name__ == "__main__":
    fix_experiment_url()
