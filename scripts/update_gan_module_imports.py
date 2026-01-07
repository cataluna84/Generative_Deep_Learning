#!/usr/bin/env python3
"""
Update WGAN notebook imports to use renamed gan_ prefixed modules.

This script updates the notebook to import from:
- utils.gan_stability_analysis (was utils.stability_analysis)
- utils.gan_report_generator (was utils.report_generator)

Usage:
    python scripts/update_gan_module_imports.py
"""

import json
import os


def update_imports():
    """Update notebook imports to use renamed modules."""
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

    updated = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for i, line in enumerate(cell["source"]):
                # Update stability_analysis import
                if "from utils.stability_analysis import" in line:
                    cell["source"][i] = line.replace(
                        "utils.stability_analysis",
                        "utils.gan_stability_analysis"
                    )
                    updated = True
                    print("✓ Updated stability_analysis import")
                
                # Update report_generator import
                if "from utils.report_generator import" in line:
                    cell["source"][i] = line.replace(
                        "utils.report_generator",
                        "utils.gan_report_generator"
                    )
                    updated = True
                    print("✓ Updated report_generator import")

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=4)
        print(f"✓ Saved updated notebook: {notebook_path}")
    else:
        print("⚠ No imports found to update")

    return updated


if __name__ == "__main__":
    update_imports()
