#!/usr/bin/env python3
"""
Update WGAN notebook imports to use new utils/gan package structure.

This script updates the notebook to import from:
- utils.gan.stability_analysis
- utils.gan.report_generator
- utils.gan.metrics
- utils.gan.quality_metrics

Usage:
    python scripts/update_wgan_imports_to_gan_package.py
"""

import json
import os


def update_imports():
    """Update notebook imports to use new package structure."""
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
            new_source = []
            for line in cell["source"]:
                # Update various import patterns
                if "from utils.gan_stability_analysis import" in line:
                    line = line.replace(
                        "utils.gan_stability_analysis",
                        "utils.gan.stability_analysis"
                    )
                    updated = True
                    print("✓ Updated stability_analysis import")
                
                if "from utils.gan_report_generator import" in line:
                    line = line.replace(
                        "utils.gan_report_generator",
                        "utils.gan.report_generator"
                    )
                    updated = True
                    print("✓ Updated report_generator import")

                new_source.append(line)
            
            cell["source"] = new_source

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=4)
        print(f"✓ Saved updated notebook: {notebook_path}")
    else:
        print("⚠ No imports found to update")

    return updated


if __name__ == "__main__":
    update_imports()
