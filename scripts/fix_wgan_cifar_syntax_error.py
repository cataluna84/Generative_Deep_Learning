#!/usr/bin/env python3
"""
Fix syntax error in WGAN CIFAR notebook.

This script fixes the missing closing parenthesis in the generated sample images
cell of the 04_02_wgan_cifar_train.ipynb notebook.

Bug: gen_imgs = 0.5 * (gen_imgs + 1
Fix: gen_imgs = 0.5 * (gen_imgs + 1)

Usage:
    python scripts/fix_wgan_cifar_syntax_error.py
"""

import json
import os


def fix_notebook():
    """Fix the missing closing parenthesis in the rescaling line."""
    # Define paths
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "v1",
        "notebooks",
        "04_02_wgan_cifar_train.ipynb"
    )
    notebook_path = os.path.normpath(notebook_path)

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Track if we made a fix
    fixed = False

    # Search for the buggy cell and fix it
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for i, line in enumerate(cell["source"]):
                # Find the buggy line (missing closing parenthesis)
                if line == "gen_imgs = 0.5 * (gen_imgs + 1\n":
                    cell["source"][i] = "gen_imgs = 0.5 * (gen_imgs + 1)\n"
                    fixed = True
                    print(f"✓ Fixed line {i + 1} in cell")
                    print(f"  Before: gen_imgs = 0.5 * (gen_imgs + 1")
                    print(f"  After:  gen_imgs = 0.5 * (gen_imgs + 1)")
                    break
        if fixed:
            break

    if not fixed:
        print("⚠ Bug not found - notebook may already be fixed")
        return False

    # Write the fixed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved fixed notebook to: {notebook_path}")
    return True


if __name__ == "__main__":
    fix_notebook()
