#!/usr/bin/env python3
"""
Script to add W&B artifact logging to the GAN Camel notebook.
"""
import json
import sys
import os

def load_notebook(path):
    """Load notebook as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, path):
    """Save notebook as JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

def find_cell_containing(notebook, text):
    """Find index of cell containing specified text."""
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if text in source:
            return i
    return -1

def modify_cleanup_cell(notebook):
    """Add artifact logging to W&B cleanup cell."""
    idx = find_cell_containing(notebook, "W&B CLEANUP")
    if idx == -1:
        print("ERROR: Could not find W&B CLEANUP cell")
        return False
    
    new_source = [
        "# =============================================================================\n",
        "# W&B CLEANUP\n",
        "# =============================================================================\n",
        "# Finish the W&B run to ensure all data is synced\n",
        "\n",
        "# Log the 'viz' directory as an artifact\n",
        "if os.path.exists(os.path.join(RUN_FOLDER, 'viz')):\n",
        "    artifact = wandb.Artifact(f'gan-camel-viz-{EXPERIMENT_RUN_ID}', type='predictions')\n",
        "    artifact.add_dir(os.path.join(RUN_FOLDER, 'viz'))\n",
        "    wandb.log_artifact(artifact)\n",
        "    print(f\"\\u2713 Logged artifact: gan-camel-viz-{EXPERIMENT_RUN_ID}\")\n",
        "\n",
        "wandb.finish()\n",
        "print(\"\\u2713 W&B run finished and synced\")"
    ]
    
    notebook['cells'][idx]['source'] = new_source
    notebook['cells'][idx]['outputs'] = [] # Clear outputs
    print("  ✓ Updated W&B cleanup cell with artifact logging")
    return True

def main():
    notebook_path = "v1/notebooks/04_01_gan_camel_train.ipynb"
    
    # Ensure correct path if running from root or scripts/
    if not os.path.exists(notebook_path) and os.path.exists(f"../{notebook_path}"):
        notebook_path = f"../{notebook_path}"
    
    print(f"Loading notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"ERROR: Notebook not found at {notebook_path}")
        sys.exit(1)
        
    notebook = load_notebook(notebook_path)
    
    print("\nApplying modifications:")
    
    if modify_cleanup_cell(notebook):
        save_notebook(notebook, notebook_path)
        print(f"\n✓ Notebook saved: {notebook_path}")
    else:
        print("\n✗ Modification failed, notebook not saved")
        sys.exit(1)

if __name__ == "__main__":
    main()
