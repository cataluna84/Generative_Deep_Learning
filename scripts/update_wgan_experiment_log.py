#!/usr/bin/env python3
"""
Update WGAN Experiment Log Script

This script adds Run 005 (the 10,000 epoch run) to the master experiment log
table in the WGAN CIFAR training notebook.

Run information:
- Date: 2026-01-12
- Epochs: 10000
- Batch Size: 512
- LR: 5e-5
- Stability: ⚠️ Unstable
- Final D Loss: -0.08
- Final G Loss: -1.73
- Notes: Full 10K epochs; 3/5 indicators passed; mode collapse detected
"""

import json
import sys
from pathlib import Path

# Define paths
NOTEBOOK_PATH = Path(__file__).parent.parent / "v1" / "notebooks" / "04_02_wgan_cifar_train.ipynb"

# New experiment entry to add
NEW_ENTRY = (
    "| 005 | 2026-01-12 | "
    "[View](https://wandb.ai/cataluna84/generative-deep-learning/runs/1ex40gx6) | "
    "512 | 10000 | 5e-5 | ⚠️ Unstable | -0.08 | -1.73 | "
    "Full 10K epochs; 3/5 indicators passed; mode collapse detected |\\n"
)

# The line to find (Run 004)
SEARCH_LINE = (
    "| 004 | 2026-01-09 | "
    "[View](https://wandb.ai/cataluna84/generative-deep-learning/runs/6fjb8dca) | "
    "512 | 4500/10000 | 5e-5 | ⚠️ Partial | 3.09 | -74.59 | "
    "Terminated early; D/G=0.04, Clip%=0.1% (healthy) |\\n"
)


def update_notebook():
    """Update the notebook to add Run 005 to the experiment log."""
    print(f"Reading notebook: {NOTEBOOK_PATH}")

    # Read the notebook
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find and update the experiment log cell
    updated = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"]
            # Check if this is the experiment log cell
            if any("Master Experiment Log" in line for line in source):
                print("Found Master Experiment Log cell")
                # Look for Run 004 and add Run 005 after it
                new_source = []
                for line in source:
                    new_source.append(line)
                    if SEARCH_LINE in line:
                        print("Found Run 004 entry, adding Run 005")
                        new_source.append(f"                \"{NEW_ENTRY}\",\n")
                        updated = True
                if updated:
                    cell["source"] = new_source
                    break

    if not updated:
        # Try a different approach - find the line containing Run 004's partial info
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                source = cell["source"]
                new_source = []
                for i, line in enumerate(source):
                    new_source.append(line)
                    # Check for Run 004 entry
                    if "| 004 |" in line and "6fjb8dca" in line:
                        print("Found Run 004 entry (alternative method)")
                        # Add the new entry line
                        new_entry_line = (
                            "| 005 | 2026-01-12 | "
                            "[View](https://wandb.ai/cataluna84/generative-deep-learning/runs/1ex40gx6) | "
                            "512 | 10000 | 5e-5 | ⚠️ Unstable | -0.08 | -1.73 | "
                            "Full 10K epochs; 3/5 indicators passed; mode collapse detected |\\n"
                        )
                        new_source.append(new_entry_line)
                        updated = True
                if updated:
                    cell["source"] = new_source
                    break

    if updated:
        # Write the updated notebook
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=4)
        print("✅ Notebook updated successfully!")
        print(f"   Added Run 005 to the Master Experiment Log")
    else:
        print("❌ Could not find the experiment log entry to update")
        print("   Manual update may be required")
        sys.exit(1)


if __name__ == "__main__":
    update_notebook()
