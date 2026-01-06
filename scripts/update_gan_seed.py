#!/usr/bin/env python3
"""
Update GAN Notebook Seed for Next Triage Experiment

Changes SEED from 42 to 123 and documents the failed seed.

Run from project root:
    python scripts/update_gan_seed.py
"""

import json
from pathlib import Path


def main():
    """Update the SEED value in the GAN notebook."""
    notebook_path = Path(__file__).parent.parent / "v1" / "notebooks" / "04_01_gan_camel_train.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"✓ Loaded notebook: {notebook_path}")
    
    # Find and update the seed control cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'SEED = 42' in source and 'SEED CONTROL' in source:
                new_source = [
                    "# =============================================================================\n",
                    "# SEED CONTROL FOR REPRODUCIBILITY\n",
                    "# =============================================================================\n",
                    "# Control all random seeds for reproducible experiments.\n",
                    "# Change SEED value to test different weight initializations.\n",
                    "#\n",
                    "# Experiment Design:\n",
                    "#   - Run 2-3 times with same seed: verify reproducibility\n",
                    "#   - Try different seeds with same batch size: test init sensitivity\n",
                    "#   - Try different batch sizes with same seed: test batch size sensitivity\n",
                    "#\n",
                    "# Triage Log (BATCH_SIZE=1024):\n",
                    "#   - SEED=42:  MODE COLLAPSE (D acc stuck at 51.5%, D loss 0.715)\n",
                    "#   - SEED=123: Testing...\n",
                    "\n",
                    "SEED = 123  # Testing different initialization\n",
                    "\n",
                    "random.seed(SEED)\n",
                    "np.random.seed(SEED)\n",
                    "tf.random.set_seed(SEED)\n",
                    "\n",
                    "# Additional GPU determinism settings\n",
                    "os.environ['TF_DETERMINISTIC_OPS'] = '1'\n",
                    "os.environ['PYTHONHASHSEED'] = str(SEED)\n",
                    "\n",
                    "print(f\"✓ Random seed set to {SEED}\")\n",
                    "print(f\"  • Python random: seeded\")\n",
                    "print(f\"  • NumPy: seeded\")\n",
                    "print(f\"  • TensorFlow: seeded\")\n",
                    "print(f\"  • TF_DETERMINISTIC_OPS: enabled\")"
                ]
                nb['cells'][i]['source'] = new_source
                print("✓ Updated SEED: 42 → 123")
                print("✓ Added triage log documenting SEED=42 failure")
                break
    
    # Also update W&B config to include seed in run name
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'wandb.init(' in source and 'name=f"gan-{DATA_NAME}-bs{BATCH_SIZE}"' in source:
                # Update the run name to include seed
                new_source = source.replace(
                    'name=f"gan-{DATA_NAME}-bs{BATCH_SIZE}"',
                    'name=f"gan-{DATA_NAME}-bs{BATCH_SIZE}-seed{SEED}"'
                )
                # Also add seed to config
                new_source = new_source.replace(
                    '"batch_size": BATCH_SIZE,',
                    '"batch_size": BATCH_SIZE,\n        "seed": SEED,'
                )
                nb['cells'][i]['source'] = new_source.split('\n')
                nb['cells'][i]['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
                print("✓ Updated W&B config: added seed to run name and config")
                break
    
    # Clear outputs from previous run
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    print("✓ Cleared all cell outputs for fresh run")
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✓ Notebook saved: {notebook_path}")
    print("\n" + "=" * 60)
    print("READY FOR NEXT EXPERIMENT")
    print("=" * 60)
    print("  SEED: 123")
    print("  BATCH_SIZE: 1024")
    print("  EPOCHS: 1500")
    print("\nRun the notebook to test if SEED=123 avoids mode collapse.")


if __name__ == "__main__":
    main()
