#!/usr/bin/env python3
"""
Configure GAN Notebook for Batch Size Triage Experiments

This script modifies 04_01_gan_camel_train.ipynb to:
1. Update header documentation (Dynamic -> Fixed Configuration)
2. Add seed control code for reproducibility
3. Replace dynamic batch size with hardcoded BATCH_SIZE=1024
4. Clean up unused imports

Run from project root:
    python scripts/configure_gan_triage.py
"""

import json
import sys
from pathlib import Path


def main():
    """Modify the GAN notebook for triage experiments."""
    # Path to notebook
    notebook_path = Path(__file__).parent.parent / "v1" / "notebooks" / "04_01_gan_camel_train.ipynb"
    
    if not notebook_path.exists():
        print(f"✗ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"✓ Loaded notebook: {notebook_path}")
    
    # Track changes
    changes = []
    
    # =========================================================================
    # CHANGE 1: Update header documentation
    # =========================================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '# GAN Training - Camel Dataset' in source and 'Dynamic Configuration' in source:
                new_source = [
                    "# GAN Training - Camel Dataset\n",
                    "\n",
                    "This notebook trains a Generative Adversarial Network (GAN) on the Camel\n",
                    "dataset (Quick, Draw!) to generate hand-drawn camel sketches.\n",
                    "\n",
                    "## Features\n",
                    "\n",
                    "- **Fixed Configuration**: BATCH_SIZE=1024, EPOCHS=1500 for triage experiments\n",
                    "- **Seed Control**: Reproducible training with configurable random seed\n",
                    "- **W&B Integration**: Full experiment tracking with Weights & Biases\n",
                    "- **LR Scheduling**: Step decay learning rate for stable training\n",
                    "- **Enhanced Visualization**: Loss, accuracy, and LR history plots\n",
                    "\n",
                    "## Architecture\n",
                    "\n",
                    "- **Discriminator**: 4-layer CNN with strided convolutions\n",
                    "- **Generator**: 4-layer deconvolution network with upsampling\n",
                    "\n",
                    "## References\n",
                    "\n",
                    "- Goodfellow et al. \"Generative Adversarial Networks\" (2014)\n",
                    "- Chapter 4 of \"Generative Deep Learning\" book"
                ]
                nb['cells'][i]['source'] = new_source
                changes.append("Updated header: Dynamic Configuration → Fixed Configuration")
                break
    
    # =========================================================================
    # CHANGE 2: Update imports (remove unused dynamic batch size imports)
    # =========================================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'find_optimal_batch_size' in source and 'from utils.gpu_utils' in source:
                new_source = [
                    "# =============================================================================\n",
                    "# IMPORTS\n",
                    "# =============================================================================\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Path Configuration\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import sys\n",
                    "sys.path.insert(0, '..')      # For v1/src modules\n",
                    "sys.path.insert(0, '../..')   # For project root utils/\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Standard Library\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import os\n",
                    "import random\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Third-Party Libraries\n",
                    "# -----------------------------------------------------------------------------\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import wandb\n",
                    "\n",
                    "# -----------------------------------------------------------------------------\n",
                    "# Project Modules\n",
                    "# -----------------------------------------------------------------------------\n",
                    "from src.models.GAN import GAN\n",
                    "from src.utils.loaders import load_safari\n",
                    "\n",
                    "# GPU utilities for VRAM info and config display\n",
                    "from utils.gpu_utils import (\n",
                    "    get_gpu_vram_gb,\n",
                    "    print_training_config\n",
                    ")"
                ]
                nb['cells'][i]['source'] = new_source
                changes.append("Cleaned up imports: removed find_optimal_batch_size, calculate_adjusted_epochs")
                break
    
    # =========================================================================
    # CHANGE 3: Add seed control cell after GPU setup
    # =========================================================================
    seed_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Seed Control for Reproducibility\n",
            "\n",
            "Set random seeds to ensure reproducible experiments. Change SEED to test\n",
            "different weight initializations."
        ]
    }
    
    seed_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
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
            "\n",
            "SEED = 42  # Change this for different experiments\n",
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
    }
    
    # Find the Imports cell and insert seed control after it
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## Global Configuration - Part 1' in source:
                # Insert before Global Configuration Part 1
                nb['cells'].insert(i, seed_code_cell)
                nb['cells'].insert(i, seed_cell)
                changes.append("Added seed control markdown and code cells")
                break
    
    # =========================================================================
    # CHANGE 4: Replace dynamic batch size cell with fixed configuration
    # =========================================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## Global Configuration - Part 2: Dynamic Batch Size' in source:
                # Update the markdown cell
                nb['cells'][i]['source'] = [
                    "## Global Configuration - Part 2: Fixed Batch Size\n",
                    "\n",
                    "Fixed batch size for triage experiments. No dynamic allocation."
                ]
                changes.append("Updated Part 2 header: Dynamic → Fixed Batch Size")
                break
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'find_optimal_batch_size' in source and 'BATCH_SIZE = 1024' in source:
                new_source = [
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# GLOBAL CONFIGURATION - PART 2: FIXED BATCH SIZE\n",
                    "# ═══════════════════════════════════════════════════════════════════\n",
                    "# Fixed batch size for triage experiments (no dynamic allocation).\n",
                    "# This allows controlled experiments to isolate batch size vs initialization.\n",
                    "\n",
                    "# Fixed training configuration\n",
                    "BATCH_SIZE = 1024  # Fixed for triage experiments\n",
                    "EPOCHS = 1500      # Scaled from reference: 6000 * (256/1024) = 1500\n",
                    "\n",
                    "# Checkpoint frequency (scaled proportionally)\n",
                    "PRINT_EVERY_N_BATCHES = 13  # Scaled from reference: 50 * (256/1024) ≈ 13\n",
                    "\n",
                    "# Get VRAM for logging purposes only\n",
                    "GPU_VRAM_GB = get_gpu_vram_gb()\n",
                    "\n",
                    "# Print configuration summary\n",
                    "print_training_config(\n",
                    "    batch_size=BATCH_SIZE,\n",
                    "    epochs=EPOCHS,\n",
                    "    model_params=gan.generator.count_params() + gan.discriminator.count_params(),\n",
                    "    reference_batch=REFERENCE_BATCH_SIZE,\n",
                    "    reference_epochs=REFERENCE_EPOCHS,\n",
                    "    vram_gb=GPU_VRAM_GB\n",
                    ")"
                ]
                nb['cells'][i]['source'] = new_source
                changes.append("Replaced dynamic batch size logic with fixed BATCH_SIZE=1024")
                break
    
    # =========================================================================
    # Save modified notebook
    # =========================================================================
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✓ Notebook saved: {notebook_path}")
    print(f"\nChanges made ({len(changes)}):")
    for change in changes:
        print(f"  • {change}")
    
    print("\n" + "=" * 60)
    print("TRIAGE EXPERIMENT SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Open the notebook and verify changes")
    print("  2. Run with SEED=42 and note the result (success/collapse)")
    print("  3. Run again with SEED=42 to verify reproducibility")
    print("  4. Try different seeds (123, 456) to test init sensitivity")
    print("")


if __name__ == "__main__":
    main()
