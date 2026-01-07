#!/usr/bin/env python3
"""
Add experiment logging markdown cell to WGAN CIFAR notebook.

This script adds a comprehensive experiment logging section at the end of the
04_02_wgan_cifar_train.ipynb notebook, before the kernel restart cell.

The logging section includes:
1. Experiment Log Table - Master table for tracking all experiments
2. Current Run Configuration - Detailed hyperparameters
3. Phase-wise Metrics Analysis - Training progression at key checkpoints
4. Stability Analysis - Interpretation of training dynamics
5. Experiment Notes & Changelog - Free-form observations

Usage:
    python scripts/add_wgan_experiment_logging.py
"""

import json
import os


# =============================================================================
# EXPERIMENT LOGGING MARKDOWN CONTENT
# =============================================================================
EXPERIMENT_LOGGING_MARKDOWN = """## Experiment Log

This section tracks all training experiments for systematic comparison and reproducibility.

### Master Experiment Log

| Run | Date | W&B URL | Batch Size | Epochs | LR (Critic) | LR (Gen) | LR Scheduler | Early Stop | Callbacks | Final D Loss | Final G Loss | Stability | Notes |
|-----|------|---------|------------|--------|-------------|----------|--------------|------------|-----------|--------------|--------------|-----------|-------|
| 001 | 2026-01-07 | [View Run](URL_HERE) | 64 | 6000 | 5e-5 | 5e-5 | None | None | [] | 5.35 | -116.4 | ✅ Stable | Baseline run |
| 002 | - | - | - | - | - | - | - | - | - | - | - | - | *Placeholder* |
| 003 | - | - | - | - | - | - | - | - | - | - | - | - | *Placeholder* |

---

## Current Run Configuration

| Category | Parameter | Value |
|----------|-----------|-------|
| **Run Info** | Run ID | 0002_horses |
| | W&B Run URL | [View on W&B](URL_HERE) |
| | Date | 2026-01-07 |
| **Data** | Dataset | CIFAR-10 (Horses) |
| | Training Samples | 6,000 |
| | Image Size | 32×32×3 |
| **Training** | Batch Size | 64 |
| | Epochs | 6000 |
| | Critic Steps (n_critic) | 5 |
| | Clip Threshold | 0.01 |
| **Optimizer** | Type | RMSprop |
| | Critic LR | 0.00005 |
| | Generator LR | 0.00005 |
| **LR Scheduler** | Type | *None* |
| | Decay Rate | - |
| | Patience | - |
| **Early Stopping** | Enabled | *No* |
| | Patience | - |
| **Callbacks** | List | *None* |

---

## Training Progression (Phase-wise Metrics)

| Phase | Epoch Range | D Loss (Start → End) | G Loss (Start → End) | Δ D Loss/epoch | Δ G Loss/epoch | Notes |
|-------|-------------|----------------------|----------------------|----------------|----------------|-------|
| Warm-up | 0-50 | 0.00 → 0.02 | 0.00 → -0.10 | ~0.0004 | ~0.002 | Initial exploration |
| Early | 50-500 | 0.02 → 0.50 | -0.10 → -10 | ~0.001 | ~0.02 | Critic gaining signal |
| Mid | 500-2000 | 0.50 → 2.70 | -10 → -62 | ~0.0015 | ~0.03 | Steady progression |
| Late | 2000-4000 | 2.70 → 4.42 | -62 → -98 | ~0.0009 | ~0.02 | Approaching convergence |
| Final | 4000-6000 | 4.42 → 5.35 | -98 → -116 | ~0.0005 | ~0.01 | Near convergence |

---

## Stability Analysis

### Key Indicators

| Indicator | Status | Observation |
|-----------|--------|-------------|
| Loss Monotonicity | ✅ Good | D and G losses change smoothly without oscillations |
| Real/Fake Balance | ✅ Good | D_loss_real ≈ D_loss_fake throughout training |
| Wasserstein Distance | ✅ Increasing | |G loss| grows steadily (expected behavior) |
| Mode Collapse | ✅ None | No sudden plateaus or repetitive outputs observed |
| Gradient Signal | ✅ Healthy | Critic maintains discrimination ability |

### Interpretation

**Wasserstein Loss Understanding:**
- **D loss = E[critic(real)] - E[critic(fake)]**: Critic maximizes this (high scores for real, low for fake)
- **G loss = -E[critic(fake)]**: Generator minimizes this (wants critic to score fakes highly)

**Expected WGAN Behavior:**
- D loss should be positive and gradually increasing
- G loss magnitude (|G loss|) should increase as generator improves
- Real/Fake discrimination should remain balanced

### Overall Verdict

| Metric | Value |
|--------|-------|
| Training Stability | ✅ **STABLE** |
| Convergence Quality | Good |
| Recommended Action | Continue with current hyperparameters or experiment with LR scheduling |

---

## Experiment Notes & Changelog

### Run 001 (Baseline) - 2026-01-07
- **Configuration**: Default hyperparameters from book
- **Outcome**: Stable training over 6000 epochs
- **Observations**: 
  - No mode collapse observed
  - Generated images show recognizable horse features
  - Smooth monotonic loss progression

### Future Experiments (Planned)
- [ ] Try ReduceLROnPlateau scheduler
- [ ] Test early stopping with patience=500
- [ ] Experiment with higher batch size (128)
- [ ] Compare with WGAN-GP (gradient penalty variant)
- [ ] Test different n_critic values (3, 10)
"""


def add_experiment_logging_cell():
    """Add experiment logging markdown cell to the WGAN CIFAR notebook."""
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

    # Check if the cell already exists
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            source_text = "".join(cell["source"])
            if "## Experiment Log" in source_text:
                print("⚠ Experiment logging cell already exists. Skipping.")
                return False

    # Create the new markdown cell
    experiment_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in EXPERIMENT_LOGGING_MARKDOWN.split("\n")]
    }

    # Find the kernel restart cell (if exists) to insert before it
    insert_index = len(notebook["cells"])
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source_text = "".join(cell["source"])
            if "do_shutdown" in source_text or "restart" in source_text.lower():
                insert_index = i
                break

    # Insert the experiment logging cell
    notebook["cells"].insert(insert_index, experiment_cell)

    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Added experiment logging cell to: {notebook_path}")
    print(f"  Inserted at position: {insert_index}")
    return True


if __name__ == "__main__":
    add_experiment_logging_cell()
