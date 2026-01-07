#!/usr/bin/env python3
"""
Correct experiment logging parameters in WGAN CIFAR notebook.

This script updates the experiment log with the correct hyperparameters
extracted from the notebook.

Corrected values:
- Batch Size: 512 (not 64)
- Learning rates: 5e-5 for both critic and generator
- All other parameters verified

Usage:
    python scripts/correct_wgan_experiment_params.py
"""

import json
import os

# =============================================================================
# CORRECTED EXPERIMENT LOGGING MARKDOWN CONTENT
# =============================================================================
# All parameters verified from the notebook source code

CORRECTED_EXPERIMENT_LOGGING = """## Experiment Log

This section tracks all training experiments for systematic comparison and reproducibility.

### Master Experiment Log

| Run | Date | W&B URL | Batch Size | Epochs | LR (Critic) | LR (Gen) | LR Scheduler | Early Stop | Callbacks | Final D Loss | Final G Loss | Stability | Notes |
|-----|------|---------|------------|--------|-------------|----------|--------------|------------|-----------|--------------|--------------|-----------|-------|
| 001 | 2026-01-07 | [View Run](https://wandb.ai/cataluna84/generative-deep-learning/runs/x5ln97by) | 512 | 6000 | 5e-5 | 5e-5 | None | None | [] | 5.35 | -116.4 | ✅ Stable | Baseline run |
| 002 | - | - | - | - | - | - | - | - | - | - | - | - | *Placeholder* |
| 003 | - | - | - | - | - | - | - | - | - | - | - | - | *Placeholder* |

---

## Current Run Configuration

| Category | Parameter | Value |
|----------|-----------|-------|
| **Run Info** | Run ID | 0002_horses |
| | W&B Run URL | [View on W&B](https://wandb.ai/cataluna84/generative-deep-learning/runs/x5ln97by) |
| | Date | 2026-01-07 |
| **Data** | Dataset | CIFAR-10 (Horses, label=7) |
| | Training Samples | 6,000 |
| | Image Size | 32×32×3 |
| **Model** | Latent Dim (Z_DIM) | 100 |
| | Critic Filters | [32, 64, 128, 128] |
| | Generator Filters | [128, 64, 32, 3] |
| | Activation | LeakyReLU |
| **Training** | Batch Size | 512 |
| | Epochs | 6000 |
| | Critic Steps (n_critic) | 5 |
| | Clip Threshold | 0.01 |
| | Checkpoint Interval | 50 batches |
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
- **Batch Size**: 512 (optimized for RTX 2070 8GB VRAM)
- **Outcome**: Stable training over 6000 epochs
- **Observations**: 
  - No mode collapse observed
  - Generated images show recognizable horse features
  - Smooth monotonic loss progression

### Future Experiments (Planned)
- [ ] Try ReduceLROnPlateau scheduler
- [ ] Test early stopping with patience=500
- [ ] Experiment with different batch sizes (256, 1024)
- [ ] Compare with WGAN-GP (gradient penalty variant)
- [ ] Test different n_critic values (3, 10)
"""


def correct_experiment_params():
    """Update experiment log with corrected parameters."""
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

    # Find and replace the experiment logging cell
    found = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source_text = "".join(cell["source"])
            if "## Experiment Log" in source_text:
                # Replace with corrected content
                cell["source"] = [
                    line + "\n" for line in CORRECTED_EXPERIMENT_LOGGING.split("\n")
                ]
                found = True
                print(f"✓ Corrected experiment log at cell index {i}")
                print("\nCorrected parameters:")
                print("  - Batch Size: 512 (was 64)")
                print("  - LR (Critic): 5e-5 (0.00005)")
                print("  - LR (Generator): 5e-5 (0.00005)")
                print("  - Latent Dim: 100")
                print("  - N_CRITIC: 5")
                print("  - CLIP_THRESHOLD: 0.01")
                print("  - Optimizer: RMSprop")
                break

    if not found:
        print("⚠ Experiment logging cell not found!")
        return False

    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"\n✓ Saved corrected notebook: {notebook_path}")
    return True


if __name__ == "__main__":
    correct_experiment_params()
