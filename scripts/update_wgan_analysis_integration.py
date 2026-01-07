#!/usr/bin/env python3
"""
Update WGAN notebook with simplified experiment log and analysis integration.

This script:
1. Replaces the detailed experiment log with a simplified master table
2. Adds analysis and report generation code cells
3. Documents what's logged to W&B

Usage:
    python scripts/update_wgan_analysis_integration.py
"""

import json
import os

# =============================================================================
# SIMPLIFIED EXPERIMENT LOG (MARKDOWN)
# =============================================================================
SIMPLIFIED_EXPERIMENT_LOG = """## Experiment Log

This section tracks all training experiments. **Full analysis details are logged to W&B and saved as markdown reports.**

### Master Experiment Log

| Run | Date | W&B URL | Batch Size | Epochs | LR | Stability | Final D Loss | Final G Loss | Notes |
|-----|------|---------|------------|--------|-----|-----------|--------------|--------------|-------|
| 001 | 2026-01-07 | [View](https://wandb.ai/cataluna84/generative-deep-learning/runs/x5ln97by) | 512 | 6000 | 5e-5 | ✅ Stable | 5.35 | -116.4 | Baseline |
| 002 | - | - | - | - | - | - | - | - | *Placeholder* |
| 003 | - | - | - | - | - | - | - | - | *Placeholder* |

### Comparison Notes

*Add observations comparing runs here after experiments.*

---

## What's Logged to W&B

Each run automatically logs the following (click W&B URL to view):

| Category | Items Logged |
|----------|--------------|
| **Config** | batch_size, epochs, lr_critic, lr_gen, n_critic, clip_threshold, z_dim |
| **Metrics** | d_loss, d_loss_real, d_loss_fake, g_loss (per epoch) |
| **Tables** | phase_metrics, stability_indicators |
| **Summary** | training_stability, training_quality, final losses, verdict |
| **Media** | Generated images, loss plots |

### Per-Run Analysis Reports

Each run generates `analysis_report.md` in the run folder containing:
- Training verdict (stability score)
- Full configuration table
- Phase-wise metrics breakdown
- Stability indicators with observations
"""

# =============================================================================
# ANALYSIS INTEGRATION CODE CELL
# =============================================================================
ANALYSIS_CODE_CELL = """# =============================================================================
# POST-TRAINING ANALYSIS
# =============================================================================
# Run automated stability analysis and log to W&B + markdown report.

from utils.stability_analysis import analyze_training_run
from utils.report_generator import generate_run_report
from utils.wandb_utils import log_training_report

# -----------------------------------------------------------------------------
# Run Automated Stability Analysis
# -----------------------------------------------------------------------------
# Analyze training dynamics from loss curves
analysis = analyze_training_run(gan.d_losses, gan.g_losses)

# Print verdict summary
print("═" * 60)
print("TRAINING ANALYSIS VERDICT")
print("═" * 60)
print(f"Stability:    {analysis['verdict']['stability']}")
print(f"Quality:      {analysis['verdict']['quality']}")
print(f"Score:        {analysis['verdict']['passed']}/{analysis['verdict']['total']} checks passed")
print(f"Recommendation: {analysis['verdict']['recommendation']}")
print("═" * 60)

# Print stability indicators
print("\\nStability Indicators:")
for name, (passed, obs) in analysis['indicators'].items():
    status = "✅" if passed else "❌"
    print(f"  {status} {name.replace('_', ' ').title()}: {obs}")

# -----------------------------------------------------------------------------
# Log Complete Analysis to W&B
# -----------------------------------------------------------------------------
# Build config dict from global constants for logging
config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr_critic": CRITIC_LEARNING_RATE,
    "lr_generator": GENERATOR_LEARNING_RATE,
    "optimizer": OPTIMIZER,
    "z_dim": Z_DIM,
    "n_critic": N_CRITIC,
    "clip_threshold": CLIP_THRESHOLD,
    "input_dim": INPUT_DIM,
    "critic_filters": CRITIC_FILTERS,
    "generator_filters": GENERATOR_FILTERS,
}

# Log to W&B
log_training_report(
    config=config,
    analysis=analysis,
    run_folder=RUN_FOLDER,
    notes="Baseline run with default hyperparameters"
)

# -----------------------------------------------------------------------------
# Generate Markdown Report
# -----------------------------------------------------------------------------
# Save analysis report to run folder
report_path = generate_run_report(
    run_folder=RUN_FOLDER,
    config=config,
    d_losses=gan.d_losses,
    g_losses=gan.g_losses,
    wandb_url=wandb.run.url if wandb.run else None,
    notes="Baseline run with default hyperparameters"
)

print(f"\\n✓ Analysis complete! Report saved to: {report_path}")
"""


def update_notebook():
    """Update notebook with simplified experiment log and analysis cells."""
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

    # Find and replace the experiment log cell
    exp_log_replaced = False
    analysis_cell_exists = False
    
    for i, cell in enumerate(notebook["cells"]):
        # Check for existing experiment log
        if cell["cell_type"] == "markdown":
            source_text = "".join(cell["source"])
            if "## Experiment Log" in source_text:
                # Replace with simplified version
                cell["source"] = [
                    line + "\n" for line in SIMPLIFIED_EXPERIMENT_LOG.split("\n")
                ]
                exp_log_replaced = True
                print(f"✓ Replaced experiment log cell at index {i}")
        
        # Check if analysis cell already exists
        if cell["cell_type"] == "code":
            source_text = "".join(cell["source"])
            if "POST-TRAINING ANALYSIS" in source_text:
                analysis_cell_exists = True

    if not exp_log_replaced:
        print("⚠ Experiment log cell not found!")
        return False

    # Add analysis code cell if it doesn't exist
    if not analysis_cell_exists:
        # Find position after "Log Training Metrics to W&B" cell
        insert_idx = None
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source_text = "".join(cell["source"])
                if "Training metrics logged to W&B" in source_text:
                    insert_idx = i + 1
                    break
        
        if insert_idx is None:
            # Fall back to inserting before experiment log
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "markdown":
                    source_text = "".join(cell["source"])
                    if "## Experiment Log" in source_text:
                        insert_idx = i
                        break
        
        if insert_idx:
            # Create analysis markdown header
            analysis_header_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Post-Training Analysis\n",
                    "\n",
                    "Run automated stability analysis and generate reports.\n",
                    "This cell:\n",
                    "1. Analyzes training dynamics using rules-based stability checks\n",
                    "2. Logs phase metrics and stability indicators to W&B\n",
                    "3. Generates a markdown analysis report in the run folder"
                ]
            }
            
            # Create analysis code cell
            analysis_code_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    line + "\n" for line in ANALYSIS_CODE_CELL.split("\n")
                ]
            }
            
            notebook["cells"].insert(insert_idx, analysis_code_cell)
            notebook["cells"].insert(insert_idx, analysis_header_cell)
            print(f"✓ Added analysis cells at index {insert_idx}")
    else:
        print("⚠ Analysis cell already exists, skipping")

    # Write updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved updated notebook: {notebook_path}")
    return True


if __name__ == "__main__":
    update_notebook()
