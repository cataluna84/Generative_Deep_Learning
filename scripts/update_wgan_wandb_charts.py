#!/usr/bin/env python3
"""
Update WGAN CIFAR notebook with W&B chart configuration.

This script:
1. Updates the W&B init cell to call define_wgan_charts()
2. Adds post-training visualization cells for stability analysis

Usage:
    python scripts/update_wgan_wandb_charts.py

Author:
    Auto-generated with PEP-8 documentation.
"""

import json
import os


def update_notebook():
    """Update notebook with W&B chart configuration and visualizations."""
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

    # =========================================================================
    # Find and update W&B initialization cell
    # =========================================================================
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # Look for W&B init cell
            if "wandb.init(" in source and "define_wgan_charts" not in source:
                # Add define_wgan_charts() import and call
                new_source = []
                for line in cell["source"]:
                    new_source.append(line)
                    # After the wandb.init() block, add chart definition
                    if "run = wandb.init(" in line or "wandb.init(" in line:
                        pass  # Continue until we find the closing of init
                
                # Find where the init block ends and add the chart definition
                updated_source = []
                found_init = False
                added_charts = False
                
                for line in cell["source"]:
                    updated_source.append(line)
                    if "wandb.init(" in line:
                        found_init = True
                    if found_init and ")" in line and not added_charts:
                        # After closing paren of wandb.init, add chart config
                        updated_source.append("\n")
                        updated_source.append("# -----------------------------------------------------------------------------\n")
                        updated_source.append("# Configure W&B Custom Charts\n")
                        updated_source.append("# -----------------------------------------------------------------------------\n")
                        updated_source.append("# Organize metrics into logical groups for better visualization.\n")
                        updated_source.append("from utils.wandb_utils import define_wgan_charts\n")
                        updated_source.append("define_wgan_charts()\n")
                        added_charts = True
                        found_init = False
                
                if added_charts:
                    cell["source"] = updated_source
                    print(f"✓ Updated W&B init cell (cell {i})")
                break

    # =========================================================================
    # Add post-training visualization cell after training cell
    # =========================================================================
    visualization_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# TRAINING METRICS DASHBOARD\n",
            "# =============================================================================\n",
            "# Visualize all training metrics collected during training.\n",
            "# This cell creates a comprehensive dashboard of training dynamics.\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "# Create figure with subplots\n",
            "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
            "fig.suptitle('WGAN Training Metrics Dashboard', fontsize=16, fontweight='bold')\n",
            "\n",
            "epochs = range(len(gan.d_losses))\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 1. Loss Curves\n",
            "# -----------------------------------------------------------------------------\n",
            "ax1 = axes[0, 0]\n",
            "d_losses = [d[0] for d in gan.d_losses]\n",
            "ax1.plot(epochs, d_losses, label='D Loss', alpha=0.8)\n",
            "ax1.plot(epochs, gan.g_losses, label='G Loss', alpha=0.8)\n",
            "ax1.set_xlabel('Epoch')\n",
            "ax1.set_ylabel('Loss')\n",
            "ax1.set_title('Discriminator vs Generator Loss')\n",
            "ax1.legend()\n",
            "ax1.grid(True, alpha=0.3)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 2. Wasserstein Distance\n",
            "# -----------------------------------------------------------------------------\n",
            "ax2 = axes[0, 1]\n",
            "if gan.metrics_history.get('wasserstein_dist'):\n",
            "    ax2.plot(gan.metrics_history['wasserstein_dist'], color='green')\n",
            "else:\n",
            "    w_dist = [abs(g) for g in gan.g_losses]\n",
            "    ax2.plot(w_dist, color='green')\n",
            "ax2.set_xlabel('Epoch')\n",
            "ax2.set_ylabel('W-Distance')\n",
            "ax2.set_title('Wasserstein Distance (Higher = Learning)')\n",
            "ax2.grid(True, alpha=0.3)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 3. D/G Ratio (Stability Indicator)\n",
            "# -----------------------------------------------------------------------------\n",
            "ax3 = axes[0, 2]\n",
            "if gan.metrics_history.get('dg_ratio'):\n",
            "    ax3.plot(gan.metrics_history['dg_ratio'], color='orange')\n",
            "    ax3.axhline(y=0.1, color='red', linestyle='--', label='Healthy Zone', alpha=0.5)\n",
            "ax3.set_xlabel('Epoch')\n",
            "ax3.set_ylabel('D/G Ratio')\n",
            "ax3.set_title('D/G Loss Ratio (Lower = More Stable)')\n",
            "ax3.grid(True, alpha=0.3)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 4. Clip Ratio (WGAN Specific)\n",
            "# -----------------------------------------------------------------------------\n",
            "ax4 = axes[1, 0]\n",
            "if gan.metrics_history.get('clip_ratio'):\n",
            "    clip_pct = [c * 100 for c in gan.metrics_history['clip_ratio']]\n",
            "    ax4.plot(clip_pct, color='purple')\n",
            "ax4.set_xlabel('Epoch')\n",
            "ax4.set_ylabel('Clip %')\n",
            "ax4.set_title('Weight Clip Ratio (Lower = Settled)')\n",
            "ax4.grid(True, alpha=0.3)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 5. Weight Statistics\n",
            "# -----------------------------------------------------------------------------\n",
            "ax5 = axes[1, 1]\n",
            "if gan.metrics_history.get('critic_weight_std'):\n",
            "    ax5.plot(gan.metrics_history['critic_weight_std'], label='Critic σ', alpha=0.8)\n",
            "    ax5.plot(gan.metrics_history['generator_weight_std'], label='Gen σ', alpha=0.8)\n",
            "ax5.set_xlabel('Epoch')\n",
            "ax5.set_ylabel('Weight Std')\n",
            "ax5.set_title('Weight Standard Deviation')\n",
            "ax5.legend()\n",
            "ax5.grid(True, alpha=0.3)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 6. Epoch Timing\n",
            "# -----------------------------------------------------------------------------\n",
            "ax6 = axes[1, 2]\n",
            "if gan.metrics_history.get('epoch_time'):\n",
            "    ax6.plot(gan.metrics_history['epoch_time'], color='brown', alpha=0.6)\n",
            "    avg_time = np.mean(gan.metrics_history['epoch_time'])\n",
            "    ax6.axhline(y=avg_time, color='red', linestyle='--', label=f'Avg: {avg_time:.2f}s')\n",
            "    ax6.legend()\n",
            "ax6.set_xlabel('Epoch')\n",
            "ax6.set_ylabel('Time (s)')\n",
            "ax6.set_title('Epoch Duration')\n",
            "ax6.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(RUN_FOLDER, 'viz/training_dashboard.png'), dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Log dashboard to W&B\n",
            "if wandb.run is not None:\n",
            "    wandb.log({'training_dashboard': wandb.Image(fig)})\n",
            "    print('✓ Training dashboard logged to W&B')\n"
        ]
    }

    # Find the training cell and insert visualization after it
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "gan.train(" in source and "TRAINING EXECUTION" in source:
                # Insert visualization cell after training cell
                notebook["cells"].insert(i + 1, visualization_cell)
                print(f"✓ Added training dashboard cell after cell {i}")
                break

    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved notebook: {notebook_path}")


if __name__ == "__main__":
    update_notebook()
