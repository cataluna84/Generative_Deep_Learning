#!/usr/bin/env python3
"""
Refactor training metrics dashboard in WGAN CIFAR notebook.

This script:
1. Removes the 2x3 dashboard cell that was previously added
2. Creates separate markdown + code cells for each training metric
3. Places them after the Wasserstein loss visualization section

Each metric gets its own markdown description and full-size plot.

Usage:
    python scripts/refactor_training_dashboard.py

Author:
    Auto-generated with PEP-8 documentation.
"""

import json
import os


def create_metric_cells():
    """Create separate markdown and code cells for each training metric."""
    # =========================================================================
    # D/G Ratio Plot
    # =========================================================================
    dg_ratio_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## D/G Loss Ratio\n",
            "\n",
            "The D/G loss ratio is a key stability indicator that measures the balance "
            "between the critic and generator during adversarial training.\n",
            "\n",
            "### Theory\n",
            "\n",
            "In a GAN, the discriminator (critic) and generator are locked in a minimax "
            "game. The **D/G ratio** is computed as:\n",
            "\n",
            "$$\\text{D/G Ratio} = \\frac{|D_{loss}|}{|G_{loss}|}$$\n",
            "\n",
            "This ratio tells us which network is \"winning\" the adversarial game:\n",
            "\n",
            "- **Ratio < 0.1**: Generator is strong, critic is providing useful gradients\n",
            "- **Ratio ≈ 1.0**: Balance point - neither network dominates\n",
            "- **Ratio > 1.0**: Critic is dominating, may block gradients to generator\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "A stable WGAN typically shows a D/G ratio that starts moderate and "
            "decreases over training as the generator improves. Sudden spikes may "
            "indicate mode collapse or training instability.\n"
        ]
    }
    
    dg_ratio_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# D/G Loss Ratio Plot\n",
            "# =============================================================================\n",
            "# The D/G ratio measures the balance between discriminator and generator.\n",
            "# Formula: |D_loss| / |G_loss|\n",
            "# A low, stable ratio indicates healthy adversarial dynamics.\n",
            "# =============================================================================\n",
            "\n",
            "fig = plt.figure(figsize=(12, 6))\n",
            "\n",
            "if gan.metrics_history.get('dg_ratio'):\n",
            "    plt.plot(gan.metrics_history['dg_ratio'], color='orange', alpha=0.8, "
            "label='D/G Ratio')\n",
            "    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, "
            "label='Healthy Zone (< 0.1)')\n",
            "\n",
            "plt.xlabel('Epoch', fontsize=12)\n",
            "plt.ylabel('D/G Ratio', fontsize=12)\n",
            "plt.title('WGAN Training: D/G Loss Ratio vs Epoch', fontsize=16)\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "\n",
            "fig.savefig(os.path.join(RUN_FOLDER, 'viz/dg_ratio_vs_epoch.png'), dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved D/G ratio plot to: {os.path.join(RUN_FOLDER, 'viz/dg_ratio_vs_epoch.png')}\")\n"
        ]
    }
    
    # =========================================================================
    # Clip Ratio Plot
    # =========================================================================
    clip_ratio_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Weight Clip Ratio\n",
            "\n",
            "The clip ratio measures the percentage of critic weights that have been "
            "clipped to the boundary threshold. This is a WGAN-specific metric.\n",
            "\n",
            "### Theory\n",
            "\n",
            "The original WGAN paper (Arjovsky et al., 2017) requires the critic to be "
            "a **K-Lipschitz function** to ensure the Wasserstein distance is valid. "
            "Weight clipping enforces this constraint by clamping all weights to "
            "[-c, c] after each gradient update:\n",
            "\n",
            "$$w \\leftarrow \\text{clip}(w, -c, c)$$\n",
            "\n",
            "The **clip ratio** is:\n",
            "\n",
            "$$\\text{Clip Ratio} = \\frac{\\text{\\# weights at } \\pm c}"
            "{\\text{total weights}}$$\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "- **High clip % (>30%)**: Many weights hitting boundary - Lipschitz "
            "constraint is actively enforced, may limit capacity\n",
            "- **Low clip % (<1%)**: Weights have settled within valid range - "
            "constraint satisfied naturally\n",
            "- **Decreasing over time**: Normal behavior as training stabilizes\n"
        ]
    }
    
    clip_ratio_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Weight Clip Ratio Plot\n",
            "# =============================================================================\n",
            "# In WGAN, weight clipping enforces the Lipschitz constraint.\n",
            "# This plot shows % of weights at the clipping boundary (±CLIP_THRESHOLD).\n",
            "# Lower values indicate the constraint is naturally satisfied.\n",
            "# =============================================================================\n",
            "\n",
            "fig = plt.figure(figsize=(12, 6))\n",
            "\n",
            "if gan.metrics_history.get('clip_ratio'):\n",
            "    clip_pct = [c * 100 for c in gan.metrics_history['clip_ratio']]\n",
            "    plt.plot(clip_pct, color='purple', alpha=0.8)\n",
            "\n",
            "plt.xlabel('Epoch', fontsize=12)\n",
            "plt.ylabel('Clip Ratio (%)', fontsize=12)\n",
            "plt.title('WGAN Training: Weight Clip Ratio vs Epoch', fontsize=16)\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "\n",
            "fig.savefig(os.path.join(RUN_FOLDER, 'viz/clip_ratio_vs_epoch.png'), dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved clip ratio plot to: {os.path.join(RUN_FOLDER, 'viz/clip_ratio_vs_epoch.png')}\")\n"
        ]
    }
    
    # =========================================================================
    # Weight Statistics Plot
    # =========================================================================
    weight_stats_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Weight Statistics\n",
            "\n",
            "Weight statistics (mean and standard deviation) provide insight into the "
            "internal state of both networks during training.\n",
            "\n",
            "### Theory\n",
            "\n",
            "Neural network weights should maintain certain statistical properties "
            "for healthy training:\n",
            "\n",
            "- **Mean (μ)**: Should remain close to 0 for most architectures\n",
            "- **Standard deviation (σ)**: Should be stable, typically 0.01-0.1\n",
            "\n",
            "In WGAN specifically:\n",
            "- **Critic σ**: Bounded by the clipping constraint (max = CLIP_THRESHOLD)\n",
            "- **Generator σ**: Unbounded, typically larger than critic\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "- **Stable σ**: Healthy training\n",
            "- **Exploding σ**: Gradient explosion, training failure\n",
            "- **Vanishing σ**: Dead neurons, gradient vanishing\n",
            "- **Sudden changes**: Mode collapse or distribution shift\n"
        ]
    }
    
    weight_stats_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Weight Statistics Plot\n",
            "# =============================================================================\n",
            "# Monitoring weight statistics helps detect training instabilities.\n",
            "# Critic weights are constrained by clipping; generator weights are free.\n",
            "# Stable std values indicate healthy training dynamics.\n",
            "# =============================================================================\n",
            "\n",
            "fig = plt.figure(figsize=(12, 6))\n",
            "\n",
            "if gan.metrics_history.get('critic_weight_std'):\n",
            "    plt.plot(gan.metrics_history['critic_weight_std'], label='Critic σ', "
            "color='blue', alpha=0.8)\n",
            "    plt.plot(gan.metrics_history['generator_weight_std'], label='Generator σ', "
            "color='green', alpha=0.8)\n",
            "\n",
            "plt.xlabel('Epoch', fontsize=12)\n",
            "plt.ylabel('Weight Std', fontsize=12)\n",
            "plt.title('WGAN Training: Weight Standard Deviation vs Epoch', fontsize=16)\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "\n",
            "fig.savefig(os.path.join(RUN_FOLDER, 'viz/weight_stats_vs_epoch.png'), dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved weight stats plot to: {os.path.join(RUN_FOLDER, 'viz/weight_stats_vs_epoch.png')}\")\n"
        ]
    }
    
    # =========================================================================
    # Epoch Timing Plot
    # =========================================================================
    epoch_time_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Epoch Timing\n",
            "\n",
            "Epoch timing tracks the wall-clock duration of each training epoch, "
            "providing insight into computational efficiency.\n",
            "\n",
            "### Theory\n",
            "\n",
            "Each WGAN epoch involves:\n",
            "1. **n_critic** critic updates (forward + backward + weight clip)\n",
            "2. **1** generator update (forward + backward)\n",
            "3. Metrics computation (weight stats, stability indicators)\n",
            "4. Quality metrics (FID/IS) every 100 epochs\n",
            "\n",
            "Expected timing breakdown:\n",
            "- Base epoch: ~1-2 seconds (depends on batch size, GPU)\n",
            "- Quality metric epochs: ~3-5 seconds (FID computation is expensive)\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "- **Consistent times**: Stable GPU utilization\n",
            "- **Periodic spikes**: Quality metric computation (expected)\n",
            "- **Increasing trend**: Memory pressure, thermal throttling\n",
            "- **Large variance**: System resource contention\n"
        ]
    }
    
    epoch_time_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Epoch Timing Plot\n",
            "# =============================================================================\n",
            "# This plot shows how long each epoch took to complete.\n",
            "# Spikes every 100 epochs are expected (FID/IS computation).\n",
            "# Consistent timing indicates stable GPU utilization.\n",
            "# =============================================================================\n",
            "\n",
            "fig = plt.figure(figsize=(12, 6))\n",
            "\n",
            "if gan.metrics_history.get('epoch_time'):\n",
            "    plt.plot(gan.metrics_history['epoch_time'], color='brown', alpha=0.6)\n",
            "    avg_time = np.mean(gan.metrics_history['epoch_time'])\n",
            "    plt.axhline(y=avg_time, color='red', linestyle='--', "
            "label=f'Average: {avg_time:.2f}s')\n",
            "    plt.legend()\n",
            "\n",
            "plt.xlabel('Epoch', fontsize=12)\n",
            "plt.ylabel('Time (seconds)', fontsize=12)\n",
            "plt.title('WGAN Training: Epoch Duration vs Epoch', fontsize=16)\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "\n",
            "fig.savefig(os.path.join(RUN_FOLDER, 'viz/epoch_time_vs_epoch.png'), dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved epoch time plot to: {os.path.join(RUN_FOLDER, 'viz/epoch_time_vs_epoch.png')}\")\n"
        ]
    }
    
    # =========================================================================
    # Loss Variance Plot
    # =========================================================================
    loss_variance_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Loss Variance\n",
            "\n",
            "Loss variance measures the variability of the discriminator loss over a "
            "rolling window, indicating training stability.\n",
            "\n",
            "### Theory\n",
            "\n",
            "The rolling variance is computed over the last 100 epochs:\n",
            "\n",
            "$$\\text{Variance} = \\frac{1}{N}\\sum_{i=1}^{N}(D_{loss,i} - \\bar{D}_{loss})^2$$\n",
            "\n",
            "Loss variance is a proxy for training stability:\n",
            "\n",
            "- **High variance**: Large oscillations in loss - unstable training\n",
            "- **Low variance**: Consistent losses - stable training or convergence\n",
            "- **Decreasing variance**: Training is stabilizing (expected behavior)\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "In a healthy WGAN training run:\n",
            "1. Early epochs show higher variance (learning phase)\n",
            "2. Variance decreases as the critic finds stable gradients\n",
            "3. Late training shows very low variance (convergence)\n",
            "\n",
            "Sudden spikes in variance may indicate mode collapse or distribution shift.\n"
        ]
    }
    
    loss_variance_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Loss Variance Plot\n",
            "# =============================================================================\n",
            "# Rolling variance of D loss over 100 epochs.\n",
            "# High variance = unstable training; Low variance = stable/converged.\n",
            "# Decreasing variance over time is expected in healthy training.\n",
            "# =============================================================================\n",
            "\n",
            "fig = plt.figure(figsize=(12, 6))\n",
            "\n",
            "# Calculate rolling variance\n",
            "if gan.d_losses:\n",
            "    d_losses = [d[0] for d in gan.d_losses]\n",
            "    window = 100\n",
            "    variances = []\n",
            "    for i in range(len(d_losses)):\n",
            "        start = max(0, i - window + 1)\n",
            "        variances.append(np.var(d_losses[start:i+1]))\n",
            "    plt.plot(variances, color='teal', alpha=0.8)\n",
            "\n",
            "plt.xlabel('Epoch', fontsize=12)\n",
            "plt.ylabel('Loss Variance', fontsize=12)\n",
            "plt.title('WGAN Training: D Loss Variance (Rolling 100-epoch) vs Epoch', fontsize=16)\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "\n",
            "fig.savefig(os.path.join(RUN_FOLDER, 'viz/loss_variance_vs_epoch.png'), dpi=300)\n",
            "plt.show()\n",
            "print(f\"Saved loss variance plot to: {os.path.join(RUN_FOLDER, 'viz/loss_variance_vs_epoch.png')}\")\n"
        ]
    }
    
    return [
        dg_ratio_md, dg_ratio_code,
        clip_ratio_md, clip_ratio_code,
        weight_stats_md, weight_stats_code,
        epoch_time_md, epoch_time_code,
        loss_variance_md, loss_variance_code
    ]


def update_notebook():
    """Update notebook with separate metric cells."""
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
    # Remove the old dashboard cell (TRAINING METRICS DASHBOARD)
    # =========================================================================
    cells_to_remove = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "TRAINING METRICS DASHBOARD" in source:
                cells_to_remove.append(i)
                print(f"✓ Found old dashboard cell at index {i} - will remove")
    
    # Remove in reverse order to maintain indices
    for i in reversed(cells_to_remove):
        del notebook["cells"][i]
        print(f"✓ Removed dashboard cell at index {i}")

    # =========================================================================
    # Find the Wasserstein loss code cell and insert new cells after it
    # =========================================================================
    insert_index = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "Wasserstein Loss Plot" in source and "wassersteinloss_vs_batch.png" in source:
                insert_index = i + 1
                print(f"✓ Found Wasserstein loss cell at index {i}")
                break
    
    if insert_index is None:
        print("⚠ Could not find Wasserstein loss cell")
        return

    # =========================================================================
    # Insert new metric cells
    # =========================================================================
    new_cells = create_metric_cells()
    for j, cell in enumerate(reversed(new_cells)):
        notebook["cells"].insert(insert_index, cell)
    
    print(f"✓ Inserted {len(new_cells)} new cells after Wasserstein loss plot")

    # =========================================================================
    # Save notebook
    # =========================================================================
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved: {notebook_path}")


if __name__ == "__main__":
    update_notebook()
