#!/usr/bin/env python3
"""
Script to standardize 03_04_vae_digits_analysis.ipynb notebook.
Applies all 11 changes from the implementation plan.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "v1", "notebooks", "03_04_vae_digits_analysis.ipynb"
)


def main():
    """Apply all standardization changes to the notebook."""
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Build new cells list
    new_cells = []

    # =========================================================================
    # Cell 1: Enhanced Header
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03_04 VAE Digits Analysis\n",
            "\n",
            "**Chapter 3**: Variational Autoencoders | **Notebook 4 of 6**\n",
            "\n",
            "Analyzes the trained VAE model:\n",
            "- Reconstruction quality visualization\n",
            "- Latent space exploration\n",
            "- Digit generation from latent samples"
        ]
    })
    print("✅ 1. Enhanced header")

    # =========================================================================
    # Cell 2: Imports section header
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Imports"]
    })
    print("✅ 2. Section header: Imports")

    # =========================================================================
    # Cell 3: GPU Setup with error handling
    # =========================================================================
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import tensorflow as tf\n",
            "\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GPU MEMORY CONFIGURATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "gpus = tf.config.list_physical_devices('GPU')\n",
            "if gpus:\n",
            "    for gpu in gpus:\n",
            "        tf.config.experimental.set_memory_growth(gpu, True)\n",
            "    print(f\"GPU(s) available: {[gpu.name for gpu in gpus]}\")\n",
            "else:\n",
            "    print(\"WARNING: No GPU detected, running on CPU\")"
        ]
    })
    print("✅ 3. GPU setup with error handling")

    # =========================================================================
    # Cell 4: PEP 8 Imports
    # =========================================================================
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# IMPORTS\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "# Standard library\n",
            "import os\n",
            "import sys\n",
            "\n",
            "# Path setup for local imports\n",
            "sys.path.insert(0, '../..')\n",
            "sys.path.insert(0, '..')\n",
            "\n",
            "# Third-party\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy.stats import norm\n",
            "\n",
            "# Local imports\n",
            "from src.models.VAE import VariationalAutoencoder\n",
            "from src.utils.loaders import load_mnist, load_model"
        ]
    })
    print("✅ 4. PEP 8 imports (removed duplicate numpy)")

    # =========================================================================
    # Cell 5: Global Configuration
    # =========================================================================
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GLOBAL CONFIGURATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "SECTION = 'vae'\n",
            "RUN_ID = '0002'\n",
            "DATA_NAME = 'digits'\n",
            "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'\n",
            "\n",
            "print(f\"Run folder: {RUN_FOLDER}\")"
        ]
    })
    print("✅ 5. Global config with f-string")

    # =========================================================================
    # Cell 6-7: Load Data
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Data"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load MNIST dataset\n",
            "(x_train, y_train), (x_test, y_test) = load_mnist()\n",
            "print(f\"Training samples: {len(x_train)}, Test samples: {len(x_test)}\")"
        ]
    })
    print("✅ 6-7. Load Data section")

    # =========================================================================
    # Cell 8-9: Load Model
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Model"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load pre-trained VAE model\n",
            "vae = load_model(VariationalAutoencoder, RUN_FOLDER)\n",
            "print(f\"Model loaded from: {RUN_FOLDER}\")"
        ]
    })
    print("✅ 8-9. Load Model section")

    # =========================================================================
    # Cell 10-11: Reconstruction Visualization
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Reconstruction Visualization"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# RECONSTRUCTION VISUALIZATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Encode test images to latent space, then decode to visualize reconstruction.\n",
            "\n",
            "n_to_show = 10\n",
            "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
            "example_images = x_test[example_idx]\n",
            "\n",
            "# Encode to latent space\n",
            "z_points = vae.encoder.predict(example_images)\n",
            "\n",
            "# Decode back to image space\n",
            "reconst_images = vae.decoder.predict(z_points)\n",
            "\n",
            "# Plot original (top row) and reconstruction (bottom row)\n",
            "fig = plt.figure(figsize=(15, 3))\n",
            "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
            "\n",
            "for i in range(n_to_show):\n",
            "    img = example_images[i].squeeze()\n",
            "    ax = fig.add_subplot(2, n_to_show, i + 1)\n",
            "    ax.axis('off')\n",
            "    ax.text(0.5, -0.35, str(np.round(z_points[i], 1)),\n",
            "            fontsize=10, ha='center', transform=ax.transAxes)\n",
            "    ax.imshow(img, cmap='gray_r')\n",
            "\n",
            "for i in range(n_to_show):\n",
            "    img = reconst_images[i].squeeze()\n",
            "    ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)\n",
            "    ax.axis('off')\n",
            "    ax.imshow(img, cmap='gray_r')\n",
            "\n",
            "plt.suptitle('Original (top) vs Reconstruction (bottom)')\n",
            "plt.show()"
        ]
    })
    print("✅ 10-11. Reconstruction Visualization")

    # =========================================================================
    # Cell 12-13: Latent Space Visualization
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Latent Space Visualization"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# LATENT SPACE SCATTER PLOT\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Visualize distribution of test images in 2D latent space.\n",
            "\n",
            "n_to_show = 5000\n",
            "figsize = 12\n",
            "\n",
            "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
            "example_images = x_test[example_idx]\n",
            "example_labels = y_test[example_idx]\n",
            "\n",
            "# Encode to latent space\n",
            "z_points = vae.encoder.predict(example_images)\n",
            "\n",
            "# Store bounds for later use\n",
            "min_x, max_x = min(z_points[:, 0]), max(z_points[:, 0])\n",
            "min_y, max_y = min(z_points[:, 1]), max(z_points[:, 1])\n",
            "\n",
            "# Plot latent distribution\n",
            "plt.figure(figsize=(figsize, figsize))\n",
            "plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)\n",
            "plt.xlabel('z[0]')\n",
            "plt.ylabel('z[1]')\n",
            "plt.title('Latent Space Distribution')\n",
            "plt.show()"
        ]
    })
    print("✅ 12-13. Latent Space Visualization")

    # =========================================================================
    # Cell 14-15: Generated Samples (Random)
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Generated Samples"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# RANDOM SAMPLE GENERATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Sample random points from N(0,1) and decode to generate new digits.\n",
            "\n",
            "figsize = 8\n",
            "grid_size = 15\n",
            "grid_depth = 2\n",
            "\n",
            "# Show latent distribution with sampled points\n",
            "plt.figure(figsize=(figsize, figsize))\n",
            "plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)\n",
            "\n",
            "# Sample from standard normal (the VAE prior)\n",
            "x = np.random.normal(size=grid_size * grid_depth)\n",
            "y = np.random.normal(size=grid_size * grid_depth)\n",
            "z_grid = np.array(list(zip(x, y)))\n",
            "\n",
            "# Generate images from latent samples\n",
            "reconst = vae.decoder.predict(z_grid)\n",
            "\n",
            "# Plot sampled points in red\n",
            "plt.scatter(z_grid[:, 0], z_grid[:, 1], c='red', alpha=1, s=20)\n",
            "plt.title('Latent Space with Sampled Points (red)')\n",
            "plt.show()\n",
            "\n",
            "# Display generated images\n",
            "fig = plt.figure(figsize=(15, grid_depth))\n",
            "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
            "\n",
            "for i in range(grid_size * grid_depth):\n",
            "    ax = fig.add_subplot(grid_depth, grid_size, i + 1)\n",
            "    ax.axis('off')\n",
            "    ax.text(0.5, -0.35, str(np.round(z_grid[i], 1)),\n",
            "            fontsize=8, ha='center', transform=ax.transAxes)\n",
            "    ax.imshow(reconst[i, :, :, 0], cmap='Greys')\n",
            "\n",
            "plt.suptitle('Generated Digits from Random Latent Samples')\n",
            "plt.show()"
        ]
    })
    print("✅ 14-15. Generated Samples (Random)")

    # =========================================================================
    # Cell 16: Colored Latent Space
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Colored Latent Space by Digit"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# COLORED LATENT SPACE BY DIGIT LABEL\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Color-code latent space by digit class to visualize clustering.\n",
            "# Also show CDF transform for comparison.\n",
            "\n",
            "n_to_show = 5000\n",
            "fig_height = 7\n",
            "fig_width = 15\n",
            "\n",
            "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
            "example_images = x_test[example_idx]\n",
            "example_labels = y_test[example_idx]\n",
            "\n",
            "# Encode to latent space\n",
            "z_points = vae.encoder.predict(example_images)\n",
            "\n",
            "# CDF transform (maps to uniform [0,1])\n",
            "p_points = norm.cdf(z_points)\n",
            "\n",
            "fig = plt.figure(figsize=(fig_width, fig_height))\n",
            "\n",
            "# Left: Original latent space (colored by digit)\n",
            "ax = fig.add_subplot(1, 2, 1)\n",
            "scatter1 = ax.scatter(z_points[:, 0], z_points[:, 1],\n",
            "                      cmap='rainbow', c=example_labels, alpha=0.5, s=2)\n",
            "plt.colorbar(scatter1, ax=ax, label='Digit')\n",
            "ax.set_title('Latent Space (z)')\n",
            "ax.set_xlabel('z[0]')\n",
            "ax.set_ylabel('z[1]')\n",
            "\n",
            "# Right: CDF-transformed space\n",
            "ax = fig.add_subplot(1, 2, 2)\n",
            "scatter2 = ax.scatter(p_points[:, 0], p_points[:, 1],\n",
            "                      cmap='rainbow', c=example_labels, alpha=0.5, s=5)\n",
            "ax.set_title('CDF Transform (uniform)')\n",
            "ax.set_xlabel('CDF(z[0])')\n",
            "ax.set_ylabel('CDF(z[1])')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    print("✅ 16. Colored Latent Space")

    # =========================================================================
    # Cell 17: Digit Manifold (Grid)
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Digit Manifold (Latent Space Grid)"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# DIGIT MANIFOLD (LATENT SPACE GRID)\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Generate a grid of images by sampling uniformly across latent space.\n",
            "# Uses PPF (percent point function / inverse CDF) to map [0.01, 0.99] to z-space.\n",
            "\n",
            "n_to_show = 5000\n",
            "grid_size = 20\n",
            "figsize = 8\n",
            "\n",
            "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
            "example_images = x_test[example_idx]\n",
            "example_labels = y_test[example_idx]\n",
            "\n",
            "# Encode to get latent distribution\n",
            "z_points = vae.encoder.predict(example_images)\n",
            "\n",
            "# Show latent scatter colored by digit\n",
            "plt.figure(figsize=(5, 5))\n",
            "plt.scatter(z_points[:, 0], z_points[:, 1],\n",
            "            cmap='rainbow', c=example_labels, alpha=0.5, s=2)\n",
            "plt.colorbar(label='Digit')\n",
            "\n",
            "# Create grid in latent space using inverse CDF (PPF)\n",
            "# Maps uniform [0.01, 0.99] to normally distributed z values\n",
            "x = norm.ppf(np.linspace(0.01, 0.99, grid_size))\n",
            "y = norm.ppf(np.linspace(0.01, 0.99, grid_size))\n",
            "xv, yv = np.meshgrid(x, y)\n",
            "z_grid = np.array(list(zip(xv.flatten(), yv.flatten())))\n",
            "\n",
            "# Decode grid to generate digit manifold\n",
            "reconst = vae.decoder.predict(z_grid)\n",
            "\n",
            "# Overlay grid points\n",
            "plt.scatter(z_grid[:, 0], z_grid[:, 1], c='black', alpha=1, s=2)\n",
            "plt.title('Latent Space with Grid Points')\n",
            "plt.show()\n",
            "\n",
            "# Display digit manifold (20x20 grid)\n",
            "fig = plt.figure(figsize=(figsize, figsize))\n",
            "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
            "\n",
            "for i in range(grid_size ** 2):\n",
            "    ax = fig.add_subplot(grid_size, grid_size, i + 1)\n",
            "    ax.axis('off')\n",
            "    ax.imshow(reconst[i, :, :, 0], cmap='Greys')\n",
            "\n",
            "plt.suptitle('Digit Manifold (20x20 Latent Grid)')\n",
            "plt.show()"
        ]
    })
    print("✅ 17. Digit Manifold (Grid)")

    # =========================================================================
    # Cell 18-19: Kernel Restart (commented out)
    # =========================================================================
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Cleanup: Restart Kernel to Release GPU Memory"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# CLEANUP: Restart kernel to fully release GPU memory\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# TensorFlow/CUDA does not release GPU memory within a running Python process.\n",
            "# Restarting the kernel is the only guaranteed way to free all GPU resources.\n",
            "# Run this cell only after all work is complete and saved.\n",
            "\n",
            "# import IPython\n",
            "# print(\"Restarting kernel to release GPU memory...\")\n",
            "# IPython.Application.instance().kernel.do_shutdown(restart=True)"
        ]
    })
    print("✅ 18-19. Kernel restart (commented out)")

    # Update notebook cells
    nb['cells'] = new_cells

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"\n✅ All 11 changes applied successfully!")
    print(f"✅ Notebook saved: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
