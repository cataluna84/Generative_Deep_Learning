#!/usr/bin/env python3
"""
Script to standardize 03_02_autoencoder_analysis.ipynb.

This is an analysis notebook (not training), so only partial standards apply.

Usage:
    uv run python scripts/standardize_03_02_notebook.py
"""

import json
import os


def create_standardized_notebook():
    """Create a fully standardized autoencoder analysis notebook."""
    
    cells = []
    
    # =========================================================================
    # Cell 0: Header Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Autoencoder Analysis\n",
            "\n",
            "This notebook analyzes a trained Autoencoder model, visualizing reconstructions\n",
            "and exploring the learned 2D latent space.\n",
            "\n",
            "**Standards Applied:**\n",
            "- ✅ GPU memory growth enabled\n",
            "- ✅ Clean imports section\n",
            "- ✅ Proper section headers\n",
            "- ✅ Kernel restart cell (commented out)"
        ]
    })
    
    # =========================================================================
    # Cell 1: GPU Setup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## GPU Setup"]
    })
    
    # =========================================================================
    # Cell 2: GPU Memory Growth Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GPU MEMORY GROWTH\n",
            "# Enable memory growth to prevent TensorFlow from allocating all GPU memory\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "import tensorflow as tf\n",
            "\n",
            "gpus = tf.config.list_physical_devices('GPU')\n",
            "if gpus:\n",
            "    for gpu in gpus:\n",
            "        tf.config.experimental.set_memory_growth(gpu, True)\n",
            "    print(f\"GPU(s) available: {[gpu.name for gpu in gpus]}\")\n",
            "else:\n",
            "    print(\"WARNING: No GPU detected, running on CPU\")"
        ]
    })
    
    # =========================================================================
    # Cell 3: Imports Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Imports"]
    })
    
    # =========================================================================
    # Cell 4: Imports Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Standard library imports\n",
            "import sys\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy.stats import norm\n",
            "\n",
            "# Path setup for project utilities\n",
            "sys.path.insert(0, '../..')    # For project root utils/\n",
            "sys.path.insert(0, '..')       # For v1/src modules\n",
            "\n",
            "# Project utilities\n",
            "from src.models.AE import Autoencoder\n",
            "from src.utils.loaders import load_mnist, load_model"
        ]
    })
    
    # =========================================================================
    # Cell 5: Configuration Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Configuration"]
    })
    
    # =========================================================================
    # Cell 6: Configuration Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run params - must match 03_01_autoencoder_train.ipynb\n",
            "SECTION = 'vae'\n",
            "RUN_ID = '0001'\n",
            "DATA_NAME = 'digits'\n",
            "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'\n",
            "\n",
            "print(f\"Loading model from: {RUN_FOLDER}\")"
        ]
    })
    
    # =========================================================================
    # Cell 7: Load Data Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Data"]
    })
    
    # =========================================================================
    # Cell 8: Load Data Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load MNIST dataset\n",
            "(x_train, y_train), (x_test, y_test) = load_mnist()\n",
            "\n",
            "print(f\"Training samples: {x_train.shape[0]}\")\n",
            "print(f\"Test samples: {x_test.shape[0]}\")"
        ]
    })
    
    # =========================================================================
    # Cell 9: Load Model Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Model"]
    })
    
    # =========================================================================
    # Cell 10: Load Model Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load trained autoencoder\n",
            "AE = load_model(Autoencoder, RUN_FOLDER)\n",
            "print(\"Model loaded successfully!\")"
        ]
    })
    
    # =========================================================================
    # Cell 11: Reconstruction Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Reconstructing Images\n",
            "\n",
            "Visualize original images and their autoencoder reconstructions."
        ]
    })
    
    # =========================================================================
    # Cell 12: Reconstruction Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select random test samples\n",
            "n_to_show = 10\n",
            "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
            "example_images = x_test[example_idx]\n",
            "\n",
            "# Encode and decode\n",
            "z_points = AE.encoder.predict(example_images)\n",
            "reconst_images = AE.decoder.predict(z_points)\n",
            "\n",
            "# Visualize\n",
            "fig = plt.figure(figsize=(15, 3))\n",
            "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
            "\n",
            "for i in range(n_to_show):\n",
            "    # Original\n",
            "    img = example_images[i].squeeze()\n",
            "    ax = fig.add_subplot(2, n_to_show, i + 1)\n",
            "    ax.axis('off')\n",
            "    ax.text(0.5, -0.35, str(np.round(z_points[i], 1)),\n",
            "            fontsize=10, ha='center', transform=ax.transAxes)\n",
            "    ax.imshow(img, cmap='gray_r')\n",
            "\n",
            "for i in range(n_to_show):\n",
            "    # Reconstruction\n",
            "    img = reconst_images[i].squeeze()\n",
            "    ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)\n",
            "    ax.axis('off')\n",
            "    ax.imshow(img, cmap='gray_r')\n",
            "\n",
            "plt.suptitle('Original (top) vs Reconstruction (bottom)', y=1.02)\n",
            "plt.show()"
        ]
    })
    
    # =========================================================================
    # Cell 13: Latent Space Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Exploring the Latent Space\n",
            "\n",
            "Visualize the 2D latent space with digit class coloring."
        ]
    })
    
    # =========================================================================
    # Cell 14: Latent Space Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Encode all training data\n",
            "z_train = AE.encoder.predict(x_train)\n",
            "\n",
            "# Plot latent space colored by digit class\n",
            "fig = plt.figure(figsize=(8, 8))\n",
            "scatter = plt.scatter(z_train[:, 0], z_train[:, 1],\n",
            "                       c=y_train, cmap='tab10', alpha=0.5, s=2)\n",
            "plt.colorbar(scatter, label='Digit Class')\n",
            "plt.xlabel('z[0]')\n",
            "plt.ylabel('z[1]')\n",
            "plt.title('Latent Space Visualization')\n",
            "plt.show()"
        ]
    })
    
    # =========================================================================
    # Cell 15: Latent Grid Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Latent Space Grid\n",
            "\n",
            "Generate a grid of images by sampling uniformly across the latent space."
        ]
    })
    
    # =========================================================================
    # Cell 16: Latent Grid Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Grid settings\n",
            "n = 20  # Number of samples per dimension\n",
            "figure_size = 8\n",
            "\n",
            "# Get latent space bounds\n",
            "z_min = z_train.min(axis=0)\n",
            "z_max = z_train.max(axis=0)\n",
            "\n",
            "# Create grid\n",
            "z0_vals = np.linspace(z_min[0], z_max[0], n)\n",
            "z1_vals = np.linspace(z_min[1], z_max[1], n)[::-1]  # Flip for image layout\n",
            "z_grid = np.array([[z0, z1] for z1 in z1_vals for z0 in z0_vals])\n",
            "\n",
            "# Decode grid\n",
            "reconst = AE.decoder.predict(z_grid)\n",
            "\n",
            "# Create mosaic image\n",
            "image_size = 28\n",
            "mosaic = np.zeros((n * image_size, n * image_size))\n",
            "\n",
            "for i, img in enumerate(reconst):\n",
            "    row = i // n\n",
            "    col = i % n\n",
            "    mosaic[row * image_size:(row + 1) * image_size,\n",
            "           col * image_size:(col + 1) * image_size] = img.squeeze()\n",
            "\n",
            "# Display\n",
            "plt.figure(figsize=(figure_size, figure_size))\n",
            "plt.imshow(mosaic, cmap='gray_r')\n",
            "plt.axis('off')\n",
            "plt.title('Latent Space Grid')\n",
            "plt.show()"
        ]
    })
    
    # =========================================================================
    # Cell 17: Cleanup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Cleanup"]
    })
    
    # =========================================================================
    # Cell 18: Kernel Restart Code (commented out)
    # =========================================================================
    cells.append({
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
            "#\n",
            "# NOTE: Only run this cell after all work is complete and saved.\n",
            "#       The kernel restart will clear all variables and outputs.\n",
            "\n",
            "# import IPython\n",
            "# print(\"Restarting kernel to release GPU memory...\")\n",
            "# IPython.Application.instance().kernel.do_shutdown(restart=True)"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Main function to run the standardization."""
    notebook_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'v1', 'notebooks',
        '03_02_autoencoder_analysis.ipynb'
    )
    notebook_path = os.path.abspath(notebook_path)
    
    print(f"Standardizing notebook: {notebook_path}")
    
    # Create standardized notebook
    notebook = create_standardized_notebook()
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n{'='*60}")
    print("STANDARDIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cells: {len(notebook['cells'])}")
    print("\nStandards applied:")
    print("  ✅ GPU memory growth cell (first code cell)")
    print("  ✅ Clean imports section")
    print("  ✅ Proper section headers (## Title Case)")
    print("  ✅ Configuration section")
    print("  ✅ Kernel restart cell (commented out)")
    print("\nNOTE: This is an ANALYSIS notebook - training standards don't apply:")
    print("  ❌ Dynamic batch/epoch scaling (N/A)")
    print("  ❌ LRFinder (N/A)")
    print("  ❌ Callback stack (N/A)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
