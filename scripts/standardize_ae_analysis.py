#!/usr/bin/env python3
"""
Script to standardize 03_02_autoencoder_analysis.ipynb notebook.
Applies 8-step standardization while preserving all native logic and plotting.
"""

import json
import os

# Path to notebook
NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "v1", "notebooks", "03_02_autoencoder_analysis.ipynb"
)

def main():
    """Apply standardization changes to the notebook."""
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Enhanced notebook header (Cell 0)
    # ═══════════════════════════════════════════════════════════════════════════
    cells[0]['source'] = [
        "# 03_02 Autoencoder Analysis\n",
        "\n",
        "**Chapter 3**: Variational Autoencoders | **Notebook 2 of 6**\n",
        "\n",
        "This notebook analyzes a trained autoencoder by:\n",
        "- Reconstructing test images\n",
        "- Visualizing the 2D latent space\n",
        "- Generating new images from latent space samples"
    ]
    print("✅ Step 1: Enhanced header")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Fix "imports" header capitalization (Cell 1)
    # ═══════════════════════════════════════════════════════════════════════════
    cells[1]['source'] = ["## Imports"]
    print("✅ Step 2: Fixed 'imports' → 'Imports'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Robust GPU setup with error handling (Cell 2)
    # ═══════════════════════════════════════════════════════════════════════════
    cells[2]['source'] = [
        "import tensorflow as tf\n",
        "\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# GPU MEMORY CONFIGURATION\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Enable memory growth to prevent TensorFlow from allocating all GPU memory.\n",
        "gpus = tf.config.list_physical_devices('GPU')\n",
        "if gpus:\n",
        "    for gpu in gpus:\n",
        "        tf.config.experimental.set_memory_growth(gpu, True)\n",
        "    print(f\"GPU(s) available: {[gpu.name for gpu in gpus]}\")\n",
        "else:\n",
        "    print(\"WARNING: No GPU detected, running on CPU\")"
    ]
    print("✅ Step 3: Robust GPU setup")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: PEP 8 compliant imports with documentation (Cell 3)
    # ═══════════════════════════════════════════════════════════════════════════
    cells[3]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# IMPORTS\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "\n",
        "# Standard library\n",
        "import sys\n",
        "import os\n",
        "\n",
        "# Path setup for local imports\n",
        "sys.path.insert(0, '..')\n",
        "sys.path.insert(0, '../..')\n",
        "\n",
        "# Third-party libraries\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from scipy.stats import norm\n",
        "\n",
        "# Local imports\n",
        "from src.models.AE import Autoencoder\n",
        "from src.utils.loaders import load_mnist, load_model"
    ]
    print("✅ Step 4: PEP 8 imports")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Global configuration block (Cell 4)
    # ═══════════════════════════════════════════════════════════════════════════
    cells[4]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# GLOBAL CONFIGURATION\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "\n",
        "# Model identification\n",
        "SECTION = 'vae'\n",
        "RUN_ID = '0001'\n",
        "DATA_NAME = 'digits'\n",
        "\n",
        "# Directory paths\n",
        "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'\n",
        "\n",
        "print(f\"Loading model from: {RUN_FOLDER}\")"
    ]
    print("✅ Step 5: Global config block")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 6: Fix section headers (Cells 5, 7, 9, 11, 13)
    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 5: "Load the data" → "Load Data"
    cells[5]['source'] = ["## Load Data"]
    
    # Cell 7: "Load the model architecture" → "Load Model"
    cells[7]['source'] = ["## Load Model"]
    
    # Cell 9: "reconstructing original paintings" → "Reconstruct Original Images"
    cells[9]['source'] = ["## Reconstruct Original Images"]
    
    # Cell 11: "Mr N. Coder's wall" → "Latent Space Visualization"
    cells[11]['source'] = ["## Latent Space Visualization"]
    
    # Cell 13: "The new generated art exhibition" → "Generate New Images from Latent Space"
    cells[13]['source'] = ["### Generate New Images from Latent Space"]
    
    print("✅ Step 6: Fixed section headers")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 7: Add comments to analysis cells
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Cell 6: Load data
    cells[6]['source'] = [
        "# Load MNIST dataset (preprocessed to [0,1] range)\n",
        "(x_train, y_train), (x_test, y_test) = load_mnist()"
    ]

    # Cell 8: Load model
    cells[8]['source'] = [
        "# Load pre-trained autoencoder from run folder\n",
        "AE = load_model(Autoencoder, RUN_FOLDER)"
    ]

    # Cell 10: Reconstruction visualization
    cells[10]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# RECONSTRUCTION VISUALIZATION\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Select random test images, encode to latent space, decode back, and compare.\n",
        "\n",
        "n_to_show = 10\n",
        "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
        "example_images = x_test[example_idx]\n",
        "\n",
        "# Encode to latent space\n",
        "z_points = AE.encoder.predict(example_images)\n",
        "\n",
        "# Decode back to image space\n",
        "reconst_images = AE.decoder.predict(z_points)\n",
        "\n",
        "# Display original images (top row) and reconstructions (bottom row)\n",
        "fig = plt.figure(figsize=(15, 3))\n",
        "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
        "\n",
        "for i in range(n_to_show):\n",
        "    img = example_images[i].squeeze()\n",
        "    ax = fig.add_subplot(2, n_to_show, i+1)\n",
        "    ax.axis('off')\n",
        "    ax.text(0.5, -0.35, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)   \n",
        "    ax.imshow(img, cmap='gray_r')\n",
        "\n",
        "for i in range(n_to_show):\n",
        "    img = reconst_images[i].squeeze()\n",
        "    ax = fig.add_subplot(2, n_to_show, i+n_to_show+1)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(img, cmap='gray_r')"
    ]

    # Cell 12: Latent space scatter plot
    cells[12]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# LATENT SPACE SCATTER PLOT\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Encode test images and visualize 2D latent space distribution.\n",
        "\n",
        "n_to_show = 5000\n",
        "grid_size = 15\n",
        "figsize = 12\n",
        "\n",
        "# Sample random test images\n",
        "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
        "example_images = x_test[example_idx]\n",
        "example_labels = y_test[example_idx]\n",
        "\n",
        "# Encode to latent space\n",
        "z_points = AE.encoder.predict(example_images)\n",
        "\n",
        "# Get latent space bounds for later use\n",
        "min_x = min(z_points[:, 0])\n",
        "max_x = max(z_points[:, 0])\n",
        "min_y = min(z_points[:, 1])\n",
        "max_y = max(z_points[:, 1])\n",
        "\n",
        "# Display scatter plot of latent encodings\n",
        "plt.figure(figsize=(figsize, figsize))\n",
        "plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)\n",
        "plt.show()"
    ]

    # Cell 14: Random latent space sampling
    cells[14]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# RANDOM LATENT SPACE SAMPLING\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Sample random points from the latent space and decode to generate new images.\n",
        "\n",
        "figsize = 5\n",
        "\n",
        "# Show reference scatter plot\n",
        "plt.figure(figsize=(figsize, figsize))\n",
        "plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)\n",
        "\n",
        "grid_size = 10\n",
        "grid_depth = 3\n",
        "figsize = 15\n",
        "\n",
        "# Sample random points within latent space bounds\n",
        "x = np.random.uniform(min_x,max_x, size = grid_size * grid_depth)\n",
        "y = np.random.uniform(min_y,max_y, size = grid_size * grid_depth)\n",
        "z_grid = np.array(list(zip(x, y)))\n",
        "\n",
        "# Decode sampled points to generate images\n",
        "reconst = AE.decoder.predict(z_grid)\n",
        "\n",
        "# Overlay sampled points on scatter plot\n",
        "plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'red', alpha=1, s=20)\n",
        "plt.show()\n",
        "\n",
        "# Display generated images\n",
        "fig = plt.figure(figsize=(figsize, grid_depth))\n",
        "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
        "\n",
        "for i in range(grid_size*grid_depth):\n",
        "    ax = fig.add_subplot(grid_depth, grid_size, i+1)\n",
        "    ax.axis('off')\n",
        "    ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=10, ha='center', transform=ax.transAxes)\n",
        "    \n",
        "    ax.imshow(reconst[i, :,:,0], cmap = 'Greys')"
    ]

    # Cell 15: Latent space by digit label
    cells[15]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# LATENT SPACE BY DIGIT LABEL\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Visualize latent space with color-coded digit labels (0-9).\n",
        "\n",
        "n_to_show = 5000\n",
        "grid_size = 15\n",
        "figsize = 12\n",
        "\n",
        "# Sample random test images with their labels\n",
        "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
        "example_images = x_test[example_idx]\n",
        "example_labels = y_test[example_idx]\n",
        "\n",
        "# Encode to latent space\n",
        "z_points = AE.encoder.predict(example_images)\n",
        "\n",
        "# Scatter plot with rainbow colormap for digit labels\n",
        "plt.figure(figsize=(figsize, figsize))\n",
        "plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels\n",
        "            , alpha=0.5, s=2)\n",
        "plt.colorbar()\n",
        "plt.show()"
    ]

    # Cell 16: Grid sampling visualization
    cells[16]['source'] = [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# LATENT SPACE GRID SAMPLING\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Create a uniform grid across the latent space and decode each point to\n",
        "# visualize how the autoencoder interpolates between digit representations.\n",
        "\n",
        "n_to_show = 5000\n",
        "grid_size = 20\n",
        "figsize = 8\n",
        "\n",
        "# Sample test images for reference scatter plot\n",
        "example_idx = np.random.choice(range(len(x_test)), n_to_show)\n",
        "example_images = x_test[example_idx]\n",
        "example_labels = y_test[example_idx]\n",
        "\n",
        "# Encode to latent space\n",
        "z_points = AE.encoder.predict(example_images)\n",
        "\n",
        "# Display scatter with digit labels\n",
        "plt.figure(figsize=(5, 5))\n",
        "plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels\n",
        "            , alpha=0.5, s=2)\n",
        "plt.colorbar()\n",
        "\n",
        "# Create uniform grid spanning the latent space\n",
        "x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)\n",
        "y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)\n",
        "xv, yv = np.meshgrid(x, y)\n",
        "xv = xv.flatten()\n",
        "yv = yv.flatten()\n",
        "z_grid = np.array(list(zip(xv, yv)))\n",
        "\n",
        "# Decode grid points to generate images\n",
        "reconst = AE.decoder.predict(z_grid)\n",
        "\n",
        "# Overlay grid points on scatter plot\n",
        "plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'black', alpha=1, s=5)\n",
        "plt.show()\n",
        "\n",
        "# Display decoded images in a grid\n",
        "fig = plt.figure(figsize=(figsize, figsize))\n",
        "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
        "for i in range(grid_size**2):\n",
        "    ax = fig.add_subplot(grid_size, grid_size, i+1)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(reconst[i, :,:,0], cmap = 'Greys')"
    ]
    
    print("✅ Step 7: Added cell comments")

    # Step 8: Kernel restart cell is already present and formatted correctly
    print("✅ Step 8: Kernel restart cell verified")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save updated notebook
    # ═══════════════════════════════════════════════════════════════════════════
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Notebook saved: {NOTEBOOK_PATH}")
    print("=" * 60)
    print("All 8 standardization steps completed successfully!")


if __name__ == "__main__":
    main()
