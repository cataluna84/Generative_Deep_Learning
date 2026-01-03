#!/usr/bin/env python3
"""
Script to standardize 02_02_deep_learning_convolutions.ipynb.

This notebook demonstrates manual convolution operations - not a training notebook.
Applicable standards: 1 (not needed - no TF), 2 (documentation), 4 (W&B), 7 (kernel restart).
Not applicable: 3 (batch size), 5 (LR scheduler), 6 (visualizations - already has them).

Usage:
    uv run python scripts/standardize_02_02_notebook.py
"""

import json
import os


def create_standardized_notebook():
    """Create a fully standardized convolution demonstration notebook."""
    
    cells = []
    
    # =========================================================================
    # Cell 0: Header Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Convolution Operations\n",
            "\n",
            "This notebook demonstrates manual convolution operations using NumPy.\n",
            "It visualizes how edge detection filters work with different strides.\n",
            "\n",
            "**Topics Covered:**\n",
            "- Horizontal edge detection filter\n",
            "- Vertical edge detection filter\n",
            "- Stride effects on output dimensions\n",
            "\n",
            "**Standards Applied:**\n",
            "- ✅ W&B integration for image tracking\n",
            "- ✅ Proper documentation and section headers\n",
            "- ✅ Kernel restart cell (commented out)"
        ]
    })
    
    # =========================================================================
    # Cell 1: Imports Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Imports"]
    })
    
    # =========================================================================
    # Cell 2: Imports Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Standard library imports\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Image processing imports\n",
            "from scipy.ndimage import correlate\n",
            "from skimage import data\n",
            "from skimage.color import rgb2gray\n",
            "from skimage.transform import rescale, resize\n",
            "\n",
            "# Path setup for project utilities\n",
            "import sys\n",
            "sys.path.insert(0, '../..')    # For project root utils/\n",
            "\n",
            "# W&B integration\n",
            "from utils.wandb_utils import init_wandb\n",
            "import wandb"
        ]
    })
    
    # =========================================================================
    # Cell 3: W&B Init Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## W&B Initialization"]
    })
    
    # =========================================================================
    # Cell 4: W&B Init Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize W&B for experiment tracking\n",
            "run = init_wandb(\n",
            "    name=\"02_02_convolutions\",\n",
            "    config={\n",
            "        \"notebook\": \"02_02_deep_learning_convolutions\",\n",
            "        \"description\": \"Manual convolution operations with NumPy\",\n",
            "        \"image_size\": 64,\n",
            "        \"filters\": [\"horizontal_edge\", \"vertical_edge\"],\n",
            "    }\n",
            ")"
        ]
    })
    
    # =========================================================================
    # Cell 5: Load Image Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load and Display Original Image"]
    })
    
    # =========================================================================
    # Cell 6: Load Image Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load sample image and convert to grayscale\n",
            "im = rgb2gray(data.coffee())\n",
            "im = resize(im, (64, 64))\n",
            "\n",
            "print(f\"Image shape: {im.shape}\")\n",
            "print(f\"Pixel value range: [{im.min():.3f}, {im.max():.3f}]\")\n",
            "\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.axis('off')\n",
            "plt.title('Original Grayscale Image')\n",
            "plt.imshow(im, cmap='gray')\n",
            "plt.show()\n",
            "\n",
            "# Log to W&B\n",
            "wandb.log({\"original_image\": wandb.Image(im, caption=\"Original Grayscale\")})"
        ]
    })
    
    # =========================================================================
    # Cell 7: Horizontal Edge Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Horizontal Edge Detection Filter\n",
            "\n",
            "This filter detects horizontal edges by computing the difference between\n",
            "pixels above and below each position."
        ]
    })
    
    # =========================================================================
    # Cell 8: Horizontal Edge Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define horizontal edge detection filter\n",
            "filter_horizontal = np.array([\n",
            "    [ 1,  1,  1],\n",
            "    [ 0,  0,  0],\n",
            "    [-1, -1, -1]\n",
            "])\n",
            "\n",
            "print(\"Horizontal Edge Filter:\")\n",
            "print(filter_horizontal)\n",
            "\n",
            "# Apply convolution manually\n",
            "new_image = np.zeros(im.shape)\n",
            "im_pad = np.pad(im, 1, 'constant')\n",
            "\n",
            "for i in range(im.shape[0]):\n",
            "    for j in range(im.shape[1]):\n",
            "        new_image[i, j] = np.sum(im_pad[i:i+3, j:j+3] * filter_horizontal)\n",
            "\n",
            "# Display result\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.axis('off')\n",
            "plt.title('Horizontal Edge Detection')\n",
            "plt.imshow(new_image, cmap='Greys')\n",
            "plt.show()\n",
            "\n",
            "# Log to W&B\n",
            "wandb.log({\"horizontal_edge\": wandb.Image(new_image, caption=\"Horizontal Edge Filter\")})"
        ]
    })
    
    # =========================================================================
    # Cell 9: Vertical Edge Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Vertical Edge Detection Filter\n",
            "\n",
            "This filter detects vertical edges by computing the difference between\n",
            "pixels to the left and right of each position."
        ]
    })
    
    # =========================================================================
    # Cell 10: Vertical Edge Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define vertical edge detection filter\n",
            "filter_vertical = np.array([\n",
            "    [-1,  0,  1],\n",
            "    [-1,  0,  1],\n",
            "    [-1,  0,  1]\n",
            "])\n",
            "\n",
            "print(\"Vertical Edge Filter:\")\n",
            "print(filter_vertical)\n",
            "\n",
            "# Apply convolution manually\n",
            "new_image = np.zeros(im.shape)\n",
            "im_pad = np.pad(im, 1, 'constant')\n",
            "\n",
            "for i in range(im.shape[0]):\n",
            "    for j in range(im.shape[1]):\n",
            "        new_image[i, j] = np.sum(im_pad[i:i+3, j:j+3] * filter_vertical)\n",
            "\n",
            "# Display result\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.axis('off')\n",
            "plt.title('Vertical Edge Detection')\n",
            "plt.imshow(new_image, cmap='Greys')\n",
            "plt.show()\n",
            "\n",
            "# Log to W&B\n",
            "wandb.log({\"vertical_edge\": wandb.Image(new_image, caption=\"Vertical Edge Filter\")})"
        ]
    })
    
    # =========================================================================
    # Cell 11: Stride Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Stride Effects\n",
            "\n",
            "Using a stride > 1 reduces the output dimensions. With stride=2, the output\n",
            "is half the size in each dimension."
        ]
    })
    
    # =========================================================================
    # Cell 12: Horizontal Stride Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Horizontal edge with stride 2\n",
            "stride = 2\n",
            "output_shape = (im.shape[0] // stride, im.shape[1] // stride)\n",
            "new_image = np.zeros(output_shape)\n",
            "\n",
            "im_pad = np.pad(im, 1, 'constant')\n",
            "\n",
            "for i in range(0, im.shape[0], stride):\n",
            "    for j in range(0, im.shape[1], stride):\n",
            "        new_image[i // stride, j // stride] = np.sum(\n",
            "            im_pad[i:i+3, j:j+3] * filter_horizontal\n",
            "        )\n",
            "\n",
            "print(f\"Output shape with stride {stride}: {new_image.shape}\")\n",
            "\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.axis('off')\n",
            "plt.title(f'Horizontal Edge (Stride {stride})')\n",
            "plt.imshow(new_image, cmap='Greys')\n",
            "plt.show()\n",
            "\n",
            "# Log to W&B\n",
            "wandb.log({\"horizontal_stride_2\": wandb.Image(new_image, caption=\"Horizontal Edge (Stride 2)\")})"
        ]
    })
    
    # =========================================================================
    # Cell 13: Vertical Stride Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Vertical edge with stride 2\n",
            "stride = 2\n",
            "output_shape = (im.shape[0] // stride, im.shape[1] // stride)\n",
            "new_image = np.zeros(output_shape)\n",
            "\n",
            "im_pad = np.pad(im, 1, 'constant')\n",
            "\n",
            "for i in range(0, im.shape[0], stride):\n",
            "    for j in range(0, im.shape[1], stride):\n",
            "        new_image[i // stride, j // stride] = np.sum(\n",
            "            im_pad[i:i+3, j:j+3] * filter_vertical\n",
            "        )\n",
            "\n",
            "print(f\"Output shape with stride {stride}: {new_image.shape}\")\n",
            "\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.axis('off')\n",
            "plt.title(f'Vertical Edge (Stride {stride})')\n",
            "plt.imshow(new_image, cmap='Greys')\n",
            "plt.show()\n",
            "\n",
            "# Log to W&B\n",
            "wandb.log({\"vertical_stride_2\": wandb.Image(new_image, caption=\"Vertical Edge (Stride 2)\")})"
        ]
    })
    
    # =========================================================================
    # Cell 14: Summary Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            "| Filter | Description | Output Size (stride=1) | Output Size (stride=2) |\n",
            "|--------|-------------|------------------------|------------------------|\n",
            "| Horizontal | Detects horizontal edges | 64×64 | 32×32 |\n",
            "| Vertical | Detects vertical edges | 64×64 | 32×32 |"
        ]
    })
    
    # =========================================================================
    # Cell 15: Cleanup Markdown
    # =========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Cleanup"]
    })
    
    # =========================================================================
    # Cell 16: W&B Finish Code
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Finish W&B run\n",
            "wandb.finish()\n",
            "print(\"W&B run finished successfully.\")"
        ]
    })
    
    # =========================================================================
    # Cell 17: Kernel Restart Code (commented out)
    # =========================================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# CLEANUP: Restart kernel to fully release memory\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# NOTE: This notebook doesn't use GPU, but kernel restart is still useful\n",
            "#       to free up memory after processing images.\n",
            "#\n",
            "# Only run this cell after all work is complete and saved.\n",
            "\n",
            "# import IPython\n",
            "# print(\"Restarting kernel to release memory...\")\n",
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
        '02_02_deep_learning_convolutions.ipynb'
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
    print("  ✅ Header with notebook description")
    print("  ✅ Proper section headers (## Title Case)")
    print("  ✅ Clean, commented imports")
    print("  ✅ W&B integration for image logging")
    print("  ✅ Improved convolution code (vectorized inner loop)")
    print("  ✅ Summary table")
    print("  ✅ Kernel restart cell (commented out)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
