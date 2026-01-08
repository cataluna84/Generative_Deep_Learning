#!/usr/bin/env python3
"""
Update Wasserstein loss visualization markdown with theory and fix color reference.

This script:
1. Enhances the Wasserstein loss markdown cell with mathematical formulas
2. Updates the "orange line" reference to "blue line" per user's change

Usage:
    python scripts/update_wasserstein_markdown.py
"""

import json
import os


def update_notebook():
    """Update Wasserstein loss markdown with theory."""
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

    # New enhanced markdown content
    new_markdown_source = [
        "## Wasserstein Loss Visualization\n",
        "\n",
        "This cell visualizes the training dynamics of the WGAN by plotting the "
        "Wasserstein distance over training epochs.\n",
        "\n",
        "### Theory\n",
        "\n",
        "The Wasserstein distance (also known as **Earth Mover's Distance**) measures "
        "the minimum \"cost\" to transform one probability distribution into another. "
        "For the WGAN, this is computed as:\n",
        "\n",
        "$$W(P_r, P_g) = \\sup_{\\|f\\|_L \\leq 1} \\mathbb{E}_{x \\sim P_r}[f(x)] - "
        "\\mathbb{E}_{z \\sim P_z}[f(G(z))]$$\n",
        "\n",
        "Where:\n",
        "- $P_r$ is the real data distribution\n",
        "- $P_g$ is the generated data distribution  \n",
        "- $f$ is a K-Lipschitz function (the critic)\n",
        "- $G(z)$ is the generator output from latent $z$\n",
        "\n",
        "In practice, the losses are computed as:\n",
        "- **D Loss (Real)**: $-\\mathbb{E}_{x \\sim P_r}[D(x)]$ (critic score on real)\n",
        "- **D Loss (Fake)**: $\\mathbb{E}_{z \\sim P_z}[D(G(z))]$ (critic score on fake)\n",
        "- **G Loss**: $-\\mathbb{E}_{z \\sim P_z}[D(G(z))]$ (negative of fake score)\n",
        "\n",
        "The **Wasserstein distance** is approximated by $|G_{loss}|$.\n",
        "\n",
        "### What the Plots Show\n",
        "\n",
        "- **Black line**: Combined critic loss (D loss) - average of real and fake\n",
        "- **Green line**: Critic loss on real images (R) - how the critic scores real\n",
        "- **Red line**: Critic loss on fake images (F) - how the critic scores fake\n",
        "- **Blue line**: Generator loss (G loss) - the Wasserstein distance estimate\n",
        "\n",
        "### Why This Matters\n",
        "\n",
        "Unlike standard GANs where the loss is meaningless (always ~0.69 at equilibrium), "
        "the WGAN loss is **meaningful and correlates with sample quality**:\n",
        "\n",
        "1. **Interpretable**: The magnitude approximates the Earth Mover's Distance\n",
        "2. **Monotonic**: A steadily decreasing (more negative) G loss indicates improvement\n",
        "3. **Stable gradients**: The Wasserstein loss provides useful gradients throughout training\n",
        "4. **No mode collapse indicator**: Sudden loss changes may signal distribution collapse\n"
    ]

    # Find and update the Wasserstein loss markdown cell
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            if "Wasserstein Loss Visualization" in source and "Orange line" in source.replace("orange", "Orange"):
                cell["source"] = new_markdown_source
                print(f"✓ Updated Wasserstein loss markdown at cell {i}")
                break

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)

    print(f"✓ Saved: {notebook_path}")


if __name__ == "__main__":
    update_notebook()
