#!/usr/bin/env python3
"""
Standardize the VAE Faces Analysis notebook (03_06_vae_faces_analysis.ipynb).

Changes:
1. Fix GPU setup with error handling
2. Fix imports (duplicate numpy, path setup)
3. Standardize section headers
4. Use ALL_CAPS for config
5. Add PEP-8 compliance and documentation to all cells
6. Comment out kernel restart
7. Delete empty cells
"""

import json

NOTEBOOK_PATH = "v1/notebooks/03_06_vae_faces_analysis.ipynb"

with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

print(f"Original cells: {len(nb['cells'])}")

# Build new cells list
new_cells = []

# =============================================================================
# Cell 0: Title
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# VAE Analysis - Faces Dataset (CelebA)\n",
        "\n",
        "Analysis of a trained Variational Autoencoder on the CelebA face dataset.\n",
        "\n",
        "**Contents:**\n",
        "- Face reconstruction (original vs reconstructed)\n",
        "- Latent space distribution analysis\n",
        "- Attribute vector arithmetic (adding/removing features)\n",
        "- Face morphing between images"
    ]
})

# =============================================================================
# Cell 1-2: GPU Setup
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## GPU Setup"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import tensorflow as tf\n",
        "\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# GPU MEMORY GROWTH SETUP\n",
        "# Prevents TensorFlow from allocating all GPU memory at once\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "gpus = tf.config.list_physical_devices('GPU')\n",
        "if gpus:\n",
        "    for gpu in gpus:\n",
        "        tf.config.experimental.set_memory_growth(gpu, True)\n",
        "    print(f\"GPU enabled: {gpus[0].name}\")\n",
        "else:\n",
        "    print(\"No GPU found, using CPU\")"
    ]
})

# =============================================================================
# Cell 3-4: Imports
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Imports"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Standard library imports\n",
        "import os\n",
        "import sys\n",
        "\n",
        "# Path setup for project utilities\n",
        "sys.path.insert(0, '../..')  # Project root for utils/\n",
        "sys.path.insert(0, '..')     # v1/ for src/\n",
        "\n",
        "# Scientific computing\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from scipy.stats import norm\n",
        "\n",
        "# Visualization\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Local imports\n",
        "from src.models.VAE import VariationalAutoencoder\n",
        "from src.utils.loaders import load_model, ImageLabelLoader"
    ]
})

# =============================================================================
# Cell 5-6: Global Configuration
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Global Configuration"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# GLOBAL CONFIGURATION\n",
        "# All paths and settings are defined here for easy modification\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "\n",
        "# Run identification\n",
        "SECTION = 'vae'\n",
        "RUN_ID = '0001'\n",
        "DATA_NAME = 'faces'\n",
        "\n",
        "# Paths\n",
        "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATA_NAME}'\n",
        "DATA_FOLDER = '../data/img_align_celeba/'\n",
        "IMAGE_FOLDER = '../data/img_align_celeba/images/'\n",
        "\n",
        "# Image dimensions (must match training)\n",
        "INPUT_DIM = (128, 128, 3)\n",
        "\n",
        "print(f\"Run folder: {RUN_FOLDER}\")\n",
        "print(f\"Data folder: {DATA_FOLDER}\")"
    ]
})

# =============================================================================
# Cell 7-9: Data Loading
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Data Loading"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Load CelebA attributes CSV\n",
        "# Contains 40 binary attributes (e.g., Smiling, Eyeglasses, Male) for each image\n",
        "att = pd.read_csv(os.path.join(DATA_FOLDER, 'list_attr_celeba.csv'))\n",
        "\n",
        "# Initialize image loader for loading and resizing images\n",
        "imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])\n",
        "\n",
        "print(f\"Loaded {len(att)} attribute records\")\n",
        "print(f\"Number of attributes: {len(att.columns) - 1}\")  # -1 for image_id column"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Preview attribute data\n",
        "att.head()"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Dataset size\n",
        "print(f\"Dataset shape: {att.shape}\")"
    ]
})

# =============================================================================
# Cell 10-12: Model Architecture
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Model Architecture"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Load trained VAE model from saved weights\n",
        "vae = load_model(VariationalAutoencoder, RUN_FOLDER)\n",
        "\n",
        "print(f\"Loaded VAE with z_dim={vae.z_dim}\")\n",
        "print(f\"Encoder input shape: {vae.encoder.input_shape}\")\n",
        "print(f\"Decoder output shape: {vae.decoder.output_shape}\")"
    ]
})

# =============================================================================
# Cell 13-16: Face Reconstruction
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Face Reconstruction\n",
        "\n",
        "Test the VAE's ability to encode and reconstruct faces.\n",
        "The top row shows original images, the bottom row shows reconstructions."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Number of faces to reconstruct\n",
        "N_TO_SHOW = 10\n",
        "\n",
        "# Load sample images\n",
        "data_flow_generic = imageLoader.build(att, N_TO_SHOW)\n",
        "example_batch = next(data_flow_generic)\n",
        "example_images = example_batch[0]\n",
        "\n",
        "print(f\"Loaded {len(example_images)} sample images\")"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Encode images to latent space, then decode back\n",
        "z_points = vae.encoder.predict(example_images)\n",
        "reconst_images = vae.decoder.predict(z_points)\n",
        "\n",
        "# Plot original vs reconstructed\n",
        "fig = plt.figure(figsize=(15, 3))\n",
        "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
        "\n",
        "# Top row: Original images\n",
        "for i in range(N_TO_SHOW):\n",
        "    ax = fig.add_subplot(2, N_TO_SHOW, i + 1)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(example_images[i].squeeze())\n",
        "    if i == 0:\n",
        "        ax.set_title('Original', fontsize=10)\n",
        "\n",
        "# Bottom row: Reconstructed images\n",
        "for i in range(N_TO_SHOW):\n",
        "    ax = fig.add_subplot(2, N_TO_SHOW, i + N_TO_SHOW + 1)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(reconst_images[i].squeeze())\n",
        "    if i == 0:\n",
        "        ax.set_title('Reconstructed', fontsize=10)\n",
        "\n",
        "plt.suptitle('Face Reconstruction: Original (top) vs VAE Output (bottom)', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# =============================================================================
# Cell 17-18: Latent Space Distribution
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Latent Space Distribution\n",
        "\n",
        "Visualize the distribution of latent dimensions.\n",
        "A well-trained VAE should have latent dimensions that approximate a standard normal distribution N(0,1)."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Encode many images to analyze latent distribution\n",
        "z_test = vae.encoder.predict(data_flow_generic, steps=20, verbose=1)\n",
        "\n",
        "# Reference: standard normal distribution\n",
        "x = np.linspace(-3, 3, 100)\n",
        "\n",
        "# Plot histograms for first 50 latent dimensions\n",
        "fig = plt.figure(figsize=(20, 20))\n",
        "fig.subplots_adjust(hspace=0.6, wspace=0.4)\n",
        "\n",
        "for i in range(50):\n",
        "    ax = fig.add_subplot(5, 10, i + 1)\n",
        "    ax.hist(z_test[:, i], density=True, bins=20, alpha=0.7)\n",
        "    ax.plot(x, norm.pdf(x), 'r-', linewidth=2)  # Standard normal reference\n",
        "    ax.axis('off')\n",
        "    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)\n",
        "\n",
        "plt.suptitle('Latent Dimension Distributions (red = N(0,1) reference)', fontsize=14)\n",
        "plt.show()"
    ]
})

# =============================================================================
# Cell 19-20: Generated Faces
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Newly Generated Faces\n",
        "\n",
        "Generate new faces by sampling from the latent space.\n",
        "Since the latent space is trained to be approximately N(0,1), we sample random vectors."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generate new faces from random latent vectors\n",
        "N_GENERATED = 30\n",
        "\n",
        "# Sample from standard normal distribution\n",
        "z_new = np.random.normal(size=(N_GENERATED, vae.z_dim))\n",
        "\n",
        "# Decode to image space\n",
        "generated_faces = vae.decoder.predict(z_new)\n",
        "\n",
        "# Display generated faces\n",
        "fig = plt.figure(figsize=(18, 5))\n",
        "fig.subplots_adjust(hspace=0.4, wspace=0.4)\n",
        "\n",
        "for i in range(N_GENERATED):\n",
        "    ax = fig.add_subplot(3, 10, i + 1)\n",
        "    ax.imshow(generated_faces[i])\n",
        "    ax.axis('off')\n",
        "\n",
        "plt.suptitle('Newly Generated Faces (sampled from latent space)', fontsize=14)\n",
        "plt.show()"
    ]
})

# =============================================================================
# Cell 21-22: Attribute Vector Functions
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Attribute Vector Arithmetic\n",
        "\n",
        "Find directions in latent space that correspond to facial attributes.\n",
        "By computing the mean latent vector for images with/without an attribute,\n",
        "we can find a \"feature vector\" that represents that attribute."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def get_vector_from_label(label, batch_size):\n",
        "    \"\"\"\n",
        "    Compute the attribute vector for a given facial attribute.\n",
        "    \n",
        "    The attribute vector is the difference between the mean latent vector\n",
        "    of images WITH the attribute and images WITHOUT the attribute.\n",
        "    \n",
        "    Args:\n",
        "        label (str): Attribute name from CelebA (e.g., 'Smiling', 'Eyeglasses')\n",
        "        batch_size (int): Batch size for processing images\n",
        "    \n",
        "    Returns:\n",
        "        np.ndarray: Normalized attribute vector in latent space\n",
        "    \"\"\"\n",
        "    # Load images with attribute labels\n",
        "    data_flow_label = imageLoader.build(att, batch_size, label=label)\n",
        "    \n",
        "    # Initialize accumulators for positive and negative samples\n",
        "    current_sum_POS = np.zeros(shape=vae.z_dim, dtype='float32')\n",
        "    current_sum_NEG = np.zeros(shape=vae.z_dim, dtype='float32')\n",
        "    current_n_POS = 0\n",
        "    current_n_NEG = 0\n",
        "    current_mean_POS = np.zeros(shape=vae.z_dim, dtype='float32')\n",
        "    current_mean_NEG = np.zeros(shape=vae.z_dim, dtype='float32')\n",
        "    current_dist = 0\n",
        "    \n",
        "    print(f'Finding vector for: {label}')\n",
        "    print('Images : POS move : NEG move : Distance : Δ Distance')\n",
        "    \n",
        "    # Iterate until convergence or max samples\n",
        "    while current_n_POS < 10000:\n",
        "        batch = next(data_flow_label)\n",
        "        images = batch[0]\n",
        "        attributes = batch[1]\n",
        "        \n",
        "        # Encode images to latent space\n",
        "        z = vae.encoder.predict(images, verbose=0)\n",
        "        \n",
        "        # Split by attribute value\n",
        "        z_POS = z[attributes == 1]\n",
        "        z_NEG = z[attributes == -1]\n",
        "        \n",
        "        # Update positive mean\n",
        "        if len(z_POS) > 0:\n",
        "            current_sum_POS += np.sum(z_POS, axis=0)\n",
        "            current_n_POS += len(z_POS)\n",
        "            new_mean_POS = current_sum_POS / current_n_POS\n",
        "            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)\n",
        "        else:\n",
        "            movement_POS = 0\n",
        "        \n",
        "        # Update negative mean\n",
        "        if len(z_NEG) > 0:\n",
        "            current_sum_NEG += np.sum(z_NEG, axis=0)\n",
        "            current_n_NEG += len(z_NEG)\n",
        "            new_mean_NEG = current_sum_NEG / current_n_NEG\n",
        "            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)\n",
        "        else:\n",
        "            movement_NEG = 0\n",
        "        \n",
        "        # Compute attribute vector (difference of means)\n",
        "        current_vector = new_mean_POS - new_mean_NEG\n",
        "        new_dist = np.linalg.norm(current_vector)\n",
        "        dist_change = new_dist - current_dist\n",
        "        \n",
        "        print(f'{current_n_POS:6d} : {movement_POS:.4f} : {movement_NEG:.4f} : '\n",
        "              f'{new_dist:.4f} : {dist_change:+.4f}')\n",
        "        \n",
        "        # Update state\n",
        "        current_mean_POS = new_mean_POS.copy()\n",
        "        current_mean_NEG = new_mean_NEG.copy()\n",
        "        current_dist = new_dist\n",
        "        \n",
        "        # Convergence check: stop when means stabilize\n",
        "        if movement_POS + movement_NEG < 0.08:\n",
        "            current_vector = current_vector / current_dist  # Normalize\n",
        "            print(f'✓ Found {label} vector (normalized)')\n",
        "            break\n",
        "    \n",
        "    return current_vector"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def add_vector_to_images(feature_vec, n_to_show=5):\n",
        "    \"\"\"\n",
        "    Visualize the effect of adding/subtracting an attribute vector.\n",
        "    \n",
        "    Shows original face and variations with different amounts of the\n",
        "    attribute vector added (from -4 to +4).\n",
        "    \n",
        "    Args:\n",
        "        feature_vec (np.ndarray): Attribute vector to apply\n",
        "        n_to_show (int): Number of faces to show\n",
        "    \"\"\"\n",
        "    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]\n",
        "    \n",
        "    # Load sample images\n",
        "    example_batch = next(data_flow_generic)\n",
        "    example_images = example_batch[0]\n",
        "    \n",
        "    # Encode to latent space\n",
        "    z_points = vae.encoder.predict(example_images)\n",
        "    \n",
        "    # Create figure\n",
        "    fig = plt.figure(figsize=(18, 10))\n",
        "    counter = 1\n",
        "    \n",
        "    for i in range(n_to_show):\n",
        "        # Show original image\n",
        "        ax = fig.add_subplot(n_to_show, len(factors) + 1, counter)\n",
        "        ax.axis('off')\n",
        "        ax.imshow(example_images[i].squeeze())\n",
        "        if i == 0:\n",
        "            ax.set_title('Original', fontsize=8)\n",
        "        counter += 1\n",
        "        \n",
        "        # Apply attribute vector with different strengths\n",
        "        for factor in factors:\n",
        "            # Vector arithmetic: z' = z + α * feature_vec\n",
        "            modified_z = z_points[i] + feature_vec * factor\n",
        "            modified_image = vae.decoder.predict(np.array([modified_z]))[0]\n",
        "            \n",
        "            ax = fig.add_subplot(n_to_show, len(factors) + 1, counter)\n",
        "            ax.axis('off')\n",
        "            ax.imshow(modified_image.squeeze())\n",
        "            if i == 0:\n",
        "                ax.set_title(f'{factor:+d}', fontsize=8)\n",
        "            counter += 1\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()"
    ]
})

# =============================================================================
# Cell 23-26: Compute Attribute Vectors
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Compute Attribute Vectors"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"scrolled": True},
    "outputs": [],
    "source": [
        "# Batch size for attribute vector computation\n",
        "ATTR_BATCH_SIZE = 500\n",
        "\n",
        "# Compute vectors for various attributes\n",
        "attractive_vec = get_vector_from_label('Attractive', ATTR_BATCH_SIZE)\n",
        "mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', ATTR_BATCH_SIZE)\n",
        "smiling_vec = get_vector_from_label('Smiling', ATTR_BATCH_SIZE)\n",
        "lipstick_vec = get_vector_from_label('Wearing_Lipstick', ATTR_BATCH_SIZE)\n",
        "cheekbones_vec = get_vector_from_label('High_Cheekbones', ATTR_BATCH_SIZE)\n",
        "male_vec = get_vector_from_label('Male', ATTR_BATCH_SIZE)"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"scrolled": True},
    "outputs": [],
    "source": [
        "# Additional attribute vectors\n",
        "eyeglasses_vec = get_vector_from_label('Eyeglasses', ATTR_BATCH_SIZE)"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"scrolled": True},
    "outputs": [],
    "source": [
        "blonde_vec = get_vector_from_label('Blond_Hair', ATTR_BATCH_SIZE)"
    ]
})

# =============================================================================
# Cell 27: Apply Attribute Vectors
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Apply Attribute Vectors"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize attribute vector effects\n",
        "# Uncomment the vectors you want to visualize\n",
        "\n",
        "# print('Attractive Vector')\n",
        "# add_vector_to_images(attractive_vec)\n",
        "\n",
        "# print('Mouth Open Vector')\n",
        "# add_vector_to_images(mouth_open_vec)\n",
        "\n",
        "# print('Smiling Vector')\n",
        "# add_vector_to_images(smiling_vec)\n",
        "\n",
        "# print('Lipstick Vector')\n",
        "# add_vector_to_images(lipstick_vec)\n",
        "\n",
        "# print('High Cheekbones Vector')\n",
        "# add_vector_to_images(cheekbones_vec)\n",
        "\n",
        "# print('Male Vector')\n",
        "# add_vector_to_images(male_vec)\n",
        "\n",
        "print('Eyeglasses Vector')\n",
        "add_vector_to_images(eyeglasses_vec)\n",
        "\n",
        "# print('Blonde Hair Vector')\n",
        "# add_vector_to_images(blonde_vec)"
    ]
})

# =============================================================================
# Cell 28-31: Face Morphing
# =============================================================================
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Face Morphing\n",
        "\n",
        "Interpolate between two faces in latent space.\n",
        "This creates a smooth transition from one face to another,\n",
        "demonstrating that the latent space is continuous and meaningful."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def morph_faces(start_image_file, end_image_file):\n",
        "    \"\"\"\n",
        "    Create a morph sequence between two faces.\n",
        "    \n",
        "    Interpolates between two images in latent space and decodes\n",
        "    intermediate points to create a smooth transition.\n",
        "    \n",
        "    Args:\n",
        "        start_image_file (str): Filename of starting image\n",
        "        end_image_file (str): Filename of ending image\n",
        "    \"\"\"\n",
        "    factors = np.arange(0, 1, 0.1)\n",
        "    \n",
        "    # Load the two specific images\n",
        "    att_specific = att[att['image_id'].isin([start_image_file, end_image_file])]\n",
        "    att_specific = att_specific.reset_index()\n",
        "    data_flow_label = imageLoader.build(att_specific, 2)\n",
        "    \n",
        "    example_batch = next(data_flow_label)\n",
        "    example_images = example_batch[0]\n",
        "    \n",
        "    # Encode both images\n",
        "    z_points = vae.encoder.predict(example_images)\n",
        "    \n",
        "    # Create morph sequence\n",
        "    fig = plt.figure(figsize=(18, 8))\n",
        "    counter = 1\n",
        "    \n",
        "    # Show start image\n",
        "    ax = fig.add_subplot(1, len(factors) + 2, counter)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(example_images[0].squeeze())\n",
        "    ax.set_title('Start', fontsize=10)\n",
        "    counter += 1\n",
        "    \n",
        "    # Interpolate between start and end\n",
        "    for factor in factors:\n",
        "        # Linear interpolation: z = (1-α) * z_start + α * z_end\n",
        "        interpolated_z = z_points[0] * (1 - factor) + z_points[1] * factor\n",
        "        interpolated_image = vae.decoder.predict(np.array([interpolated_z]))[0]\n",
        "        \n",
        "        ax = fig.add_subplot(1, len(factors) + 2, counter)\n",
        "        ax.axis('off')\n",
        "        ax.imshow(interpolated_image.squeeze())\n",
        "        counter += 1\n",
        "    \n",
        "    # Show end image\n",
        "    ax = fig.add_subplot(1, len(factors) + 2, counter)\n",
        "    ax.axis('off')\n",
        "    ax.imshow(example_images[1].squeeze())\n",
        "    ax.set_title('End', fontsize=10)\n",
        "    \n",
        "    plt.suptitle('Face Morphing via Latent Space Interpolation', fontsize=12)\n",
        "    plt.tight_layout()\n",
        "    plt.show()"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Example morphs between different faces\n",
        "morph_faces('000238.jpg', '000193.jpg')  # Person to glasses"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "morph_faces('000112.jpg', '000258.jpg')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "morph_faces('000230.jpg', '000712.jpg')"
    ]
})

# =============================================================================
# Cell 32-33: Cleanup
# =============================================================================
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
        "# Uncomment and run this cell only after all work is complete and saved.\n",
        "\n",
        "# import IPython\n",
        "# print(\"Restarting kernel to release GPU memory...\")\n",
        "# IPython.Application.instance().kernel.do_shutdown(restart=True)"
    ]
})

# =============================================================================
# Save notebook
# =============================================================================
nb["cells"] = new_cells

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Notebook standardized!")
print(f"  Total cells: {len(new_cells)}")
print("  - GPU setup with error handling")
print("  - ALL_CAPS config variables")
print("  - PEP-8 compliant code")
print("  - Comprehensive docstrings and comments")
print("  - Kernel restart commented out")
