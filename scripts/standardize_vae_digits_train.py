#!/usr/bin/env python3
"""
Script to standardize 03_03_vae_digits_train.ipynb notebook.
Applies all 8 standardization steps including dynamic batch finder.
"""

import json
import os

# Path to notebook
NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "v1", "notebooks", "03_03_vae_digits_train.ipynb"
)


def main():
    """Apply standardization changes to the notebook."""
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    new_cells = []

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 0: Enhanced Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03_03 VAE Digits Train\n",
            "\n",
            "**Chapter 3**: Variational Autoencoders | **Notebook 3 of 6**\n",
            "\n",
            "Trains a Variational Autoencoder (VAE) on MNIST digits with:\n",
            "- 2D latent space for visualization\n",
            "- Dynamic batch size optimization\n",
            "- W&B experiment tracking\n",
            "- Step decay learning rate scheduler"
        ]
    })
    print("✅ Cell 0: Enhanced header")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 1: Imports Section Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Imports"]
    })
    print("✅ Cell 1: Section header 'Imports'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 2: GPU Setup with Error Handling
    # ═══════════════════════════════════════════════════════════════════════════
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
    print("✅ Cell 2: Robust GPU setup")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 3: PEP 8 Imports
    # ═══════════════════════════════════════════════════════════════════════════
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
            "import sys\n",
            "import os\n",
            "\n",
            "# Path setup for local imports\n",
            "sys.path.insert(0, '../../..')\n",
            "sys.path.insert(0, '../..')\n",
            "sys.path.insert(0, '..')\n",
            "\n",
            "# Local model imports\n",
            "from src.models.VAE import VariationalAutoencoder\n",
            "from src.utils.loaders import load_mnist\n",
            "\n",
            "# W&B and training utilities\n",
            "import wandb\n",
            "from wandb.integration.keras import WandbMetricsLogger\n",
            "from utils.callbacks import get_lr_scheduler, get_early_stopping, LRLogger\n",
            "from utils.wandb_utils import init_wandb\n",
            "from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs"
        ]
    })
    print("✅ Cell 3: PEP 8 imports with gpu_utils")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 4: Global Configuration (static, before model build)
    # ═══════════════════════════════════════════════════════════════════════════
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
            "# Model architecture\n",
            "INPUT_DIM = (28, 28, 1)\n",
            "Z_DIM = 2\n",
            "\n",
            "# Reference training parameters (for epoch scaling)\n",
            "REFERENCE_BATCH_SIZE = 32\n",
            "REFERENCE_EPOCHS = 200\n",
            "\n",
            "# Training parameters (will be updated after model build)\n",
            "PRINT_EVERY_N_BATCHES = 100\n",
            "INITIAL_EPOCH = 0\n",
            "\n",
            "# Experiment tracking\n",
            "OPTIMIZER_NAME = 'adam'\n",
            "DATASET_NAME = 'digits'\n",
            "MODEL_TYPE = 'vae'\n",
            "\n",
            "# Directory paths\n",
            "SECTION = 'vae'\n",
            "RUN_ID = '0002'\n",
            "RUN_FOLDER = f'../run/{SECTION}/{RUN_ID}_{DATASET_NAME}'\n",
            "\n",
            "# Create output directories\n",
            "os.makedirs(RUN_FOLDER, exist_ok=True)\n",
            "os.makedirs(os.path.join(RUN_FOLDER, 'viz'), exist_ok=True)\n",
            "os.makedirs(os.path.join(RUN_FOLDER, 'images'), exist_ok=True)\n",
            "os.makedirs(os.path.join(RUN_FOLDER, 'weights'), exist_ok=True)\n",
            "\n",
            "# Build new model or load existing\n",
            "MODE = 'build'  # 'build' | 'load'\n",
            "\n",
            "print(f\"Run folder: {RUN_FOLDER}\")"
        ]
    })
    print("✅ Cell 4: Global configuration")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 5: Load Data Section Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Data"]
    })
    print("✅ Cell 5: Section header 'Load Data'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 6: Load Data
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load MNIST dataset (preprocessed to [0,1] range)\n",
            "(x_train, y_train), (x_test, y_test) = load_mnist()\n",
            "print(f\"Training samples: {len(x_train)}, Test samples: {len(x_test)}\")"
        ]
    })
    print("✅ Cell 6: Load data")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 7: Model Architecture Section Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model Architecture"]
    })
    print("✅ Cell 7: Section header 'Model Architecture'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 8: Build VAE Model
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# BUILD VAE MODEL\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "vae = VariationalAutoencoder(\n",
            "    input_dim=INPUT_DIM,\n",
            "    # Encoder: 4 conv layers with progressive downsampling\n",
            "    encoder_conv_filters=[32, 64, 64, 64],\n",
            "    encoder_conv_kernel_size=[3, 3, 3, 3],\n",
            "    encoder_conv_strides=[1, 2, 2, 1],\n",
            "    # Decoder: 4 conv transpose layers for upsampling\n",
            "    decoder_conv_t_filters=[64, 64, 32, 1],\n",
            "    decoder_conv_t_kernel_size=[3, 3, 3, 3],\n",
            "    decoder_conv_t_strides=[1, 2, 2, 1],\n",
            "    z_dim=Z_DIM,\n",
            ")\n",
            "\n",
            "# Save or load model\n",
            "if MODE == 'build':\n",
            "    vae.save(RUN_FOLDER)\n",
            "else:\n",
            "    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.weights.h5'))"
        ]
    })
    print("✅ Cell 8: Build VAE")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 9: Encoder Summary
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["vae.encoder.summary()"]
    })
    print("✅ Cell 9: Encoder summary")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 10: Decoder Summary
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["vae.decoder.summary()"]
    })
    print("✅ Cell 10: Decoder summary")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 11: Dynamic Batch Size Section Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Dynamic Batch Size"]
    })
    print("✅ Cell 11: Section header 'Dynamic Batch Size'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 12: Find Optimal Batch Size
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# DYNAMIC BATCH SIZE CALCULATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# Uses binary search with OOM detection to find optimal batch size.\n",
            "\n",
            "BATCH_SIZE = find_optimal_batch_size(\n",
            "    model=vae.model,\n",
            "    input_shape=INPUT_DIM,\n",
            "    min_batch_size=64,\n",
            "    max_batch_size=2048,\n",
            "    log_to_wandb=False,  # Log after W&B init\n",
            ")\n",
            "\n",
            "# Scale epochs to maintain equivalent training updates\n",
            "EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)\n",
            "\n",
            "print(f\"\\nBatch size: {BATCH_SIZE} (reference: {REFERENCE_BATCH_SIZE})\")\n",
            "print(f\"Epochs: {EPOCHS} (reference: {REFERENCE_EPOCHS})\")"
        ]
    })
    print("✅ Cell 12: Dynamic batch size finder")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 13: Training Section Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Training"]
    })
    print("✅ Cell 13: Section header 'Training'")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 14: Training Hyperparameters
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# TRAINING HYPERPARAMETERS\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "LEARNING_RATE = 0.0005\n",
            "R_LOSS_FACTOR = 1000  # Reconstruction loss multiplier\n",
            "\n",
            "# VAE models cannot use LRFinder due to Lambda layer serialization issues\n",
            "OPTIMAL_LR = LEARNING_RATE\n",
            "print(f\"Using Learning Rate: {OPTIMAL_LR}\")"
        ]
    })
    print("✅ Cell 14: Training hyperparameters")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 15: W&B Init
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize W&B with training configuration\n",
            "run = init_wandb(\n",
            "    name=f\"vae_{DATASET_NAME}_{RUN_ID}\",\n",
            "    project=\"generative-deep-learning\",\n",
            "    config={\n",
            "        \"model\": MODEL_TYPE,\n",
            "        \"dataset\": DATASET_NAME,\n",
            "        \"learning_rate\": OPTIMAL_LR,\n",
            "        \"batch_size\": BATCH_SIZE,\n",
            "        \"epochs\": EPOCHS,\n",
            "        \"batch_size_source\": \"dynamic_finder\",\n",
            "        \"model_params\": vae.model.count_params(),\n",
            "    }\n",
            ")"
        ]
    })
    print("✅ Cell 15: W&B init")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 16: Compile
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compile VAE with learning rate and loss factor\n",
            "vae.compile(OPTIMAL_LR, R_LOSS_FACTOR)"
        ]
    })
    print("✅ Cell 16: Compile")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 17: Train
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"scrolled": True},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# TRAIN VAE\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "vae.train(\n",
            "    x_train,\n",
            "    batch_size=BATCH_SIZE,\n",
            "    epochs=EPOCHS,\n",
            "    run_folder=RUN_FOLDER,\n",
            "    print_every_n_batches=PRINT_EVERY_N_BATCHES,\n",
            "    initial_epoch=INITIAL_EPOCH,\n",
            "    lr_decay=1,  # Disable internal scheduler to use external\n",
            "    extra_callbacks=[\n",
            "        WandbMetricsLogger(),\n",
            "        LRLogger(),\n",
            "        get_lr_scheduler(monitor='loss', patience=5),\n",
            "        get_early_stopping(monitor='loss', patience=10),\n",
            "    ]\n",
            ")"
        ]
    })
    print("✅ Cell 17: Train")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 18: W&B Finish
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Finalize W&B run\n",
            "wandb.finish()"
        ]
    })
    print("✅ Cell 18: W&B finish")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 19: Training Visualization
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Training Visualization"]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# TRAINING VISUALIZATION\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "if hasattr(vae, 'model') and hasattr(vae.model, 'history') and vae.model.history:\n",
            "    history = vae.model.history.history\n",
            "    \n",
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    \n",
            "    # Loss plot\n",
            "    if 'loss' in history:\n",
            "        axes[0].plot(history['loss'], 'b-', linewidth=2)\n",
            "        axes[0].set_xlabel('Epoch')\n",
            "        axes[0].set_ylabel('Loss')\n",
            "        axes[0].set_title('Training Loss')\n",
            "        axes[0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # LR plot (log scale)\n",
            "    if 'learning_rate' in history:\n",
            "        axes[1].semilogy(history['learning_rate'], 'r-', linewidth=2)\n",
            "        axes[1].set_xlabel('Epoch')\n",
            "        axes[1].set_ylabel('Learning Rate (log)')\n",
            "        axes[1].set_title('Learning Rate Schedule')\n",
            "        axes[1].grid(True, which='both', alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(os.path.join(RUN_FOLDER, 'training_summary.png'), dpi=150)\n",
            "    plt.show()\n",
            "    \n",
            "    print(f\"\\n{'='*50}\")\n",
            "    print(\"TRAINING SUMMARY\")\n",
            "    print(f\"{'='*50}\")\n",
            "    print(f\"  Final Loss: {history['loss'][-1]:.6f}\")\n",
            "    print(f\"  Min Loss  : {min(history['loss']):.6f}\")\n",
            "    print(f\"  Epochs    : {len(history['loss'])}\")\n",
            "    print(f\"{'='*50}\")\n",
            "else:\n",
            "    print(\"No training history available.\")"
        ]
    })
    print("✅ Cells 19-20: Training visualization")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 21: Cleanup Header
    # ═══════════════════════════════════════════════════════════════════════════
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Cleanup: Restart Kernel to Release GPU Memory"]
    })
    print("✅ Cell 21: Cleanup header")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cell 22: Kernel Restart (Commented Out)
    # ═══════════════════════════════════════════════════════════════════════════
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
    print("✅ Cell 22: Kernel restart (commented out)")

    # Update notebook cells
    nb['cells'] = new_cells

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Notebook saved: {NOTEBOOK_PATH}")
    print("=" * 60)
    print("All standardization steps completed!")


if __name__ == "__main__":
    main()
