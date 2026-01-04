#!/usr/bin/env python3
"""
Add GPU Memory Analysis section to the DNN notebook.
Includes visualization and research-oriented explanation.
"""

import json

NOTEBOOK_PATH = "v1/notebooks/02_01_deep_learning_deep_neural_network.ipynb"

# Load notebook
with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

# First, update the GPUMemoryLogger to store history
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "class GPUMemoryLogger" in source:
        print(f"Found GPUMemoryLogger at index {i}")
        
        new_source = [
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "# GPU Memory Logger Callback\n",
            "# ═══════════════════════════════════════════════════════════════════════════════\n",
            "class GPUMemoryLogger(tf.keras.callbacks.Callback):\n",
            '    """Log GPU memory usage at end of each epoch and store history."""\n',
            "    \n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        self.memory_history = []  # Store (current_mb, peak_mb) per epoch\n",
            "    \n",
            "    def on_epoch_end(self, epoch, logs=None):\n",
            "        try:\n",
            "            info = tf.config.experimental.get_memory_info('GPU:0')\n",
            "            current_mb = info.get('current', 0) / (1024 ** 2)\n",
            "            peak_mb = info.get('peak', 0) / (1024 ** 2)\n",
            "            self.memory_history.append((current_mb, peak_mb))\n",
            '            print(f"  GPU Memory: {current_mb:.0f} MB (peak: {peak_mb:.0f} MB)")\n',
            "            \n",
            "            # Log to W&B\n",
            "            if wandb.run:\n",
            "                wandb.log({\n",
            '                    "gpu_memory_mb": current_mb,\n',
            '                    "gpu_memory_peak_mb": peak_mb\n',
            "                })\n",
            "        except Exception:\n",
            "            self.memory_history.append((0, 0))\n",
            "\n",
            "# Instantiate callback (will be added to callbacks list)\n",
            "gpu_memory_logger = GPUMemoryLogger()\n",
            "\n",
            "# Compile model with optimal learning rate\n",
            "model.compile(\n",
            "    loss='categorical_crossentropy',\n",
            "    optimizer=Adam(learning_rate=LEARNING_RATE),\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "# Define full callback stack\n",
            "callbacks = [\n",
            "    WandbMetricsLogger(),                              # W&B logging\n",
            "    get_lr_scheduler(monitor='val_loss', patience=2),  # Reduce LR on plateau\n",
            "    get_early_stopping(monitor='val_loss', patience=5),# Stop if no improvement\n",
            "    LRLogger(),                                        # Log learning rate\n",
            "    gpu_memory_logger,                                 # Log GPU memory usage\n",
            "]\n",
            "\n",
            "# Train the model\n",
            "history = model.fit(\n",
            "    x_train,\n",
            "    y_train,\n",
            "    batch_size=BATCH_SIZE,\n",
            "    epochs=EPOCHS,\n",
            "    shuffle=True,\n",
            "    validation_data=(x_test, y_test),\n",
            "    callbacks=callbacks\n",
            ")"
        ]
        nb["cells"][i]["source"] = new_source
        print("  Updated GPUMemoryLogger with history tracking")
        break

# Find the Training Visualization cell and add GPU Memory Analysis after it
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "## Model Evaluation" in source:
        print(f"Found Model Evaluation header at index {i}")
        
        # Create GPU Memory Analysis markdown header
        gpu_header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## GPU Memory Analysis\n",
                "\n",
                "Understanding GPU memory utilization is crucial for optimizing deep learning workloads.\n",
                "This section analyzes the memory footprint of our MLP model."
            ]
        }
        
        # Create GPU Memory Analysis code cell
        gpu_analysis = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ═══════════════════════════════════════════════════════════════════════════════\n",
                "# GPU MEMORY ANALYSIS\n",
                "# ═══════════════════════════════════════════════════════════════════════════════\n",
                "\n",
                "# Extract GPU memory history\n",
                "if gpu_memory_logger.memory_history:\n",
                "    current_memory = [m[0] for m in gpu_memory_logger.memory_history]\n",
                "    peak_memory = [m[1] for m in gpu_memory_logger.memory_history]\n",
                "    epochs_range = range(1, len(current_memory) + 1)\n",
                "    \n",
                "    # Plot GPU memory over epochs\n",
                "    fig, ax = plt.subplots(figsize=(10, 5))\n",
                "    \n",
                "    ax.plot(epochs_range, current_memory, 'b-', linewidth=2, label='Current Memory')\n",
                "    ax.plot(epochs_range, peak_memory, 'r--', linewidth=2, label='Peak Memory')\n",
                "    ax.axhline(y=GPU_VRAM_GB * 1024, color='g', linestyle=':', linewidth=2, \n",
                "               label=f'Total VRAM ({GPU_VRAM_GB} GB)')\n",
                "    \n",
                "    ax.set_xlabel('Epoch', fontsize=12)\n",
                "    ax.set_ylabel('GPU Memory (MB)', fontsize=12)\n",
                "    ax.set_title('GPU Memory Utilization During Training', fontsize=14)\n",
                "    ax.legend()\n",
                "    ax.grid(True, alpha=0.3)\n",
                "    \n",
                "    # Set y-axis to show full VRAM for context\n",
                "    ax.set_ylim(0, max(peak_memory) * 1.5)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "    \n",
                "    # Print memory statistics\n",
                "    print(f\"\\n{'='*60}\")\n",
                "    print(\"GPU MEMORY ANALYSIS\")\n",
                "    print(f\"{'='*60}\")\n",
                "    print(f\"  Model Parameters    : {model.count_params():,}\")\n",
                "    print(f\"  Model Memory (est.) : {model.count_params() * 16 / (1024**2):.1f} MB\")\n",
                "    print(f\"  Average GPU Memory  : {np.mean(current_memory):.0f} MB\")\n",
                "    print(f\"  Peak GPU Memory     : {max(peak_memory):.0f} MB\")\n",
                "    print(f\"  Total VRAM Available: {GPU_VRAM_GB * 1024} MB\")\n",
                "    print(f\"  VRAM Utilization    : {max(peak_memory) / (GPU_VRAM_GB * 1024) * 100:.1f}%\")\n",
                "    print(f\"{'='*60}\")\n",
                "else:\n",
                "    print(\"GPU memory data not available.\")"
            ]
        }
        
        # Create explanation markdown cell
        gpu_explanation = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Why is GPU Memory Utilization Low?\n",
                "\n",
                "**From a research perspective**, GPU memory consumption in neural networks is determined by:\n",
                "\n",
                "1. **Model Parameters (Weights)**  \n",
                "   - Each parameter requires 4 bytes (float32)\n",
                "   - Our MLP has ~650K parameters ≈ 2.5 MB for weights alone\n",
                "\n",
                "2. **Optimizer State (Adam)**  \n",
                "   - Adam stores first moment (m) and second moment (v) for each parameter\n",
                "   - This triples the parameter memory: 2.5 MB × 3 ≈ 7.5 MB\n",
                "\n",
                "3. **Gradients**  \n",
                "   - One gradient per parameter during backpropagation\n",
                "   - Additional 2.5 MB during training\n",
                "\n",
                "4. **Activations (Batch-dependent)**  \n",
                "   - Activations are stored for backpropagation\n",
                "   - Memory scales with: `batch_size × layer_outputs × 4 bytes`\n",
                "   - For our small MLP: ~1-5 MB per batch\n",
                "\n",
                "**Total Model Memory**: ~10-20 MB (regardless of batch size)\n",
                "\n",
                "---\n",
                "\n",
                "### Implications for Model Selection\n",
                "\n",
                "| Model Type | Typical Parameters | Expected VRAM |\n",
                "|------------|-------------------|---------------|\n",
                "| Simple MLP (this notebook) | ~650K | 50-200 MB |\n",
                "| CNN (CIFAR-10) | ~1-5M | 200-500 MB |\n",
                "| VAE (CelebA 128×128) | ~5-20M | 2-4 GB |\n",
                "| GAN (WGAN-GP) | ~20-50M | 4-8 GB |\n",
                "| Transformers | ~100M+ | 8+ GB |\n",
                "\n",
                "**Key Insight**: To fully utilize GPU memory, use larger models (CNNs, VAEs, GANs) \n",
                "or higher-resolution inputs. The dynamic batch finder optimizes batch size, but \n",
                "small models inherently cannot consume large amounts of VRAM."
            ]
        }
        
        # Insert before Model Evaluation (in reverse order)
        nb["cells"].insert(i, gpu_explanation)
        nb["cells"].insert(i, gpu_analysis)
        nb["cells"].insert(i, gpu_header)
        print("  Added GPU Memory Analysis section")
        break

# Save
with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("\n✓ GPU Memory Analysis section added!")
print("  - Updated GPUMemoryLogger with history tracking")
print("  - Added GPU memory plot over epochs")
print("  - Added research-oriented explanation")
