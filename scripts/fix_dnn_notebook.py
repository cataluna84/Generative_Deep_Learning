#!/usr/bin/env python3
"""
Update notebook to import GPUMemoryLogger from utils/callbacks.py
"""

import json

NOTEBOOK_PATH = "v1/notebooks/02_01_deep_learning_deep_neural_network.ipynb"

# Load notebook
with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

# Update imports cell to include GPUMemoryLogger
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "from utils.callbacks import LRFinder" in source:
        print(f"Found imports cell at index {i}")
        
        new_source = [
            "# Standard library imports\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# TensorFlow/Keras imports\n",
            "from keras.layers import Input, Flatten, Dense\n",
            "from keras.models import Model\n",
            "from keras.optimizers import Adam\n",
            "from keras.utils import to_categorical\n",
            "from keras.datasets import cifar10\n",
            "\n",
            "# Path setup for project utilities\n",
            "import sys\n",
            "sys.path.insert(0, '../..')  # For project root utils/\n",
            "\n",
            "# Project utilities\n",
            "from utils.wandb_utils import init_wandb, get_metrics_logger\n",
            "from utils.callbacks import (\n",
            "    LRFinder,\n",
            "    get_lr_scheduler,\n",
            "    get_early_stopping,\n",
            "    LRLogger,\n",
            "    GPUMemoryLogger  # Logs GPU memory per epoch\n",
            ")\n",
            "from utils.gpu_utils import (\n",
            "    find_optimal_batch_size,\n",
            "    calculate_adjusted_epochs,\n",
            "    get_gpu_vram_gb,\n",
            "    print_training_config\n",
            ")\n",
            "\n",
            "# W&B\n",
            "import wandb\n",
            "from wandb.integration.keras import WandbMetricsLogger"
        ]
        nb["cells"][i]["source"] = new_source
        print("  Updated imports to include GPUMemoryLogger")
        break

# Update Train Model cell to remove inline GPUMemoryLogger definition
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "class GPUMemoryLogger" in source:
        print(f"Found Train Model cell with inline class at index {i}")
        
        # Simplified training cell - no inline class definition
        new_source = [
            "# Instantiate GPU memory logger (stores history for visualization)\n",
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
        print("  Removed inline class, now imports from utils.callbacks")
        break

# Save
with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("\n✓ Notebook updated!")
print("  - GPUMemoryLogger now imported from utils.callbacks")
print("  - Inline class definition removed from training cell")
