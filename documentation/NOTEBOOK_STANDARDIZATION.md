# Notebook Development & Standardization Guide

This comprehensive guide covers notebook development workflow, debugging, and the standardization process for V1/V2 notebooks.

---

## Standardization Requirements

Every notebook and source file must meet the following criteria:

- [x] **PEP 8 compliant code formatting** (consistent style, clean imports)
- [x] **Comprehensive documentation and comments** (docstrings for all classes/functions)
- [x] **Dynamic batch size and epoch scaling** (using `utils.gpu_utils`)
- [x] **W&B integration** for experiment tracking (metrics, images, configs)
- [x] **LRFinder for optimal learning rate** (run on cloned model before training)
- [x] **Step decay LR scheduler** (for stable training)
- [x] **Enhanced training visualizations** (loss, accuracy, LR history)
- [x] **Kernel restart cell** for GPU memory release (final cell)

---

## Part 1: Running & Debugging Notebooks

### Start Jupyter Lab

```bash
uv run jupyter lab
```

Navigate to `v1/notebooks/` or `v2/<chapter>/` and open the desired notebook.

### Debugging Workflow

Run cells sequentially with `Shift+Enter`. When an error occurs:

1. **Read the full traceback** - identifies the exact line
2. **Check variable types** - use `type(variable)` and `variable.shape`
3. **Isolate the issue** - create a new cell to test specific operations

### Common TensorFlow 2.20+ Updates

| Old API | New API |
|---------|---------|
| `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| `keras.layers.experimental.*` | `keras.layers.*` |
| `model.predict_on_batch()` | `model(x, training=False)` |
| `.h5` weights | `.weights.h5` (weights only) |
| `.h5` model save | `.keras` (full model) |
| `tf.keras.*` | `keras.*` |

### GPU Memory Issues

If kernel crashes with OOM, add to **first cell**:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[gpu.name for gpu in gpus]}")
```

### Kernel Management

- **Restart kernel**: After modifying utility files in `utils/`
- **Clear outputs**: Before committing to reduce file size

---

## Part 2: Notebook Standardization Workflow

Transform hardcoded, static notebooks into flexible, tracked, and optimized experiments.

### Step 1: Global Configuration

Move hardcoded parameters to the top of the notebook (after imports).

#### Static Configuration (Simple)

```python
# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 384      # Optimized for 8GB VRAM with CelebA
EPOCHS = 200
OPTIMIZER_NAME = 'adam'
DATASET_NAME = 'celeba'
MODEL_TYPE = 'vae'

# Model-specific
INPUT_DIM = (128, 128, 3)
Z_DIM = 200
```

#### Dynamic Configuration (Recommended)

Use the dynamic batch size finder to automatically determine optimal batch size:

```python
from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs

# Reference values (original notebook settings)
REFERENCE_BATCH_SIZE = 32
REFERENCE_EPOCHS = 200

# NOTE: Call AFTER building model so it can test memory usage
# (Move this cell after model build)
BATCH_SIZE = find_optimal_batch_size(
    model=my_model,
    input_shape=(28, 28, 1),
)
EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)

print(f"Batch size: {BATCH_SIZE} (reference: {REFERENCE_BATCH_SIZE})")
print(f"Epochs: {EPOCHS} (reference: {REFERENCE_EPOCHS})")
```

> [!TIP]
> See **[DYNAMIC_BATCH_SIZE.md](DYNAMIC_BATCH_SIZE.md)** for full API documentation.
> The finder uses binary search + OOM detection to find the maximum safe batch size.

### Step 2: W&B Initialization

Initialize W&B early with `learning_rate: "auto"`.

```python
import wandb
from utils.wandb_utils import init_wandb

run = init_wandb(
    name="03_05_vae_faces_train",
    project="generative-deep-learning",
    config={
        "model": MODEL_TYPE,
        "dataset": DATASET_NAME,
        "learning_rate": "auto",  # Updated after LRFinder
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "z_dim": Z_DIM,
    }
)
```

### Step 3: Learning Rate Finder

Find optimal LR using a **cloned model**.

```python
from utils.callbacks import LRFinder
import tensorflow as tf
from keras.optimizers import Adam

# Clone model for LR finding
lr_model = tf.keras.models.clone_model(model)

# Standard compile (for autoencoders)
lr_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-6))

# Run LRFinder
lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=2, 
             callbacks=[lr_finder], verbose=0)

# Visualize and get optimal LR
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()  # Default: 'recommended' (steepest/3)

# Update W&B config
wandb.config.update({"learning_rate": optimal_lr})
print(f"Optimal learning rate: {optimal_lr:.2e}")
```

#### VAE LRFinder

The VAE's `sampling` function is registered with `@keras.saving.register_keras_serializable`, enabling model cloning. Define a reconstruction loss:

```python
import keras.backend as K

def vae_r_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis=[1,2,3])
    return 1000 * r_loss

lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

#### Selection Methods

| Color | Method | Description |
|-------|--------|-------------|
| 🔴 | `'steepest'` | Aggressive |
| 🟠 | `'recommended'` ★ | **DEFAULT** - Steepest / 3 |
| 🟣 | `'valley'` | Robust (80% decline) |
| 🟢 | `'min_loss_10'` | Conservative |

### Step 4: Training with Callbacks

```python
from utils.callbacks import get_lr_scheduler, get_early_stopping, LRLogger
from wandb.integration.keras import WandbMetricsLogger

# Compile with optimal LR
model.compile(
    loss='...', 
    optimizer=Adam(learning_rate=optimal_lr), 
    metrics=[...]
)

# Train with full callback stack
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        WandbMetricsLogger(),                           # W&B logging
        get_lr_scheduler(monitor='loss', patience=5),   # Reduce LR on plateau
        get_early_stopping(monitor='loss', patience=10),# Stop if no improvement
        LRLogger(),                                     # Log learning rate
    ]
)
```

### Step 5: Post-Training Visualization

```python
import matplotlib.pyplot as plt

history = model.history.history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
axes[0].plot(history['loss'], 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Over Epochs')
axes[0].grid(True, alpha=0.3)

# Plot 2: Learning Rate (LOG SCALE!)
if 'learning_rate' in history:
    axes[1].semilogy(history['learning_rate'], 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate (log scale)')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, which='both', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
    axes[1].set_title('Learning Rate (Not Available)')

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'='*50}")
print("TRAINING SUMMARY")
print(f"{'='*50}")
print(f"  Initial Loss  : {history['loss'][0]:.6f}")
print(f"  Final Loss    : {history['loss'][-1]:.6f}")
print(f"  Min Loss      : {min(history['loss']):.6f} (Epoch {history['loss'].index(min(history['loss'])) + 1})")
print(f"  Total Epochs  : {len(history['loss'])}")
if 'learning_rate' in history:
    print(f"  Final LR      : {history['learning_rate'][-1]:.2e}")
print(f"{'='*50}")
```

### Step 6: Finalize

```python
wandb.finish()
```

### Step 7: Restart Kernel to Release GPU Memory

> [!IMPORTANT]
> TensorFlow/CUDA does not fully release GPU memory within a running Python process.
> The **only guaranteed way** to release all GPU memory is to restart the kernel.

Add this as the **final cell** of your notebook:

```python
# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP: Restart kernel to fully release GPU memory
# ═══════════════════════════════════════════════════════════════════════════════
# TensorFlow/CUDA does not release GPU memory within a running Python process.
# Restarting the kernel is the only guaranteed way to free all GPU resources.

import IPython
print("Restarting kernel to release GPU memory...")
IPython.Application.instance().kernel.do_shutdown(restart=True)
```

> [!NOTE]
> This cell should only be run after all work is complete and saved.
> The kernel restart will clear all variables and outputs.

---

## Checklist

- [ ] GPU memory growth enabled in first cell
- [ ] Global config at top (BATCH_SIZE, EPOCHS, etc.)
- [ ] W&B init with `learning_rate: "auto"`
- [ ] LRFinder on cloned model
- [ ] Training with callbacks (`WandbMetricsLogger`, `get_lr_scheduler`, `get_early_stopping`, `LRLogger`)
- [ ] Post-training history plot with `semilogy()` for LR
- [ ] Model saved with `.keras` extension (not legacy `.h5`)
- [ ] Weights saved with `.weights.h5` extension
- [ ] `wandb.finish()` at end
- [ ] Kernel restart cell to release GPU memory (final cell)

---

## GAN-Specific Standardization

GANs use custom training loops, so standard Keras callbacks don't apply.

### GAN Training Configuration

```python
# LR Scheduler (Step Decay)
LR_DECAY_FACTOR = 0.5  # Halve LR at each decay point
LR_DECAY_EPOCHS = EPOCHS // 4  # Decay 4 times during training

# Training call with W&B and LR scheduling
gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=50,
    use_wandb=True,
    lr_decay_factor=LR_DECAY_FACTOR,
    lr_decay_epochs=LR_DECAY_EPOCHS
)
```

### GAN Training Plots

After training, plot loss, accuracy, and LR history:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss plot
axes[0].plot([x[0] for x in gan.d_losses], label='D')
axes[0].plot([x[0] for x in gan.g_losses], label='G')
axes[0].set_title('Loss')
axes[0].legend()

# Accuracy plot
axes[1].plot([x[3] for x in gan.d_losses], label='D')
axes[1].set_title('Accuracy')
axes[1].legend()

# LR plot (log scale)
axes[2].semilogy(gan.d_lr_history, label='D LR')
axes[2].semilogy(gan.g_lr_history, label='G LR')
axes[2].set_title('Learning Rate')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(RUN_FOLDER, 'training_summary.png'))
```

---

## Import Template

```python
# Standard library
import os
import numpy as np

# TensorFlow/Keras
import tensorflow as tf
from keras import layers, Model
from keras.optimizers import Adam
import keras.backend as K

# Path setup for utilities
import sys
sys.path.insert(0, '..')      # For v1/src modules
sys.path.insert(0, '../..')   # For project root utils/

# Project utilities (from project root)
from utils.wandb_utils import init_wandb
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from utils.gpu_utils import get_optimal_batch_size, calculate_adjusted_epochs, get_gpu_vram_gb

# W&B
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Visualization
import matplotlib.pyplot as plt
```

---

## Notebook Update Scripts

Scripts for updating notebooks are located in `scripts/` at the project root:

```bash
# Update a specific cell in a notebook
uv run python scripts/update_notebook_cell.py

# Generate standardized GAN notebook from scratch
uv run python scripts/standardize_gan_notebook.py
```

---

## Related Documentation

- **[CALLBACKS.md](CALLBACKS.md)** - Full callback reference
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - W&B account setup
- **[GPU_SETUP.md](GPU_SETUP.md)** - GPU configuration & batch sizes
- **[UV_SETUP.md](UV_SETUP.md)** - UV package manager
- **[CELEBA_SETUP.md](CELEBA_SETUP.md)** - CelebA dataset setup
