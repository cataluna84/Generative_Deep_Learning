# Notebook Development & Standardization Guide

This comprehensive guide covers notebook development workflow, debugging, and the standardization process for V1/V2 notebooks.

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
| `.h5` weights | `.weights.h5` or `.keras` |
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

> [!TIP]
> **Batch Size Optimization**: For 8GB VRAM GPUs:
> - Simple datasets (MNIST/CIFAR): `BATCH_SIZE = 1024`
> - Face datasets (CelebA): `BATCH_SIZE = 256-384`
> - Use `nvidia-smi -l 1` to monitor memory usage.

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

For VAEs with custom loss, define reconstruction loss:

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

---

## Checklist

- [ ] GPU memory growth enabled in first cell
- [ ] Global config at top (BATCH_SIZE, EPOCHS, etc.)
- [ ] W&B init with `learning_rate: "auto"`
- [ ] LRFinder on cloned model
- [ ] Training with callbacks (`WandbMetricsLogger`, `get_lr_scheduler`, `get_early_stopping`, `LRLogger`)
- [ ] Post-training history plot with `semilogy()` for LR
- [ ] `wandb.finish()` at end

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
sys.path.insert(0, '..')  # For v1/notebooks
# sys.path.insert(0, '../..')  # For v2 subdirectories

# Project utilities
from utils.wandb_utils import init_wandb
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger

# W&B
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Visualization
import matplotlib.pyplot as plt
```

---

## Related Documentation

- **[CALLBACKS.md](CALLBACKS.md)** - Full callback reference
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - W&B account setup
- **[GPU_SETUP.md](GPU_SETUP.md)** - GPU configuration & batch sizes
- **[UV_SETUP.md](UV_SETUP.md)** - UV package manager
- **[CELEBA_SETUP.md](CELEBA_SETUP.md)** - CelebA dataset setup
