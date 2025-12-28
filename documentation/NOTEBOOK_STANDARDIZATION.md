# Notebook Development & Standardization Guide

This comprehensive guide covers notebook development workflow, debugging, and the standardization process for V1/V2 notebooks.

---

## Part 1: Running & Debugging Notebooks

### Start Jupyter Lab

```bash
uv run jupyter lab
```

Navigate to `v1/notebooks/` or `v2/*/` and open the desired notebook.

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

If kernel crashes with OOM, add to first cell:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
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
# ═══════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
BATCH_SIZE = 64
EPOCHS = 200
OPTIMIZER_NAME = 'adam'
DATASET_NAME = 'cifar10'
MODEL_TYPE = 'cnn'
```

### Step 2: W&B Initialization

Initialize W&B early with `learning_rate: "auto"`.

```python
import wandb
from utils.wandb_utils import init_wandb

run = init_wandb(
    name="notebook-name",
    project="generative-deep-learning",
    config={
        "model": MODEL_TYPE,
        "dataset": DATASET_NAME,
        "learning_rate": "auto",  # Updated after LRFinder
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
)
```

### Step 3: Learning Rate Finder

Find optimal LR using a **cloned model**.

```python
from utils.callbacks import LRFinder
import tensorflow as tf
from keras.optimizers import Adam

lr_model = tf.keras.models.clone_model(model)
lr_model.compile(loss='...', optimizer=Adam(learning_rate=1e-6), metrics=[...])

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=2, 
             callbacks=[lr_finder], verbose=0)

lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()  # Default: 'recommended' (steepest/3)
wandb.config.update({"learning_rate": optimal_lr})
```

**Selection Methods:**

| Color | Method | Description |
|-------|--------|-------------|
| 🔴 | `'steepest'` | Aggressive |
| 🟠 | `'recommended'` ★ | **DEFAULT** - Steepest / 3 |
| 🟣 | `'valley'` | Robust (80% decline) |
| 🟢 | `'min_loss_10'` | Conservative |

### Step 4: Training with Callbacks

```python
from utils.callbacks import get_lr_scheduler, get_early_stopping
from wandb.integration.keras import WandbMetricsLogger

model.compile(loss='...', optimizer=Adam(learning_rate=optimal_lr), metrics=[...])

model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        WandbMetricsLogger(),
        get_lr_scheduler(monitor='loss', patience=5),
        get_early_stopping(monitor='loss', patience=10),
    ]
)
```

### Step 5: Post-Training Visualization

```python
import matplotlib.pyplot as plt

history = model.history.history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['loss'], 'b-', linewidth=2)
axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)

if 'learning_rate' in history:
    axes[1].semilogy(history['learning_rate'], 'r-', linewidth=2)  # Log scale!
    axes[1].set_title('Learning Rate Schedule'); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
print(f"Initial: {history['loss'][0]:.4f} | Final: {history['loss'][-1]:.4f}")
```

### Step 6: Finalize

```python
wandb.finish()
```

---

## Checklist

- [ ] Global config at top
- [ ] W&B init with `learning_rate: "auto"`
- [ ] LRFinder on cloned model
- [ ] Training with callbacks (`WandbMetricsLogger`, `get_lr_scheduler`, `get_early_stopping`)
- [ ] Post-training history plot with `semilogy()` for LR
- [ ] `wandb.finish()` at end

---

## Related Documentation

- **[CALLBACKS.md](CALLBACKS.md)** - Full callback reference
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - W&B account setup
- **[GPU_SETUP.md](GPU_SETUP.md)** - GPU configuration
- **[UV_SETUP.md](UV_SETUP.md)** - UV package manager
