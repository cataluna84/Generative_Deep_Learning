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

GANs use custom training loops, so standard Keras callbacks don't apply. The WGAN implementation includes comprehensive per-epoch metrics logging.

### GAN W&B Initialization

```python
from utils.wandb_utils import init_wandb, define_wgan_charts

# Initialize W&B with hyperparameters
run = init_wandb(
    name=f"wgan_{DATASET}_001",
    project="generative-deep-learning",
    config={
        "model": "WGAN",
        "dataset": DATASET,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "n_critic": N_CRITIC,
        "clip_threshold": CLIP_THRESHOLD,
        "z_dim": Z_DIM,
    }
)

# Configure 23 W&B charts with step_metric='epoch'
define_wgan_charts()
```

### GAN Training with Per-Epoch Logging

The WGAN training method logs 23 metrics to W&B every epoch:

```python
gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    n_critic=N_CRITIC,
    clip_threshold=CLIP_THRESHOLD,
    verbose=True,              # Enable detailed console output
    quality_metrics_every=100, # FID/IS every 100 epochs
    wandb_log=True             # Enable W&B per-epoch logging
)
```

### WGAN-GP Per-Epoch Logging

WGAN-GP (`v1/src/models/WGANGP.py`) uses `tf.GradientTape` for Keras 3.0+ compatibility and supports per-epoch W&B logging:

```python
gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    n_critic=N_CRITIC,
    using_generator=True,
    wandb_log=True  # Enable per-epoch W&B logging
)
```

**Metrics Logged Every Epoch:**

| Metric | Description |
|--------|-------------|
| `epoch` | Current epoch number |
| `d_loss/total` | Total critic loss |
| `d_loss/real` | Critic loss on real images |
| `d_loss/fake` | Critic loss on fake images |
| `d_loss/gradient_penalty` | Gradient penalty term |
| `g_loss` | Generator loss |
| `wasserstein_distance` | Estimated Wasserstein distance |
| `d_g_ratio` | Critic/Generator loss ratio |
| `generated_images` | Sample images (every N epochs) |


### Training Log Output

Each epoch displays four categories of metrics:

```
═══════════════════════════════════════════════════════════════════════════
Epoch 11999/12000 [1.46s]
───────────────────────────────────────────────────────────────────────────
  Losses     │ D: 5.4859 (R:5.4853 F:5.4865)  G: -115.549  W-dist: 115.55
  Weights    │ Critic μ:0.0024 σ:0.0096  Gen μ:0.0028 σ:0.0810
  Stability  │ D/G Ratio: 0.0475  Clip%: 0.0%  Var: 0.000002
═══════════════════════════════════════════════════════════════════════════

  Computing quality metrics...
  Quality    │ FID: 333.1  IS: 2.12±0.19  PixVar: 0.2960
```

### Metrics Categories

| Category | Metrics | Description |
|----------|---------|-------------|
| **Losses** | D, R, F, G, W-dist | Critic/Generator losses, Wasserstein distance |
| **Weights** | Critic μ/σ, Gen μ/σ | Weight statistics for pathology detection |
| **Stability** | D/G Ratio, Clip%, Var | Training balance and stability indicators |
| **Quality** | FID, IS, PixVar | Image quality (every 100 epochs) |

> [!TIP]
> See **[GAN_TRAINING_METRICS.md](GAN_TRAINING_METRICS.md)** for detailed metric formulas and interpretation.

### Master Experiment Log

Maintain a markdown table in notebooks tracking all training runs:

```markdown
| Run | Date | W&B URL | Batch Size | Epochs | LR | Stability | D Loss | G Loss | Notes |
|-----|------|---------|------------|--------|-----|-----------|--------|--------|-------|
| 001 | 2026-01-07 | [View](url) | 512 | 6000 | 5e-5 | ✅ Stable | 5.35 | -116.4 | Baseline |
| 002 | 2026-01-08 | [View](url) | 512 | 12000 | 5e-5 | ✅ Stable | 5.49 | -115.5 | Extended run |
```

### GAN Training Visualization

After training, create separate plots for each metric:

```python
import matplotlib.pyplot as plt

# Plot 1: Losses
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot([x[0] for x in gan.d_losses], label='D Loss', alpha=0.7)
ax.plot(gan.g_losses, label='G Loss', alpha=0.7)
ax.set_title('Critic/Generator Loss Over Epochs')
ax.legend()
plt.savefig(os.path.join(RUN_FOLDER, 'plots/losses.png'))

# Plot 2: Wasserstein Distance
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(gan.metrics_history['wasserstein_dist'])
ax.set_title('Wasserstein Distance (Training Progress)')
plt.savefig(os.path.join(RUN_FOLDER, 'plots/wasserstein.png'))

# Plot 3: Weight Statistics
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(gan.metrics_history['critic_weight_mean'], label='Mean')
axes[0].plot(gan.metrics_history['critic_weight_std'], label='Std')
axes[0].set_title('Critic Weight Statistics')
axes[1].plot(gan.metrics_history['generator_weight_mean'], label='Mean')
axes[1].plot(gan.metrics_history['generator_weight_std'], label='Std')
axes[1].set_title('Generator Weight Statistics')
```

### Stability Analysis Report

Generate analysis reports for each training run saved to `run/<model>/<run_id>/`:

```markdown
# Training Stability Analysis Report

## Run Summary
- **Run ID**: 0002_horses
- **Date**: 2026-01-08
- **Final Epoch**: 12000
- **W&B URL**: [View Dashboard](https://wandb.ai/...)

## Stability Assessment: ✅ STABLE

### Key Indicators
| Metric | Final Value | Status |
|--------|-------------|--------|
| D/G Ratio | 0.0475 | ✅ Balanced |
| Clip% | 0.0% | ✅ Within bounds |
| Loss Variance | 0.000002 | ✅ Stable |

### Quality Metrics (Final)
| Metric | Value | Assessment |
|--------|-------|------------|
| FID | 333.1 | Fair (CIFAR-32 typical) |
| IS | 2.12±0.19 | Moderate |
| PixVar | 0.296 | ✅ Good diversity |
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

- **[QUICKSTART.md](QUICKSTART.md)** - Installation and GPU setup
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Callbacks, batch sizing, W&B integration
- **[GAN_GUIDE.md](GAN_GUIDE.md)** - GAN metrics, stability, and triage
- **[TRAINING_STABILITY_ANALYSIS_TEMPLATE.md](TRAINING_STABILITY_ANALYSIS_TEMPLATE.md)** - Analysis report template

---

## CelebA Dataset Setup

The CelebA dataset is required for face-related notebooks (`03_05_vae_faces_train`, `04_03_wgangp_faces_train`).

### Automated Download

```bash
bash v1/data_download_scripts/download_celeba_kaggle.sh
```

**Prerequisites:** Kaggle credentials in `.env`:
```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Manual Download

1. Download from [Kaggle CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
2. Extract to `v1/data/img_align_celeba/images/`
3. Verify: `ls v1/data/img_align_celeba/images/*.jpg | wc -l` → 202599
