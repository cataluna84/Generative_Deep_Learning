# Training Optimization Guide

Comprehensive guide for training optimization: callbacks, dynamic batch sizing, and W&B integration.

---

## Dynamic Batch Size Finder

Automatically find optimal batch size using binary search with OOM detection.

### Quick Start

```python
from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs

# Build model first
model = create_my_model()

# Find optimal batch size
BATCH_SIZE = find_optimal_batch_size(
    model=model,
    input_shape=(28, 28, 1),
)

# Scale epochs to maintain training volume
EPOCHS = calculate_adjusted_epochs(
    reference_epochs=200,
    reference_batch=32,
    actual_batch=BATCH_SIZE,
)
```

### API Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | Required | Compiled Keras model |
| `input_shape` | Required | Shape without batch dim (H, W, C) |
| `min_batch_size` | 2 | Minimum batch to test |
| `max_batch_size` | 4096 | Maximum batch to test |
| `safety_factor` | 0.9 | Return batch × this factor |

### Example Output

```
DYNAMIC BATCH SIZE FINDER
Model Parameters: 1,234,567
Estimated Model Memory: 19.8 MB
  batch_size=   64 ✓
  batch_size=  512 ✓
  batch_size= 1024 ✗ OOM
✓ Optimal batch size: 460
```

---

## Keras Callbacks

Custom callbacks in `utils/callbacks.py`.

### LRFinder

Find optimal learning rate by training with exponentially increasing LR.

```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(x, y, epochs=2, callbacks=[lr_finder])

lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()
```

**Selection Methods:**

| Color | Method | Description |
|-------|--------|-------------|
| 🔴 | `'steepest'` | Aggressive |
| 🟠 | `'recommended'` ★ | Steepest / 3 (DEFAULT) |
| 🟣 | `'valley'` | 80% loss decline |
| 🟢 | `'min_loss_10'` | Min loss / 10 |

### VAE LRFinder

```python
import keras.backend as K

def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])

lr_model = tf.keras.models.clone_model(vae.model)
lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

### LR Scheduler

Reduce learning rate on plateau.

```python
from utils.callbacks import get_lr_scheduler

# Without validation data
scheduler = get_lr_scheduler(monitor='loss', patience=5, factor=0.5)

# With validation data
scheduler = get_lr_scheduler(monitor='val_loss', patience=2, factor=0.5)
```

### Early Stopping

```python
from utils.callbacks import get_early_stopping

early_stop = get_early_stopping(
    monitor='loss',
    patience=10,
    min_delta=1e-4,
    restore_best_weights=True
)
```

### LR Logger

Log learning rate each epoch to console and W&B.

```python
from utils.callbacks import LRLogger
lr_logger = LRLogger()
```

---

## W&B Integration

### Setup

1. Create account at [wandb.ai](https://wandb.ai)
2. Add to `.env`:
   ```env
   WANDB_API_KEY=your_api_key
   WANDB_PROJECT=generative-deep-learning
   ```
3. Login: `uv run wandb login`

### Basic Integration

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(
    project="generative-deep-learning",
    name="vae-experiment-1",
    config={
        "learning_rate": "auto",
        "batch_size": 384,
        "epochs": 200,
    }
)

model.fit(x, y, callbacks=[WandbMetricsLogger()])
wandb.finish()
```

### Helper Functions

```python
from utils.wandb_utils import init_wandb, log_images

run = init_wandb(name="vae-faces-v1", config={...})
log_images(generated_batch[:16], key="generated_images")
```

### Update Config After LRFinder

```python
wandb.config.update({"learning_rate": optimal_lr})
```

---

## Recommended Training Stack

```python
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from wandb.integration.keras import WandbMetricsLogger

# Step 1: Find optimal LR on cloned model
lr_model = tf.keras.models.clone_model(model)
lr_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-6))
lr_finder = LRFinder()
lr_model.fit(x, y, epochs=2, callbacks=[lr_finder])
optimal_lr = lr_finder.get_optimal_lr()

# Step 2: Update W&B
wandb.config.update({"learning_rate": optimal_lr})

# Step 3: Train with full stack
callbacks = [
    WandbMetricsLogger(),
    get_lr_scheduler(monitor='loss', patience=5),
    get_early_stopping(monitor='loss', patience=10),
    LRLogger(),
]
model.fit(x, y, epochs=200, callbacks=callbacks)
wandb.finish()
```

---

## Run Folder Management

### Configuration Variables

Each training notebook includes these configuration variables:

```python
SECTION = 'gan'              # Parent folder type
DATASET_RUN_ID = '0002'      # Dataset identifier (groups related experiments)
DATA_NAME = 'horses'         # Human-readable dataset name
EXPERIMENT_RUN_ID = None     # Set to None for auto-increment, or specify ID
```

### Auto-Increment Logic

When `EXPERIMENT_RUN_ID = None`:
1. Scans `../run/{section}/{DATASET_RUN_ID}_{DATA_NAME}/` for existing folders
2. Finds all 3-digit numeric subdirectories (e.g., `001`, `002`, `005`)
3. Returns next sequential ID (e.g., `006`)
4. Falls back to `'001'` if no prior runs exist

### Folder Structure

```
v1/run/{section}/{DATASET_RUN_ID}_{DATA_NAME}/{EXPERIMENT_RUN_ID}/
├── {EXPERIMENT_RUN_ID}_analysis_report.md
├── model.keras
├── images/          # Generated samples (every 500 epochs)
├── viz/             # Training visualizations
└── weights/         # Model checkpoints (every 500 epochs)
```

### Master Experiment Log

Each notebook maintains a "Master Experiment Log" at the end, tracking:
- Run ID
- Date
- W&B URL
- Key hyperparameters
- Stability status
- Final losses
- Notes

After training, manually add your run to the log to maintain experiment history.

---

## Related Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Installation and setup
- **[GAN_GUIDE.md](GAN_GUIDE.md)** - GAN-specific training
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Complete workflow
