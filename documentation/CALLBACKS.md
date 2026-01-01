# Keras Callbacks Reference

This document covers the custom Keras callbacks available in `utils/callbacks.py`.

---

## LRFinder

Finds the optimal learning rate by training with exponentially increasing LR.

```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(x, y, epochs=2, callbacks=[lr_finder])

lr_finder.plot_loss()              # Visualize loss vs LR
optimal_lr = lr_finder.get_optimal_lr()  # Get recommended LR
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_lr` | `1e-6` | Starting learning rate |
| `max_lr` | `1` | Maximum learning rate |
| `steps` | `100` | Number of LR steps to test |
| `beta` | `0.98` | Smoothing factor for loss |

### Selection Methods (Color-Coded)

The `get_optimal_lr()` method supports multiple selection strategies:

| Color | Method | Description | Risk Level |
|-------|--------|-------------|------------|
| 🔴 Red | `'steepest'` | LR at steepest descent | Aggressive |
| 🟠 Orange | `'recommended'` ★ | Steepest / 3 (fastai-style) | **Balanced (DEFAULT)** |
| 🟣 Purple | `'valley'` | LR at 80% loss decline | Robust |
| 🟢 Green | `'min_loss_10'` | Min loss LR / 10 | Conservative |

### Explicit Method Selection

```python
# Use default (recommended - steepest/3)
lr = lr_finder.get_optimal_lr()

# Override with specific method
lr = lr_finder.get_optimal_lr(method='steepest')    # Aggressive
lr = lr_finder.get_optimal_lr(method='valley')      # Robust
lr = lr_finder.get_optimal_lr(method='min_loss_10') # Conservative
```

### VAE LRFinder

For VAEs with custom loss, define a reconstruction loss before running LRFinder:

```python
import keras.backend as K
from keras.optimizers import Adam

# Clone model for LR finding
lr_model = tf.keras.models.clone_model(vae.model)

# Define reconstruction loss
def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])

# Compile with custom loss
lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))

# Run LRFinder
lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x_train, x_train, epochs=2, callbacks=[lr_finder])
```

---

## get_lr_scheduler

Reduces learning rate when loss plateaus using `ReduceLROnPlateau`.

```python
from utils.callbacks import get_lr_scheduler

# For training WITH validation data
scheduler = get_lr_scheduler(monitor='val_loss', patience=2, factor=0.5)

# For training WITHOUT validation data
scheduler = get_lr_scheduler(monitor='loss', patience=5, factor=0.5)

model.fit(x, y, callbacks=[scheduler])
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `'val_loss'` | Metric to monitor |
| `patience` | `2` | Epochs without improvement before reducing |
| `factor` | `0.5` | LR multiplier when reducing |
| `min_lr` | `1e-6` | Minimum learning rate |

---

## get_early_stopping

Stops training when loss stops improving using `EarlyStopping`.

```python
from utils.callbacks import get_early_stopping

early_stop = get_early_stopping(
    monitor='loss',      # Use 'loss' without validation, 'val_loss' with validation
    patience=10,         # Stop after 10 epochs without improvement
    min_delta=1e-4,      # Minimum change to qualify as improvement
    restore_best_weights=True
)

model.fit(x, y, epochs=200, callbacks=[early_stop])
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `'loss'` | Metric to monitor |
| `patience` | `10` | Epochs without improvement before stopping |
| `min_delta` | `1e-4` | Minimum improvement threshold |
| `restore_best_weights` | `True` | Restore weights from best epoch |
| `verbose` | `1` | Print when early stopping triggers |

---

## LRLogger

Logs the learning rate at the end of each epoch to console and W&B. Useful for debugging schedulers.

```python
from utils.callbacks import LRLogger

lr_logger = LRLogger()
model.fit(x, y, callbacks=[lr_logger])
```

Output: `Epoch 1: LR = 1.0000e-03`

---

## Recommended Callback Stack

For typical training without validation data:

```python
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from wandb.integration.keras import WandbMetricsLogger
from keras.optimizers import Adam

# Step 1: Find optimal LR on cloned model
lr_model = tf.keras.models.clone_model(model)
lr_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-6))

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x, y, batch_size=BATCH_SIZE, epochs=2, callbacks=[lr_finder])
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()

# Step 2: Compile main model with optimal LR
model.compile(optimizer=Adam(learning_rate=optimal_lr), loss='mse')

# Step 3: Train with full callback stack
callbacks = [
    WandbMetricsLogger(),                              # W&B logging
    get_lr_scheduler(monitor='loss', patience=5),      # Reduce LR on plateau
    get_early_stopping(monitor='loss', patience=10),   # Stop if no improvement
    LRLogger(),                                        # Log learning rate
]

model.fit(x, y, epochs=200, callbacks=callbacks)
```

---

## Import Summary

```python
from utils.callbacks import (
    LRFinder,           # Learning rate finder
    get_lr_scheduler,   # ReduceLROnPlateau wrapper
    get_early_stopping, # EarlyStopping wrapper
    LRLogger,           # Learning rate logger
)
```

---

## Related Documentation

- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Complete standardization workflow
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - W&B integration
- **[GPU_SETUP.md](GPU_SETUP.md)** - GPU configuration and batch sizes
