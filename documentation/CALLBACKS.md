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

### Selection Methods (Color-Coded)

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

---

## get_lr_scheduler

Reduces learning rate when loss plateaus.

```python
from utils.callbacks import get_lr_scheduler

# For training WITH validation data
scheduler = get_lr_scheduler(monitor='val_loss', patience=2, factor=0.5)

# For training WITHOUT validation data
scheduler = get_lr_scheduler(monitor='loss', patience=2, factor=0.5)

model.fit(x, y, callbacks=[scheduler])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `'val_loss'` | Metric to monitor |
| `patience` | `2` | Epochs without improvement before reducing |
| `factor` | `0.5` | LR multiplier when reducing |
| `min_lr` | `1e-6` | Minimum learning rate |

---

## get_early_stopping

Stops training when loss stops improving.

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `'loss'` | Metric to monitor |
| `patience` | `10` | Epochs without improvement before stopping |
| `min_delta` | `1e-4` | Minimum improvement threshold |
| `restore_best_weights` | `True` | Restore weights from best epoch |

---

## LRLogger

Logs the learning rate at the end of each epoch to console and W&B. Useful for debugging schedulers.

```python
from utils.callbacks import LRLogger

lr_logger = LRLogger()
model.fit(x, y, callbacks=[lr_logger])
```

---

## Recommended Callback Stack

For typical training without validation data:

```python
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from wandb.integration.keras import WandbMetricsLogger

# Step 1: Find optimal LR
lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
temp_model.fit(x, y, epochs=2, callbacks=[lr_finder])
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()

# Step 2: Train with full callback stack
callbacks = [
    WandbMetricsLogger(),                              # W&B logging
    get_lr_scheduler(monitor='loss', patience=5),      # Reduce LR on plateau
    get_early_stopping(monitor='loss', patience=10),   # Stop if no improvement
    LRLogger(),                                        # Log learning rate
]

model.compile(optimizer=Adam(learning_rate=optimal_lr), ...)
model.fit(x, y, epochs=200, callbacks=callbacks)
```
