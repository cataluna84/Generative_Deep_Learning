# Learning Rate Optimization Tools

This project includes standard tools for optimizing model learning rates using Keras callbacks. These tools help in finding an optimal initial learning rate and adjusting it dynamically during training.

## 1. Learning Rate Finder (`LRFinder`)

The `LRFinder` class implements the "Learning Rate Range Test" (often associated with Leslie Smith). It trains the model for a few epochs while increasing the learning rate exponentially.

### Usage
Import the callback from `utils.callbacks` and use it on a **cloned** model (to avoid pre-training weights on your main model):

```python
from utils.callbacks import LRFinder
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Clone your model
lr_model = tf.keras.models.clone_model(model)

# 2. Compile with a low starting LR
lr_model.compile(optimizer=Adam(learning_rate=1e-6), loss='...', metrics=[...])

# 3. Running the Finder
lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x_train, y_train, epochs=2, callbacks=[lr_finder])

# 4. Automate Visualization & Selection
# Use the built-in helper to plot loss and mark the optimal learning rate
lr_finder.plot_loss()

# Automatically retrieve the optimal learning rate (steepest descent)
optimal_lr = lr_finder.get_optimal_lr()
print(f"Optimal Learning Rate: {optimal_lr}")
```

**Interpretation:** The `plot_loss()` method automatically identifies the point of steepest descent in the loss curve, which is typically the optimal initial learning rate.

## 2. Dynamic Schedule (`get_lr_scheduler`)

The `get_lr_scheduler` function returns a pre-configured `ReduceLROnPlateau` callback. This standardizes the strategy for reducing LR when training stagnates.

### Usage

```python
from utils.callbacks import get_lr_scheduler

# Default configuration:
# monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6
scheduler = get_lr_scheduler()

model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[scheduler, ...]
)
```

**Behavior:** If `val_loss` does not improve for 2 epochs, the learning rate is multiplied by 0.5. This helps the model settle into a minima without oscillating.
