# Weights & Biases Integration Guide

This guide covers W&B setup for experiment tracking, loss visualization, and model comparison.

---

## Setup

### 1. Create Account

Sign up at [wandb.ai](https://wandb.ai)

### 2. Get API Key

1. Go to **Settings** → **API Keys**
2. Copy your API key
3. Add to `.env`:
   ```env
   WANDB_API_KEY=your_api_key_here
   WANDB_PROJECT=generative-deep-learning
   ```

### 3. Login

```bash
uv run wandb login
```

Or verify existing login:
```bash
uv run wandb login --verify
```

---

## Usage in Notebooks

### Basic Integration

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Initialize run
wandb.init(
    project="generative-deep-learning",
    name="vae-experiment-1",
    config={
        "model": "VAE",
        "dataset": "CelebA",
        "learning_rate": "auto",  # Updated after LRFinder
        "batch_size": 384,
        "epochs": 200,
    }
)

# Add callbacks to model.fit()
model.fit(
    x_train, y_train,
    callbacks=[
        WandbMetricsLogger(),
        # ... other callbacks
    ]
)

# End run
wandb.finish()
```

### Using Helper Functions

The project provides helper functions in `utils/wandb_utils.py`:

```python
import sys; sys.path.insert(0, '..')
from utils.wandb_utils import init_wandb, get_metrics_logger, log_images

# Initialize with helper
run = init_wandb(
    name="vae-faces-v1",
    config={"learning_rate": "auto", "batch_size": 384}
)

# Get metrics logger callback
metrics_logger = get_metrics_logger()

# Log generated images
log_images(generated_batch[:16], key="generated_images")
```

### Log Generated Images

```python
import wandb

# Log grid of generated images
wandb.log({
    "generated_images": [wandb.Image(img) for img in generated_batch[:16]]
})
```

### Log Model Architecture

```python
# Using helper
from utils.wandb_utils import log_model_summary
log_model_summary(model)

# Or directly
summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))
wandb.log({"model_summary": wandb.Html("<pre>" + "\n".join(summary_lines) + "</pre>")})
```

### Update Config After LRFinder

```python
# After finding optimal learning rate
wandb.config.update({"learning_rate": optimal_lr})
```

---

## Viewing Results

Access your experiments at:
```
https://wandb.ai/<username>/generative-deep-learning
```

---

## Offline Mode

For air-gapped environments:

```python
import os
os.environ["WANDB_MODE"] = "offline"
```

Sync later with:
```bash
uv run wandb sync wandb/offline-run-*
```

---

## Complete Workflow Example

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger

# 1. Initialize W&B with learning_rate="auto"
wandb.init(
    project="generative-deep-learning",
    name="my-experiment",
    config={"learning_rate": "auto", "batch_size": 384, "epochs": 200}
)

# 2. Find optimal LR
lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
clone_model.fit(x, y, epochs=2, callbacks=[lr_finder])
optimal_lr = lr_finder.get_optimal_lr()

# 3. Update config
wandb.config.update({"learning_rate": optimal_lr})

# 4. Train with full callback stack
model.fit(
    x, y,
    epochs=200,
    callbacks=[
        WandbMetricsLogger(),
        get_lr_scheduler(monitor='loss', patience=5),
        get_early_stopping(monitor='loss', patience=10),
        LRLogger(),
    ]
)

# 5. Cleanup
wandb.finish()
```

---

## Related Documentation

- **[CALLBACKS.md](CALLBACKS.md)** - LRFinder, schedulers, early stopping
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Complete workflow
- **[UV_SETUP.md](UV_SETUP.md)** - Package manager setup
