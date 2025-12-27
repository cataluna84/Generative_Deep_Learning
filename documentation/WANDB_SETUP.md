# Weights & Biases Integration Guide

## Setup

### 1. Create Account

Sign up at [wandb.ai](https://wandb.ai)

### 2. Get API Key

1. Go to Settings → API Keys
2. Copy your API key
3. Add to `.env`:
   ```
   WANDB_API_KEY=your_api_key_here
   WANDB_PROJECT=generative-deep-learning
   ```

### 3. Login

```bash
uv run wandb login
```

## Usage in Notebooks

### Basic Integration

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Initialize run
wandb.init(
    project="generative-deep-learning",
    name="vae-experiment-1",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
    }
)

# Add callbacks to model.fit()
model.fit(
    x_train, y_train,
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("models/best_model.keras"),
    ]
)

# End run
wandb.finish()
```

### Log Generated Images

```python
# For VAE/GAN generated images
import wandb

# Log grid of generated images
wandb.log({
    "generated_images": [wandb.Image(img) for img in generated_batch[:16]]
})
```

### Log Model Architecture

```python
wandb.log({"model_summary": wandb.Html(model.to_html())})
```

## Viewing Results

Access your experiments at:
```
https://wandb.ai/<username>/generative-deep-learning
```

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
