# AGENTS.md - Generative Deep Learning Project

## Project Context

This is the codebase for "Generative Deep Learning" 2nd Edition (O'Reilly).
Updated to use UV package manager, TensorFlow 2.20, and Weights & Biases.

## Do

- Use TensorFlow 2.20 / Keras 3 API conventions
- Use `learning_rate` instead of deprecated `lr` for optimizers
- Use `uv run` to execute Python scripts
- Add W&B logging to training code (`WandbMetricsLogger`, `WandbCallback`)
- Enable GPU memory growth in notebooks to prevent OOM
- Format code with Black (88 char line length)
- Commit after each successfully tested chapter

## Don't

- Don't use deprecated `keras.legacy` or `tf.compat.v1` APIs
- Don't hardcode file paths - use relative paths from project root
- Don't commit `.env` files or API keys
- Don't commit model weights to git (use W&B artifacts or separate storage)
- Don't create intermediate/temporary Python files for notebook debugging

## Environment

```
Python: 3.13.x (via UV)
TensorFlow: 2.20.0 with CUDA
Package Manager: UV (not pip directly)
GPU: NVIDIA RTX 2070 (8GB VRAM)
```

## Key Patterns

### Import Convention
```python
import tensorflow as tf
from tensorflow import keras
import wandb
```

### GPU Memory Setup (add to first cell of notebooks)
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### W&B Tracking
```python
wandb.init(project="generative-deep-learning", name="experiment-name")
model.fit(..., callbacks=[WandbMetricsLogger()])
wandb.finish()
```

## File Structure

```
├── documentation/       # Setup guides
├── notebooks/           # Jupyter notebooks by chapter
│   ├── 02_deeplearning/
│   ├── 03_vae/
│   └── ...
├── pyproject.toml       # UV project config
└── .env                 # Local environment (gitignored)
```
