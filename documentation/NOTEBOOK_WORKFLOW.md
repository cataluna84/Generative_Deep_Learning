# Notebook Development Workflow

## Running Notebooks

### Start Jupyter Lab

```bash
uv run jupyter lab
```

Navigate to `notebooks/` folder and open the desired chapter.

## Debugging Workflow

### 1. Cell-by-Cell Execution

Run cells sequentially with `Shift+Enter`. When an error occurs:

1. **Read the full traceback** - identifies the exact line
2. **Check variable types** - use `type(variable)` and `variable.shape` for tensors
3. **Isolate the issue** - create a new cell to test specific operations

### 2. Common TensorFlow 2.20 Updates

The original notebooks used TensorFlow 2.10. Key changes:

| Old API | New API |
|---------|---------|
| `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| `keras.layers.experimental.*` | `keras.layers.*` |
| `model.predict_on_batch()` | `model(x, training=False)` |

### 3. GPU Memory Issues

If kernel crashes with OOM:

```python
# Add to first cell
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Direct Notebook Editing

You can edit notebooks directly without intermediate files:

1. **Minor fixes**: Edit cell and re-run
2. **Major refactors**: Use Jupyter Lab's built-in diff viewer
3. **Track changes**: Commit after each working chapter

## W&B Experiment Tracking

Add to training notebooks:

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(project="generative-deep-learning", name="notebook-name")

model.fit(
    ...,
    callbacks=[WandbMetricsLogger()]
)

wandb.finish()
```

## Kernel Management

- **Restart kernel**: After modifying utility files in `notebooks/`
- **Clear outputs**: Before committing to reduce file size
- **Save checkpoints**: Jupyter auto-saves, but manual save before long operations
