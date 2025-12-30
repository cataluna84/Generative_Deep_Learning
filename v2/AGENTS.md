# AGENTS.md - Notebooks

## Notebook Context

This folder contains Jupyter notebooks organized by book chapter.
Each chapter has subdirectories with specific experiments.

## Do

- Enable GPU memory growth in the first cell of every notebook
- Use `WandbMetricsLogger` for training callbacks
- Chain cells logically - each cell should have a single purpose
- Add markdown cells explaining key concepts before code
- Save models to `notebooks/<chapter>/<experiment>/models/`
- Clear all outputs before committing

## Standardization Notes

**Batch Size Optimization**: For 8GB VRAM GPUs, use `BATCH_SIZE = 1024` for simple datasets (MNIST, CIFAR).

**LRFinder Workflow**: Before training, run LRFinder on a cloned model:
```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
clone_model.fit(x, y, epochs=2, callbacks=[lr_finder])
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()  # Default: 'recommended'
```

**VAE LRFinder**: For VAEs with custom loss:
```python
import keras.backend as K
def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])
clone_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

## Don't

- Don't create intermediate Python files for debugging
- Don't duplicate utility code - use `notebooks/utils.py` or `notebooks/wandb_utils.py`
- Don't use deprecated `lr` parameter - use `learning_rate`
- Don't hardcode absolute paths

## Debugging Workflow

1. Run cells sequentially with `Shift+Enter`
2. When error occurs, read full traceback
3. Check variable shapes with `print(x.shape)`
4. Fix directly in the cell - no intermediate files
5. Re-run from the fixed cell

## TensorFlow 2.20 Updates

When updating old code:

| Original | Updated |
|----------|---------|
| `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| `from keras.layers.experimental import *` | `from keras.layers import *` |
| `model.predict_on_batch(x)` | `model(x, training=False)` |
| `tf.keras.backend.learning_phase()` | Remove (not needed) |

## W&B Integration Pattern

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(
    project="generative-deep-learning",
    name="chapter-experiment",
    config={"epochs": 50, "batch_size": 32}
)

model.fit(x, y, callbacks=[WandbMetricsLogger()])
wandb.finish()
```

## Chapter Structure

```
notebooks/
├── 02_deeplearning/    # MLP, CNN basics
├── 03_vae/             # Variational Autoencoders
├── 04_gan/             # GANs
├── 05_autoregressive/  # LSTM, Transformers
├── 06_normflow/        # Normalizing Flows
├── 07_ebm/             # Energy-Based Models
├── 08_diffusion/       # Diffusion Models
├── 09_transformer/     # Attention
├── 11_music/           # Music generation
└── utils.py            # Shared utilities
```
