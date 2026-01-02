# AGENTS.md - V2 (2nd Edition)

> Context for AI agents working with V2 notebooks from the 2nd Edition (2023) of "Generative Deep Learning".

---

## Context

This directory contains notebooks from the **2nd Edition (2023)** of "Generative Deep Learning", organized by chapter. Topics include:
- Deep Learning Basics
- Variational Autoencoders (VAE)
- GANs
- Autoregressive Models (LSTM, Transformers)
- Normalizing Flows
- Energy-Based Models
- Diffusion Models
- Transformers & Attention
- Music Generation

## Coding Standards

Ensure all notebooks and source code in `v2/` meet these requirements:

1.  **PEP 8 compliant code formatting**
2.  **Comprehensive documentation and comments**
3.  **Dynamic batch size and epoch scaling**
4.  **W&B integration for experiment tracking**
5.  **Step decay LR scheduler**
6.  **Enhanced training visualizations**
7.  **Kernel restart cell for GPU memory release**

---

## Directory Structure

```
v2/
├── 02_deeplearning/        # MLP, CNN basics
├── 03_vae/                 # Variational Autoencoders
│   ├── 01_autoencoder/
│   ├── 02_vae/
│   └── 03_vae_faces/
├── 04_gan/                 # GANs
│   ├── 01_dcgan/
│   ├── 02_wgan_gp/
│   └── 03_conditional/
├── 05_autoregressive/      # LSTM, Transformers
├── 06_normflow/            # Normalizing Flows
├── 07_ebm/                 # Energy-Based Models
├── 08_diffusion/           # Diffusion Models
├── 09_transformer/         # Attention Mechanisms
├── 11_music/               # Music Generation
├── src/                    # V2-specific models
│   ├── models/
│   └── utils/
├── utils.py                # Shared V2 utilities
└── AGENTS.md               # This file
```

---

## Standard Workflow

Follow the **[Notebook Standardization Guide](../documentation/NOTEBOOK_STANDARDIZATION.md)**.

### Quick Reference

1. **GPU Memory Growth** (first cell):
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Global Config**:
   ```python
   BATCH_SIZE = 1024
   EPOCHS = 200
   MODEL_TYPE = 'vae'
   DATASET_NAME = 'cifar10'
   ```

3. **W&B Initialization**:
   ```python
   import wandb
   wandb.init(
       project="generative-deep-learning",
       name="chapter-experiment",
       config={"learning_rate": "auto", "batch_size": BATCH_SIZE}
   )
   ```

4. **LRFinder**:
   ```python
   import sys; sys.path.insert(0, '../..')
   from utils.callbacks import LRFinder
   
   lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
   clone_model.fit(x, y, epochs=2, callbacks=[lr_finder])
   lr_finder.plot_loss()
   optimal_lr = lr_finder.get_optimal_lr()
   wandb.config.update({"learning_rate": optimal_lr})
   ```

5. **Training with Callbacks**:
   ```python
   from utils.callbacks import get_lr_scheduler, get_early_stopping, LRLogger
   from wandb.integration.keras import WandbMetricsLogger
   
   model.fit(x, y, callbacks=[
       WandbMetricsLogger(),
       get_lr_scheduler(monitor='loss', patience=5),
       get_early_stopping(monitor='loss', patience=10),
       LRLogger(),
   ])
   ```

6. **Cleanup**: `wandb.finish()`

7. **Kernel Restart**: Add a final cell to release GPU memory:
   ```python
   import IPython
   print("Restarting kernel to release GPU memory...")
   IPython.Application.instance().kernel.do_shutdown(restart=True)
   ```

---

## Import Patterns

**From chapter subdirectories (e.g., v2/03_vae/01_autoencoder/):**
```python
import sys
sys.path.insert(0, '../..')  # Go up to v2/

# Root utilities
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from utils.wandb_utils import init_wandb

# V2-specific utilities
from src.models.vae import VAE
```

---

## Do ✅

- Enable GPU memory growth in the first cell of every notebook
- Use `WandbMetricsLogger` for training callbacks
- Chain cells logically - each cell should have a single purpose
- Add markdown cells explaining key concepts before code
- Save models to `<chapter>/<experiment>/models/`
- Clear all outputs before committing
- Use `learning_rate` parameter (not `lr`)

## Don't 🚫

- Don't duplicate utility code - use `utils.py` or root utilities
- Don't use deprecated `lr` parameter
- Don't hardcode absolute paths
- Don't try to edit `.ipynb` files directly with text replacement tools
- Don't commit model weights to git

---

## Editing Notebooks

Since `.ipynb` files are JSON, use intermediate Python scripts in `scripts/` folder:

```python
import json

with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Modify cells as needed
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Edit cell['source'] list
        pass

with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
```

---

## Debugging Workflow

1. Run cells sequentially with `Shift+Enter`
2. When error occurs, read full traceback
3. Check variable shapes with `print(x.shape)`
4. For programmatic fixes, use an intermediate Python script in `scripts/` folder
5. Re-run from the fixed cell

---

## TensorFlow 2.20+ Updates

When updating old code:

| Original | Updated |
|----------|---------|
| `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| `from keras.layers.experimental import *` | `from keras.layers import *` |
| `model.predict_on_batch(x)` | `model(x, training=False)` |
| `tf.keras.backend.learning_phase()` | Remove (not needed) |
| `tf.keras.*` | `keras.*` |
| `.h5` files | `.keras` files |

---

## Batch Size Optimization

| Dataset | 8GB VRAM | 6GB VRAM |
|---------|----------|----------|
| MNIST/CIFAR | 1024 | 512 |
| CelebA (128×128) | 256-384 | 128-256 |

---

## VAE LRFinder

For VAEs with custom loss:

```python
import keras.backend as K
from keras.optimizers import Adam

def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])

clone_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

---

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

---

## Related Documentation

- **[../documentation/NOTEBOOK_STANDARDIZATION.md](../documentation/NOTEBOOK_STANDARDIZATION.md)** - Complete workflow
- **[../documentation/CALLBACKS.md](../documentation/CALLBACKS.md)** - Callback reference
- **[../documentation/WANDB_SETUP.md](../documentation/WANDB_SETUP.md)** - W&B setup
- **[../documentation/GPU_SETUP.md](../documentation/GPU_SETUP.md)** - GPU configuration
- **[../documentation/UV_SETUP.md](../documentation/UV_SETUP.md)** - Package manager
