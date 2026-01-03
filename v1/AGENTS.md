# AGENTS.md - V1 (1st Edition)

> Context for AI agents working with V1 notebooks from the 1st Edition (2019) of "Generative Deep Learning".

---

## Context

This directory contains notebooks from the **1st Edition (2019)** of "Generative Deep Learning", covering:
- Autoencoders (AE)
- Variational Autoencoders (VAE)
- GANs (GAN, WGAN, WGANGP)
- CycleGAN
- MuseGAN
- LSTM Text Generation

## Coding Standards

Ensure all notebooks and source code in `v1/` meet these requirements:

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
v1/
├── notebooks/              # 22 Jupyter notebooks
│   ├── 02_*                # Deep Learning basics (MLP, CNN)
│   ├── 03_*                # Autoencoders & VAEs
│   ├── 04_*                # GANs (GAN, WGAN, WGANGP)
│   ├── 05_*                # CycleGAN
│   ├── 06_*                # Text generation (LSTM, Q&A)
│   ├── 07_*                # Music generation (MuseGAN)
│   └── 09_*                # Positional encoding
├── data_download_scripts/  # Data download scripts
│   ├── download_camel_data.sh
│   ├── download_celeba_kaggle.sh
│   ├── download_cyclegan_data.sh
│   └── download_gutenburg_data.sh
├── data/                   # Downloaded datasets (gitignored)
├── run/                    # Model outputs (gitignored)
├── src/
│   ├── models/             # Model implementations
│   │   ├── AE.py           # Autoencoder
│   │   ├── VAE.py          # Variational Autoencoder
│   │   ├── GAN.py          # Vanilla GAN
│   │   ├── WGAN.py         # Wasserstein GAN
│   │   ├── WGANGP.py       # WGAN with Gradient Penalty
│   │   ├── cycleGAN.py     # Image-to-image translation
│   │   ├── MuseGAN.py      # Music generation
│   │   ├── RNNAttention.py # Attention for sequences
│   │   └── layers/         # Custom layers
│   └── utils/              # Loaders, preprocessing
│       ├── loaders.py
│       ├── callbacks.py
│       └── write.py
└── AGENTS.md               # This file
```

---

## Standard Workflow

When working on notebooks in this directory, follow the **[Notebook Standardization Guide](../documentation/NOTEBOOK_STANDARDIZATION.md)**.

### Quick Reference

1. **GPU Memory Growth**: Add to first cell:
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Global Config**: Move `BATCH_SIZE`, `EPOCHS`, etc. to top-level variables.
   - **Recommended**: Use `utils/gpu_utils.py` for dynamic batch size scaling based on VRAM.
   ```python
   from utils.gpu_utils import get_optimal_batch_size, calculate_adjusted_epochs, get_gpu_vram_gb
   GPU_VRAM_GB = get_gpu_vram_gb()
   BATCH_SIZE = get_optimal_batch_size('gan', vram_gb=GPU_VRAM_GB)
   ```

3. **W&B**: Initialize with global config and `learning_rate="auto"`:
   ```python
   import wandb
   wandb.init(project="generative-deep-learning", config={
       "learning_rate": "auto",
       "batch_size": BATCH_SIZE,
       "epochs": EPOCHS,
       "gpu_vram": GPU_VRAM_GB
   })
   ```

4. **LRFinder**: Run on cloned model before training:
   ```python
   from utils.callbacks import LRFinder
   lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
   lr_model.fit(x, y, epochs=2, callbacks=[lr_finder])
   lr_finder.plot_loss()
   optimal_lr = lr_finder.get_optimal_lr()
   wandb.config.update({"learning_rate": optimal_lr})
   ```

5. **Training Callbacks**:
   ```python
   from utils.callbacks import get_lr_scheduler, get_early_stopping, LRLogger
   from wandb.integration.keras import WandbMetricsLogger
   
   callbacks = [
       WandbMetricsLogger(),
       get_lr_scheduler(monitor='loss', patience=5),
       get_early_stopping(monitor='loss', patience=10),
       LRLogger(),
   ]
   ```

6. **Finish**: Always call `wandb.finish()` at the end.

7. **Kernel Restart**: Add a final cell to restart kernel and release GPU memory:
   ```python
   import IPython
   print("Restarting kernel to release GPU memory...")
   IPython.Application.instance().kernel.do_shutdown(restart=True)
   ```

---

## Batch Size Profiles

Use `get_optimal_batch_size(profile, vram_gb=GPU_VRAM_GB)` to auto-calculate:

| Profile | Use Case | 8GB | 6GB |
|---------|----------|-----|-----|
| `'cifar10'` | CIFAR-10/MNIST (32x32) | 2048 | 1024 |
| `'gan'` | GANs (28x28 grayscale) | 1024 | 512 |
| `'wgan'` | WGAN/WGANGP | 512 | 256 |
| `'vae'` | VAE (128x128 RGB) | 256 | 128 |
| `'ae'` | Autoencoders | 384 | 256 |

---

## VAE LRFinder

When running LRFinder on VAEs, define a custom reconstruction loss:

```python
import keras.backend as K
from keras.optimizers import Adam

def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])

lr_model = tf.keras.models.clone_model(vae.model)
lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

---

## Import Patterns

```python
# Model imports (from v1/notebooks/)
from src.models.VAE import VariationalAutoencoder
from src.models.AE import Autoencoder
from src.models.GAN import GAN
from src.utils.loaders import load_data

# Root utilities (requires path adjustment)
import sys; sys.path.insert(0, '..')
from utils.wandb_utils import init_wandb
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
```

---

## Do ✅

- Import models from `src.models.*`
- Import utils from `src.utils.*`
- Use shared root utilities (`utils.wandb_utils`, `utils.callbacks`)
- Enable GPU memory growth in the first cell
- Run notebooks from `v1/notebooks/` directory
- Download datasets before running notebooks

## Don't 🚫

- **Don't** hardcode training parameters in `model.fit()`
- **Don't** modify TF 1.x style code unless explicitly refactoring
- **Don't** commit model weights to git
- **Don't** edit `.ipynb` files directly with text replacement tools (use intermediate Python scripts in `scripts/` folder instead)
- **Don't** use deprecated `lr` parameter (use `learning_rate`)

---

## Component Specifics

### VAE Training
- **Learning Rate Scheduling**: The `VAE.train_with_generator` method has an `lr_decay` parameter (default 1).
  - If `lr_decay != 1`: An internal `step_decay_schedule` is added.
  - If `lr_decay == 1`: No internal scheduler, external callbacks (e.g., `ReduceLROnPlateau`) work correctly.

### Model Saving (Keras 3.0+)

> [!IMPORTANT]
> Use native `.keras` format for full models and `.weights.h5` for weights only.
> The legacy `.h5` format is deprecated and will emit warnings.

```python
# Save full model (use .keras, NOT .h5)
model.save("run/ae/model.keras")
discriminator.save("run/gan/discriminator.keras")
generator.save("run/gan/generator.keras")

# Save weights only (use .weights.h5)
model.save_weights("run/ae/weights.weights.h5")

# Load weights
model.load_weights("run/ae/weights.weights.h5")
```

---

## Data Download Scripts

| Script | Dataset | Required For |
|--------|---------|--------------|
| `download_camel_data.sh` | Quick Draw Camel | `04_01_gan_camel_train.ipynb` |
| `download_celeba_kaggle.sh` | CelebA Faces | `03_05_vae_faces_train.ipynb`, `04_03_wgangp_faces_train.ipynb` |
| `download_cyclegan_data.sh` | Apple2Orange | `05_01_cyclegan_train.ipynb` |
| `download_gutenburg_data.sh` | Project Gutenberg | `06_01_lstm_text_train.ipynb` |

---

## Related Documentation

- **[../documentation/NOTEBOOK_STANDARDIZATION.md](../documentation/NOTEBOOK_STANDARDIZATION.md)** - Complete workflow
- **[../documentation/CALLBACKS.md](../documentation/CALLBACKS.md)** - Callback reference
- **[../documentation/CELEBA_SETUP.md](../documentation/CELEBA_SETUP.md)** - CelebA setup
- **[../documentation/GPU_SETUP.md](../documentation/GPU_SETUP.md)** - GPU configuration
