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
5.  **LRFinder for optimal learning rate**
6.  **Step decay LR scheduler**
7.  **Enhanced training visualizations**
8.  **Kernel restart cell for GPU memory release**

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
├── src/
│   ├── models/             # Model implementations
│   │   ├── AE.py           # Autoencoder
│   │   ├── VAE.py          # Variational Autoencoder
│   │   ├── GAN.py          # Vanilla GAN
│   │   ├── WGAN.py         # Wasserstein GAN (with per-epoch W&B logging)
│   │   ├── WGANGP.py       # WGAN with Gradient Penalty (per-epoch W&B logging)
│   │   ├── cycleGAN.py     # Image-to-image translation
│   │   ├── MuseGAN.py      # Music generation
│   │   ├── RNNAttention.py # Attention for sequences
│   │   └── layers/         # Custom layers (InstanceNorm, ReflectionPadding)
│   └── utils/              # V1-specific loaders, preprocessing
├── data/                   # Downloaded datasets (gitignored)
├── run/                    # Model outputs (gitignored)
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
   - **Recommended**: Use dynamic batch finder (after model build):
   ```python
   from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs
   
   # After building model
   BATCH_SIZE = find_optimal_batch_size(model=my_model, input_shape=(28, 28, 1))
   EPOCHS = calculate_adjusted_epochs(200, 32, BATCH_SIZE)
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

### CycleGAN W&B Pattern

CycleGAN uses browsable Tables instead of Artifacts:

```python
# Log all images to browsable table
columns = ["epoch", "batch", "direction", "image"]
gallery = wandb.Table(columns=columns)
for img_file in image_files:
    gallery.add_data(epoch, batch, direction, wandb.Image(img_file))
wandb.log({"image_gallery": gallery})

# Log viz diagrams
viz_table = wandb.Table(columns=["model", "diagram"])
for viz_file in viz_files:
    viz_table.add_data(model_name, wandb.Image(viz_file))
wandb.log({"model_architecture": viz_table})

# Save report to Files tab
wandb.save(report_path)
```

7. **Kernel Restart**: Add a final cell to restart kernel and release GPU memory:
   ```python
   import IPython
   print("Restarting kernel to release GPU memory...")
   IPython.Application.instance().kernel.do_shutdown(restart=True)
   ```

---

## Dynamic Batch Size

Use binary search + OOM detection to find optimal batch size:

```python
from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs

# After building model
BATCH_SIZE = find_optimal_batch_size(model=my_model, input_shape=(28, 28, 1))
EPOCHS = calculate_adjusted_epochs(200, 32, BATCH_SIZE)
```

See **[../documentation/TRAINING_GUIDE.md](../documentation/TRAINING_GUIDE.md)** for full API.

---

## VAE LRFinder

The VAE's `sampling` function is registered with `@keras.saving.register_keras_serializable`, enabling LRFinder. Define a reconstruction loss:

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
- **[../documentation/QUICKSTART.md](../documentation/QUICKSTART.md)** - Installation and GPU setup
- **[../documentation/TRAINING_GUIDE.md](../documentation/TRAINING_GUIDE.md)** - Callbacks, batch sizing, W&B
- **[../documentation/GAN_GUIDE.md](../documentation/GAN_GUIDE.md)** - GAN metrics and stability
