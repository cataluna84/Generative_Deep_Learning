# AGENTS.md

> A machine-readable briefing for AI coding agents. Compatible with Cursor, Aider, Gemini CLI, OpenAI Codex, Jules, Factory Droids, and Zed.

---

## Setup Commands (EXECUTE FIRST)

```bash
# Install all dependencies using uv
uv sync

# Verify Python version (requires 3.13+)
uv run python --version

# Activate virtual environment (if not using uv run)
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
```

## Development Commands

```bash
# Start Jupyter Lab for notebooks
uv run jupyter lab

# Run standalone Python scripts
uv run python <script_name>.py

# Notebook standardization scripts (from project root)
uv run python scripts/update_notebook_cell.py
uv run python scripts/standardize_gan_notebook.py

# Download datasets (from project root, scripts in v1/data_download_scripts/)
bash v1/data_download_scripts/download_celeba_kaggle.sh
bash v1/data_download_scripts/download_camel_data.sh
bash v1/data_download_scripts/download_cyclegan_data.sh
bash v1/data_download_scripts/download_gutenburg_data.sh
```

## Coding Standards

Ensure all deliverables meet these requirements:

1.  **PEP 8 compliant code formatting**
2.  **Comprehensive documentation and comments**
3.  **Dynamic batch size and epoch scaling**
4.  **W&B integration for experiment tracking**
5.  **Step decay LR scheduler**
6.  **Enhanced training visualizations**
7.  **Kernel restart cell for GPU memory release**

---

## Project Architecture

```
Generative_Deep_Learning/
├── scripts/                # Notebook standardization scripts
│   ├── standardize_gan_notebook.py
│   └── update_notebook_cell.py
├── utils/                  # Shared root utilities
│   ├── callbacks.py        # LRFinder, LRLogger, get_lr_scheduler, get_early_stopping
│   ├── wandb_utils.py      # W&B integration helpers
│   └── gpu_utils.py        # Dynamic VRAM-based batch/epoch scaling
├── v1/                     # 1st Edition (2019) - 22 notebooks
│   ├── notebooks/          # Jupyter notebooks (.ipynb)
│   ├── data_download_scripts/  # Data download scripts
│   ├── data/               # Downloaded datasets (gitignored)
│   ├── run/                # Model outputs (gitignored)
│   └── src/
│       ├── models/         # AE, VAE, GAN, WGAN, WGANGP, CycleGAN, MuseGAN
│       │   └── layers/     # Custom layers (InstanceNorm, ReflectionPadding)
│       └── utils/          # Loaders, callbacks, visualization
├── v2/                     # 2nd Edition (2023) - Organized by chapter
│   ├── 02_deeplearning/    # MLP, CNN basics
│   ├── 03_vae/             # Variational Autoencoders
│   ├── 04_gan/             # GANs
│   ├── 05_autoregressive/  # LSTM, Transformers
│   ├── 06_normflow/        # Normalizing Flows
│   ├── 07_ebm/             # Energy-Based Models
│   ├── 08_diffusion/       # Diffusion Models
│   ├── 09_transformer/     # Attention
│   ├── 11_music/           # Music generation
│   ├── src/                # V2 models & utilities
│   └── utils.py            # Shared V2 utilities
├── docker/                 # Docker configuration (CPU/GPU)
├── documentation/          # Setup guides
│   ├── UV_SETUP.md
│   ├── GPU_SETUP.md
│   ├── WANDB_SETUP.md
│   ├── CALLBACKS.md
│   ├── CELEBA_SETUP.md
│   └── NOTEBOOK_STANDARDIZATION.md
├── .agent/workflows/       # Custom AI agent workflows
├── pyproject.toml          # Project dependencies
└── sample.env              # Environment template
```

---

## Import Patterns

**From v1/notebooks:**
```python
# Local model imports
from src.models.VAE import VariationalAutoencoder
from src.models.AE import Autoencoder
from src.utils.loaders import load_data

# Root utilities (requires sys.path)
import sys; sys.path.insert(0, '..')
from utils.wandb_utils import init_wandb
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
```

**From v2/notebooks (e.g., v2/03_vae/01_autoencoder/):**
```python
import sys; sys.path.insert(0, '../..')
from utils.wandb_utils import init_wandb
from utils.callbacks import LRFinder, get_lr_scheduler
```

---

## Code Conventions

### Naming
- **Model classes**: PascalCase with descriptive names (e.g., `VariationalAutoencoder`, `CycleGAN`)
- **Files**: snake_case matching class/notebook content
- **Variables**: snake_case throughout
- **Constants**: UPPER_SNAKE_CASE (e.g., `BATCH_SIZE`, `EPOCHS`)

### Import Order
```python
# 1. Standard library
import os
import numpy as np

# 2. Third-party (TensorFlow/Keras first)
import tensorflow as tf
from keras import layers, Model
from keras.optimizers import Adam
import keras.backend as K

# 3. Local imports
from src.models.AE import Autoencoder
from src.utils.loaders import load_data
```

### Keras 3.0+ Patterns
```python
# Use keras.Model subclassing or Functional API
class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define layers here

# Use keras.saving for model persistence
model.save("model.keras")  # NOT .h5

# Use keras.ops instead of tf.* for backend-agnostic code
import keras.ops as ops
```

---

## Domain Vocabulary

| Term | Definition |
|------|------------|
| **Latent Space** | Low-dimensional representation learned by encoder |
| **Reconstruction Loss** | Measures how well decoder recreates input (MSE, BCE) |
| **KL Divergence** | Regularization term in VAE forcing latent ~ N(0,1) |
| **Adversarial Loss** | Generator tries to fool discriminator |
| **Wasserstein Distance** | Earth Mover's Distance used in WGAN |
| **Gradient Penalty** | Regularization in WGANGP to enforce Lipschitz constraint |
| **Cycle Consistency** | A→B→A should equal A (CycleGAN) |
| **Attention Mechanism** | Learnable focus on relevant parts of sequence |

---

## Gotchas & Boundaries

### ⚠️ Critical Notes

1. **TensorFlow 2.20+ / Keras 3.0+ Migration**
   - Use `keras.*` imports, NOT `tf.keras.*`
   - Save models as `.keras`, not `.h5`
   - Use `keras.ops` for backend-agnostic operations
   - Lambda layers need explicit `output_shape` parameter
   - Use `learning_rate` not deprecated `lr` for optimizers

2. **GPU/CUDA Requirements**
   - TensorFlow 2.20+ requires CUDA 12.x and cuDNN 9.x
   - For CPU-only, TensorFlow will work but training is slow
   - Check GPU: `uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

3. **Data Downloads Required**
   - Most notebooks require downloading datasets first
   - Run scripts in `v1/scripts/` folder before running notebooks
   - CelebA, Camel, and text corpora are NOT included in repo

4. **Memory Management**
   - GANs and VAEs can consume significant GPU memory
   - Reduce batch size if encountering OOM errors
   - Use `tf.keras.backend.clear_session()` between experiments
   - **Batch Size Optimization**: For 8GB VRAM:
     - Simple datasets (MNIST/CIFAR): `BATCH_SIZE = 1024`
     - Face datasets (CelebA): `BATCH_SIZE = 256-384`

### 🚫 Do NOT

- Do NOT use `tf.keras.*` imports (use `keras.*` directly)
- Do NOT save models as `.h5` files (use `.keras` format)
- Do NOT modify files in `data/` or `run/` directories without backup
- Do NOT run notebooks without first downloading required datasets
- Do NOT use `tensorflow.python.*` internal APIs
- Do NOT try to edit `.ipynb` files directly with text replacement tools

### ✅ Always

- Always use intermediate Python scripts in `scripts/` folder to edit `.ipynb` files (JSON format)
- Always verify imports work with: `uv run python -c "import keras; print(keras.__version__)"`
- Always run v1 notebooks from `v1/notebooks/` directory
- Always run v2 notebooks from their chapter subdirectory
- Always check GPU availability before long training runs
- Always backup trained models before retraining
- Always enable GPU memory growth in the first cell of notebooks
- Always restart the kernel at the end of notebooks to fully release GPU memory (the only guaranteed way)

---

## W&B Integration Pattern

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(
    project="generative-deep-learning",
    name="experiment-name",
    config={
        "learning_rate": "auto",  # Updated after LRFinder
        "batch_size": 384,
        "epochs": 200
    }
)

model.fit(x, y, callbacks=[WandbMetricsLogger()])
wandb.finish()
```

---

## Notebook Standardization

For a detailed guide on standardizing notebooks (Config, W&B, LRFinder), see:
[Notebook Standardization Guide](documentation/NOTEBOOK_STANDARDIZATION.md)

---

## Callbacks & Learning Rate Tools

We provide utilities for training optimization in `utils/callbacks.py`.

### LRFinder - Find Optimal Learning Rate

```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(..., callbacks=[lr_finder], epochs=2)
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()  # Uses 'recommended' by default
```

**VAE Note**: For VAEs with custom loss, define reconstruction loss before cloning:
```python
import keras.backend as K
def vae_r_loss(y_true, y_pred):
    return 1000 * K.mean(K.square(y_true - y_pred), axis=[1,2,3])
lr_model.compile(loss=vae_r_loss, optimizer=Adam(learning_rate=1e-6))
```

**Color-Coded Selection Methods:**

| Color | Method | Description | Usage |
|-------|--------|-------------|-------|
| 🔴 | `'steepest'` | Steepest descent LR | Aggressive, fast training |
| 🟠 | `'recommended'` ★ | Steepest / 3 | **DEFAULT - Balanced** |
| 🟣 | `'valley'` | 80% loss decline | Robust, data-driven |
| 🟢 | `'min_loss_10'` | Min loss LR / 10 | Conservative, stable |

### get_lr_scheduler - Reduce LR on Plateau

```python
from utils.callbacks import get_lr_scheduler

# Without validation data (use 'loss')
scheduler = get_lr_scheduler(monitor='loss', patience=5)

# With validation data (use 'val_loss')
scheduler = get_lr_scheduler(monitor='val_loss', patience=2)
```

### get_early_stopping - Stop Training When Loss Plateaus

```python
from utils.callbacks import get_early_stopping

early_stop = get_early_stopping(monitor='loss', patience=10)
model.fit(x, y, epochs=200, callbacks=[early_stop])
```

### LRLogger - Log Learning Rate Each Epoch

```python
from utils.callbacks import LRLogger

lr_logger = LRLogger()
model.fit(x, y, callbacks=[lr_logger])
```

### Recommended Callback Stack

```python
from utils.callbacks import LRFinder, get_lr_scheduler, get_early_stopping, LRLogger
from wandb.integration.keras import WandbMetricsLogger

callbacks = [
    WandbMetricsLogger(),
    get_lr_scheduler(monitor='loss', patience=5),
    get_early_stopping(monitor='loss', patience=10),
    LRLogger()
]
model.fit(x, y, epochs=200, callbacks=callbacks)
```

**Full documentation:** [CALLBACKS.md](documentation/CALLBACKS.md)

---

## Documentation Index

| Guide | Description |
|-------|-------------|
| [UV_SETUP.md](documentation/UV_SETUP.md) | UV package manager installation |
| [GPU_SETUP.md](documentation/GPU_SETUP.md) | TensorFlow GPU/CUDA configuration |
| [WANDB_SETUP.md](documentation/WANDB_SETUP.md) | Weights & Biases integration |
| [CALLBACKS.md](documentation/CALLBACKS.md) | LRFinder, schedulers, early stopping |
| [CELEBA_SETUP.md](documentation/CELEBA_SETUP.md) | CelebA dataset download & setup |
| [NOTEBOOK_STANDARDIZATION.md](documentation/NOTEBOOK_STANDARDIZATION.md) | Notebook development workflow |

---

## Gemini CLI Configuration

To use this file with Google Gemini CLI, add to `.gemini/settings.json`:

```json
{"context":{"fileName":["AGENTS.md","GEMINI.md"]}}
```
