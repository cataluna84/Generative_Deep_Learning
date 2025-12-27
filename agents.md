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

# Download datasets (run from project root)
bash scripts/download_cyclegan_data.sh
bash scripts/download_gutenburg_data.sh
```

---

## Project Architecture

```
Generative_Deep_Learning/
├── v1/                     # 1st Edition (2019) - 22 notebooks
│   ├── notebooks/          # Notebooks and scripts
│   ├── scripts/            # Data download scripts
│   └── src/
│       ├── models/         # AE, VAE, GAN, WGAN, WGANGP, CycleGAN, MuseGAN
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
│   ├── src/                # Models (built incrementally)
│   ├── utils.py            # Shared utilities
│   └── wandb_utils.py      # W&B integration
├── docker/                 # Docker configuration
│   ├── Dockerfile.cpu      # CPU-only image
│   ├── Dockerfile.gpu      # GPU image (nvidia-docker)
│   ├── launch-docker-cpu.sh
│   └── launch-docker-gpu.sh
├── utils/                  # Shared utilities
│   └── wandb_utils.py      # W&B integration helpers
├── data/                   # Dataset storage (gitignored)
├── run/                    # Model outputs and generated samples
├── documentation/          # Setup guides
└── pyproject.toml          # Project dependencies
```

### Import Patterns

**From v1/ notebooks:**
```python
from src.models.VAE import VariationalAutoencoder
from src.utils.loaders import load_data
import sys; sys.path.insert(0, '..')
from utils.wandb_utils import init_wandb
```

**From v2/ notebooks (e.g., v2/03_vae/01_autoencoder/):**
```python
import sys; sys.path.insert(0, '../..')
from utils.wandb_utils import init_wandb
```

---

## Code Conventions

### Naming
- **Model classes**: PascalCase with descriptive names (e.g., `VariationalAutoencoder`, `CycleGAN`)
- **Files**: snake_case matching class/notebook content
- **Variables**: snake_case throughout

### Import Order
```python
# 1. Standard library
import os
import numpy as np

# 2. Third-party (TensorFlow/Keras first)
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, backend as K

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
   - Check GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

3. **Data Downloads Required**
   - Most notebooks require downloading datasets first
   - Run scripts in `scripts/` folder before running notebooks
   - CelebA, CIFAR-10, and text corpora are NOT included in repo

4. **Memory Management**
   - GANs and VAEs can consume significant GPU memory
   - Reduce batch size if encountering OOM errors
   - Use `tf.keras.backend.clear_session()` between experiments

### 🚫 Do NOT

- Do NOT use `tf.keras.*` imports (use `keras.*` directly)
- Do NOT save models as `.h5` files (use `.keras` format)
- Do NOT modify files in `data/` or `run/` directories without backup
- Do NOT run notebooks without first downloading required datasets
- Do NOT use `tensorflow.python.*` internal APIs
- Do NOT create intermediate Python files for notebook debugging

### ✅ Always

- Always verify imports work with: `uv run python -c "import keras; print(keras.__version__)"`
- Always run v1 notebooks from `v1/` directory
- Always run v2 notebooks from their chapter subdirectory
- Always check GPU availability before long training runs
- Always backup trained models before retraining
- Always enable GPU memory growth in the first cell of notebooks

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

## Gemini CLI Configuration

To use this file with Google Gemini CLI, add to `.gemini/settings.json`:

```json
{"context":{"fileName":["agents.md","GEMINI.md"]}}
```
