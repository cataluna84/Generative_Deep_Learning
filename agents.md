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
├── models/                 # Generative model implementations
│   ├── AE.py              # Autoencoder base class
│   ├── VAE.py             # Variational Autoencoder
│   ├── GAN.py             # Generative Adversarial Network
│   ├── WGAN.py            # Wasserstein GAN
│   ├── WGANGP.py          # WGAN with Gradient Penalty
│   ├── cycleGAN.py        # Cycle-Consistent GAN
│   ├── MuseGAN.py         # Multi-track Music GAN
│   ├── RNNAttention.py    # RNN with Attention mechanism
│   └── layers/            # Custom Keras layers
├── utils/                  # Utility modules
│   ├── loaders.py         # Data loading functions
│   ├── callbacks.py       # Custom Keras callbacks
│   └── write.py           # Output/visualization utilities
├── scripts/               # Data download scripts
├── data/                  # Dataset storage (gitignored)
├── run/                   # Model outputs and generated samples
├── *.ipynb                # Chapter notebooks (02-09)
└── pyproject.toml         # Project dependencies
```

### Notebook Organization (by Chapter)
| Chapter | Topic | Notebooks |
|---------|-------|-----------|
| 02 | Deep Learning Basics | `02_01_*`, `02_02_*`, `02_03_*` |
| 03 | Autoencoders & VAEs | `03_01_*` to `03_06_*` |
| 04 | GANs | `04_01_*` to `04_03_*` |
| 05 | CycleGAN | `05_01_*` |
| 06 | Text & QA | `06_01_*` to `06_03_*` |
| 07 | Music Generation | `07_01_*` to `07_05_*` |
| 09 | Transformers | `09_01_*` |

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
from models.AE import Autoencoder
from utils.loaders import load_data
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

1. **TensorFlow 2.16+ / Keras 3.0+ Migration**
   - Use `keras.*` imports, NOT `tf.keras.*`
   - Save models as `.keras`, not `.h5`
   - Use `keras.ops` for backend-agnostic operations
   - Lambda layers need explicit `output_shape` parameter

2. **GPU/CUDA Requirements**
   - TensorFlow 2.16+ requires CUDA 12.x and cuDNN 9.x
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

### ✅ Always

- Always verify imports work with: `uv run python -c "import keras; print(keras.__version__)"`
- Always run notebooks from project root directory
- Always check GPU availability before long training runs
- Always backup trained models before retraining

---

## Gemini CLI Configuration

To use this file with Google Gemini CLI, add to `.gemini/settings.json`:

```json
{"context":{"fileName":["AGENTS.md","GEMINI.md"]}}
```
