# Generative Deep Learning

Experiments based on O'Reilly's "Generative Deep Learning" books (1st & 2nd Editions).

---

## Project Structure

```
Generative_Deep_Learning/
├── v1/                     # 1st Edition (2019) - 22 notebooks
│   ├── notebooks/          # Notebooks (.ipynb)
│   ├── scripts/            # Data download scripts
│   └── src/
│       ├── models/         # AE, VAE, GAN, WGANGP, CycleGAN, MuseGAN
│       └── utils/          # Loaders, visualization
├── v2/                     # 2nd Edition (2023) - Organized by chapter
│   ├── 02_deeplearning/    # MLP, CNN basics
│   ├── 03_vae/             # Variational Autoencoders
│   ├── 04_gan/             # GANs
│   ├── 05_autoregressive/  # LSTM, Transformers
│   ├── 06_normflow/        # Normalizing Flows
│   ├── 07_ebm/             # Energy-Based Models
│   ├── 08_diffusion/       # Diffusion Models
│   ├── 09_transformer/     # Attention Mechanisms
│   └── 11_music/           # Music Generation
├── utils/                  # Shared utilities
│   ├── wandb_utils.py      # W&B integration helpers
│   └── callbacks.py        # LRFinder, LR schedulers, Early Stopping
├── docker/                 # Docker configuration (CPU/GPU)
├── documentation/          # Setup guides
│   ├── UV_SETUP.md         # Package manager setup
│   ├── GPU_SETUP.md        # CUDA/TensorFlow GPU config
│   ├── WANDB_SETUP.md      # Weights & Biases integration
│   ├── CALLBACKS.md        # Keras callbacks reference
│   └── NOTEBOOK_STANDARDIZATION.md  # Standardization workflow
├── data/                   # Downloaded datasets (gitignored)
├── run/                    # Model outputs (gitignored)
└── pyproject.toml          # UV/uv dependencies
```

---

## Quick Start

### 1. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment

```bash
uv sync
```

### 3. Run Jupyter Lab

```bash
uv run jupyter lab
```

Navigate to `v1/notebooks/` or `v2/<chapter>/` and open a notebook.

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.13+ |
| TensorFlow | 2.20+ (with bundled CUDA 12.x) |
| GPU (Recommended) | NVIDIA GTX 1060+ (8GB VRAM recommended) |

See [documentation/GPU_SETUP.md](documentation/GPU_SETUP.md) for detailed GPU configuration.

---

## Key Features

### W&B Integration

All notebooks support Weights & Biases for experiment tracking:

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(project="generative-deep-learning", config={...})
model.fit(x, y, callbacks=[WandbMetricsLogger()])
wandb.finish()
```

See [documentation/WANDB_SETUP.md](documentation/WANDB_SETUP.md).

### Learning Rate Finder

Find optimal learning rates automatically:

```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(x, y, epochs=2, callbacks=[lr_finder])
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()
```

See [documentation/CALLBACKS.md](documentation/CALLBACKS.md).

### Notebook Standardization

Standardized workflow for all notebooks:
1. Global configuration block
2. W&B initialization with `learning_rate="auto"`
3. LRFinder on cloned model
4. Training with `WandbMetricsLogger`, `LRLogger`, `get_lr_scheduler`, `get_early_stopping`
5. `wandb.finish()` cleanup

See [documentation/NOTEBOOK_STANDARDIZATION.md](documentation/NOTEBOOK_STANDARDIZATION.md).

---

## Versions

| Version | Book Edition | Content |
|---------|-------------|---------|
| `v1/` | 1st Edition (2019) | 22 notebooks covering AE, VAE, GAN, CycleGAN, MuseGAN |
| `v2/` | 2nd Edition (2023) | 40+ notebooks including Diffusion, Transformers, NormFlows |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [UV_SETUP.md](documentation/UV_SETUP.md) | UV package manager installation |
| [GPU_SETUP.md](documentation/GPU_SETUP.md) | TensorFlow GPU/CUDA configuration |
| [WANDB_SETUP.md](documentation/WANDB_SETUP.md) | Weights & Biases integration |
| [CALLBACKS.md](documentation/CALLBACKS.md) | LRFinder, schedulers, early stopping |
| [NOTEBOOK_STANDARDIZATION.md](documentation/NOTEBOOK_STANDARDIZATION.md) | Notebook development workflow |

---

## For AI Agents

This repository includes `AGENTS.md` files for AI coding assistants:
- [`AGENTS.md`](AGENTS.md) - Root-level project context
- [`v1/AGENTS.md`](v1/AGENTS.md) - V1-specific conventions
- [`v2/AGENTS.md`](v2/AGENTS.md) - V2-specific conventions
