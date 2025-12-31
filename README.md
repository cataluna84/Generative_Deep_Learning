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

## Environment Setup (.env)

This project requires a `.env` file for dataset downloads and experiment tracking. Create it by copying the template:

```bash
cp sample.env .env
```

Then edit `.env` with your credentials:

```env
JUPYTER_PORT=8888
TENSORBOARD_PORT=6006
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=generative-deep-learning
```

### Getting Kaggle Credentials

Kaggle credentials are required to download datasets like CelebA, CIFAR-10, etc.

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com)
2. Go to **Account Settings** → **API** → **Create New Token**
3. This downloads `kaggle.json` containing your credentials:
   ```json
   {"username":"your_username","key":"your_api_key"}
   ```
4. Copy these values to your `.env` file:
   - `KAGGLE_USERNAME` = `username` from kaggle.json
   - `KAGGLE_KEY` = `key` from kaggle.json

The dataset download scripts in `v1/scripts/` and `v2/scripts/` will automatically read these credentials.

### Getting W&B (Weights & Biases) Credentials

W&B is used for experiment tracking, loss visualization, and model comparison.

1. **Create a W&B account** at [wandb.ai](https://wandb.ai)
2. Go to **Settings** → **API Keys** → Copy your API key
3. Set in `.env`:
   - `WANDB_API_KEY` = your API key
   - `WANDB_PROJECT` = project name (default: `generative-deep-learning`)

Alternatively, log in via terminal:
```bash
wandb login
```

### Environment Variables Reference

| Variable | Description | Required For |
|----------|-------------|--------------|
| `JUPYTER_PORT` | Local port for Jupyter Lab (default: 8888) | Docker |
| `TENSORBOARD_PORT` | Local port for TensorBoard (default: 6006) | Docker |
| `KAGGLE_USERNAME` | Your Kaggle username | Dataset downloads |
| `KAGGLE_KEY` | Your Kaggle API key | Dataset downloads |
| `WANDB_API_KEY` | Your W&B API key | Experiment tracking |
| `WANDB_PROJECT` | W&B project name | Experiment tracking |

### Verifying Setup

```bash
# Test Kaggle credentials
source .env
kaggle datasets list

# Test W&B login
wandb login --verify
```

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
