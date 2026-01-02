# Generative Deep Learning

Experiments based on O'Reilly's "Generative Deep Learning" books (1st & 2nd Editions).

## Project Standards

All code and notebooks in this repository adhere to the following standards:

- [x] **PEP 8 compliant code formatting**
- [x] **Comprehensive documentation and comments**
- [x] **Dynamic batch size and epoch scaling**
- [x] **W&B integration for experiment tracking**
- [x] **Step decay LR scheduler**
- [x] **Enhanced training visualizations**
- [x] **Kernel restart cell for GPU memory release**

---

## Project Structure

```
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
│   │   ├── 02_*            # Deep Learning basics (MLP, CNN)
│   │   ├── 03_*            # Autoencoders & VAEs
│   │   ├── 04_*            # GANs (GAN, WGAN, WGANGP)
│   │   ├── 05_*            # CycleGAN
│   │   ├── 06_*            # Text generation (LSTM, Q&A)
│   │   ├── 07_*            # Music generation (MuseGAN)
│   │   └── 09_*                # Positional encoding
│   ├── data_download_scripts/  # Data download scripts
│   │   ├── download_camel_data.sh
│   │   ├── download_celeba_kaggle.sh
│   │   ├── download_cyclegan_data.sh
│   │   └── download_gutenburg_data.sh
│   ├── src/
│   │   ├── models/         # AE, VAE, GAN, WGAN, WGANGP, CycleGAN, MuseGAN
│   │   │   └── layers/     # Custom layers (InstanceNorm, ReflectionPadding)
│   │   └── utils/          # Loaders, preprocessing, callbacks
│   ├── data/               # Downloaded datasets (gitignored)
│   ├── run/                # Model outputs (gitignored)
│   └── AGENTS.md           # V1-specific AI agent context
│
├── v2/                     # 2nd Edition (2023) - Organized by chapter
│   ├── 02_deeplearning/    # MLP, CNN basics
│   ├── 03_vae/             # Variational Autoencoders
│   ├── 04_gan/             # GANs
│   ├── 05_autoregressive/  # LSTM, Transformers
│   ├── 06_normflow/        # Normalizing Flows
│   ├── 07_ebm/             # Energy-Based Models
│   ├── 08_diffusion/       # Diffusion Models
│   ├── 09_transformer/     # Attention Mechanisms
│   ├── 11_music/           # Music Generation
│   ├── src/                # V2 models & utilities
│   │   ├── models/
│   │   └── utils/
│   ├── utils.py            # Shared V2 utilities
│   └── AGENTS.md           # V2-specific AI agent context
│
├── docker/                 # Docker configuration
│   ├── Dockerfile.cpu      # CPU-only image
│   ├── Dockerfile.gpu      # GPU image (nvidia-docker)
│   ├── launch-docker-cpu.sh
│   ├── launch-docker-gpu.sh
│   └── README.md           # Docker usage instructions
│
├── documentation/          # Setup guides
│   ├── UV_SETUP.md         # UV package manager installation
│   ├── GPU_SETUP.md        # GPU/CUDA configuration
│   ├── WANDB_SETUP.md      # Weights & Biases integration
│   ├── CALLBACKS.md        # Keras callbacks reference
│   ├── CELEBA_SETUP.md     # CelebA dataset setup
│   └── NOTEBOOK_STANDARDIZATION.md  # Standardization workflow
│
├── .agent/                 # AI agent workflows
│   └── workflows/          # Custom workflow definitions
│
├── tests/                  # Test files
├── AGENTS.md               # Root AI agent context
├── pyproject.toml          # UV/uv dependencies
├── uv.lock                 # Locked dependencies
├── sample.env              # Environment template
└── LICENSE                 # GPL-3.0 license
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

The dataset download scripts in `v1/data_download_scripts/` will automatically read these credentials.

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

See [GPU_SETUP.md](documentation/GPU_SETUP.md) for detailed GPU configuration.

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

See [WANDB_SETUP.md](documentation/WANDB_SETUP.md).

### Learning Rate Finder

Find optimal learning rates automatically:

```python
from utils.callbacks import LRFinder

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
model.fit(x, y, epochs=2, callbacks=[lr_finder])
lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()
```

**Selection Methods:**

| Color | Method | Description |
|-------|--------|-------------|
| 🔴 | `'steepest'` | Aggressive, fast training |
| 🟠 | `'recommended'` ★ | **DEFAULT** - Steepest / 3 |
| 🟣 | `'valley'` | Robust, data-driven (80% decline) |
| 🟢 | `'min_loss_10'` | Conservative, stable |

See [CALLBACKS.md](documentation/CALLBACKS.md).

### Notebook Standardization

Standardized workflow for all notebooks:
1. Global configuration block (BATCH_SIZE, EPOCHS, etc.)
2. W&B initialization with `learning_rate="auto"`
3. LRFinder on cloned model
4. Training with `WandbMetricsLogger`, `LRLogger`, `get_lr_scheduler`, `get_early_stopping`
5. Post-training visualization with log-scale LR plot
6. `wandb.finish()` cleanup

See [NOTEBOOK_STANDARDIZATION.md](documentation/NOTEBOOK_STANDARDIZATION.md).

---

## V1 Models

| Model | File | Description |
|-------|------|-------------|
| Autoencoder | `v1/src/models/AE.py` | Standard autoencoder |
| VAE | `v1/src/models/VAE.py` | Variational Autoencoder |
| GAN | `v1/src/models/GAN.py` | Vanilla GAN |
| WGAN | `v1/src/models/WGAN.py` | Wasserstein GAN |
| WGANGP | `v1/src/models/WGANGP.py` | WGAN with Gradient Penalty |
| CycleGAN | `v1/src/models/cycleGAN.py` | Image-to-image translation |
| MuseGAN | `v1/src/models/MuseGAN.py` | Music generation |
| RNNAttention | `v1/src/models/RNNAttention.py` | Attention for sequences |

---

## V1 Data Download Scripts

| Script | Dataset | Notebook |
|--------|---------|----------|
| `download_camel_data.sh` | Quick Draw Camel | `04_01_gan_camel_train.ipynb` |
| `download_celeba_kaggle.sh` | CelebA Faces | `03_05_vae_faces_train.ipynb` |
| `download_cyclegan_data.sh` | Apple2Orange | `05_01_cyclegan_train.ipynb` |
| `download_gutenburg_data.sh` | Project Gutenberg | `06_01_lstm_text_train.ipynb` |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [UV_SETUP.md](documentation/UV_SETUP.md) | UV package manager installation |
| [GPU_SETUP.md](documentation/GPU_SETUP.md) | TensorFlow GPU/CUDA configuration |
| [WANDB_SETUP.md](documentation/WANDB_SETUP.md) | Weights & Biases integration |
| [CALLBACKS.md](documentation/CALLBACKS.md) | LRFinder, schedulers, early stopping |
| [CELEBA_SETUP.md](documentation/CELEBA_SETUP.md) | CelebA dataset download & setup |
| [NOTEBOOK_STANDARDIZATION.md](documentation/NOTEBOOK_STANDARDIZATION.md) | Notebook development workflow |

---

## Versions

| Version | Book Edition | Content |
|---------|-------------|---------|
| `v1/` | 1st Edition (2019) | 22 notebooks: AE, VAE, GAN, WGAN, WGANGP, CycleGAN, MuseGAN |
| `v2/` | 2nd Edition (2023) | 40+ notebooks: Diffusion, Transformers, NormFlows, EBMs |

---

## For AI Agents

This repository includes `AGENTS.md` files for AI coding assistants:
- [`AGENTS.md`](AGENTS.md) - Root-level project context & conventions
- [`v1/AGENTS.md`](v1/AGENTS.md) - V1-specific conventions
- [`v2/AGENTS.md`](v2/AGENTS.md) - V2-specific conventions

Custom workflows are available in `.agent/workflows/`.
