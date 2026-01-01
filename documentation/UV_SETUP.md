# UV Setup Guide

This project uses [UV](https://docs.astral.sh/uv/) for Python package management.

---

## Prerequisites

- Ubuntu 24.04 LTS (WSL2 supported)
- NVIDIA GPU with drivers installed (optional, for GPU acceleration)

---

## Quick Start

### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Python 3.13

```bash
uv python install 3.13
```

### 3. Clone and Navigate to Project

```bash
cd /path/to/Generative_Deep_Learning
```

### 4. Sync Dependencies

```bash
uv sync
```

This installs TensorFlow 2.20+ with CUDA support and all required dependencies.

### 5. Run Jupyter Lab

```bash
uv run jupyter lab
```

Navigate to `v1/notebooks/` or `v2/<chapter>/` and open a notebook.

---

## Common Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/update all dependencies |
| `uv add <package>` | Add a new dependency |
| `uv run <command>` | Run command in virtual environment |
| `uv pip list` | List installed packages |
| `uv run python <script>` | Run a Python script |
| `uv run jupyter lab` | Start Jupyter Lab |

---

## Project Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `uv.lock` | Locked dependencies (auto-generated) |
| `.venv/` | Virtual environment (auto-created) |
| `.python-version` | Python version pin (3.13) |

---

## Troubleshooting

### GPU Not Detected

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If empty, check NVIDIA drivers:
```bash
nvidia-smi
```

See [GPU_SETUP.md](GPU_SETUP.md) for detailed GPU configuration.

### Memory Issues

For large models on GPUs with limited VRAM (e.g., RTX 2070 8GB), add to notebooks:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Dependency Conflicts

If you encounter dependency issues:

```bash
# Remove lock file and resync
rm uv.lock
uv sync
```

---

## Related Documentation

- **[GPU_SETUP.md](GPU_SETUP.md)** - GPU/CUDA configuration
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Weights & Biases integration
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Notebook development workflow
