# Quick Start Guide

Get up and running with the Generative Deep Learning project in minutes.

---

## Prerequisites

- Ubuntu 24.04 LTS (WSL2 supported)
- NVIDIA GPU with drivers (optional, for GPU acceleration)

---

## Installation

### 1. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Python 3.13

```bash
uv python install 3.13
```

### 3. Clone and Setup Project

```bash
cd /path/to/Generative_Deep_Learning
uv sync
```

### 4. Verify Installation

```bash
uv run python --version  # Should show 3.13+
uv run python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

## GPU Setup

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow GPU
uv run python -c "
import tensorflow as tf
print('GPUs:', tf.config.list_physical_devices('GPU'))
"
```

### Enable Memory Growth (Required for Notebooks)

Add to the **first cell** of every notebook:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[gpu.name for gpu in gpus]}")
```

---

## Running Notebooks

```bash
# Start Jupyter Lab
uv run jupyter lab
```

Navigate to `v1/notebooks/` or `v2/<chapter>/` and open a notebook.

---

## Common Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/update dependencies |
| `uv run jupyter lab` | Start Jupyter Lab |
| `uv run python <script>` | Run Python script |
| `nvidia-smi` | Check GPU status |

---

## Troubleshooting

### GPU Not Detected

1. Check driver: `nvidia-smi`
2. Reinstall TensorFlow: `uv sync --reinstall`
3. For WSL2: Ensure latest Windows NVIDIA drivers

### Out of Memory

1. Reduce batch size
2. Enable memory growth (see above)
3. Clear session: `tf.keras.backend.clear_session()`

### Dependency Conflicts

```bash
rm uv.lock && uv sync
```

---

## Next Steps

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training optimization (callbacks, batch sizing, W&B)
- **[GAN_GUIDE.md](GAN_GUIDE.md)** - GAN-specific training and metrics
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Complete standardization workflow
