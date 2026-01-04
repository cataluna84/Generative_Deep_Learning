# GPU & CUDA Setup Guide

This guide covers GPU configuration for TensorFlow 2.20+ with CUDA support.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060+ | RTX 2070+ |
| VRAM | 6GB | 8GB+ |
| Driver | 525.x+ | Latest |
| CUDA | 12.x (bundled with TensorFlow) | - |

---

## Verified Configuration

This setup has been tested on:
- **GPU**: NVIDIA GeForce RTX 2070 (8GB VRAM)
- **Driver**: 591.44
- **OS**: Ubuntu 24.04.3 LTS (WSL2)
- **TensorFlow**: 2.20.0 (with bundled CUDA 12.9)
- **Python**: 3.13+

---

## Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow GPU access
uv run python -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
print('Built with CUDA:', tf.test.is_built_with_cuda())
"
```

Expected output should show at least one GPU device.

---

## Memory Management

### Enable Memory Growth (Required)

Add to the **first cell** of every notebook:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("WARNING: No GPU detected, running on CPU")
```

### Limit GPU Memory

If running multiple notebooks or encountering OOM:

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
    )
```

### Clear Session Between Experiments

```python
tf.keras.backend.clear_session()
```

---

## Batch Size Configuration

### Dynamic Batch Size (Recommended)

Use the dynamic batch size finder to automatically determine optimal batch size:

```python
from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs

# Build your model first
model = create_my_model()

# Find optimal batch size using binary search + OOM detection
BATCH_SIZE = find_optimal_batch_size(
    model=model,
    input_shape=(28, 28, 1),
)

# Scale epochs to maintain training volume
EPOCHS = calculate_adjusted_epochs(200, 32, BATCH_SIZE)
```

> [!TIP]
> See **[DYNAMIC_BATCH_SIZE.md](DYNAMIC_BATCH_SIZE.md)** for full API documentation.

### How It Works

1. Tests progressively larger batch sizes
2. Detects OOM errors automatically
3. Uses binary search to find maximum safe size
4. Logs model parameters to console and W&B

### Example Output

```
DYNAMIC BATCH SIZE FINDER
Model Parameters: 1,234,567
Estimated Model Memory: 19.8 MB
  batch_size=   64 ✓
  batch_size=  512 ✓
  batch_size= 1024 ✗ OOM
✓ Optimal batch size: 460
```

---

## Troubleshooting

### "Could not load dynamic library"

TensorFlow 2.20+ bundles CUDA libraries. If errors persist:

```bash
# Verify CUDA build info
uv run python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
```

### Out of Memory (OOM) Errors

1. **Reduce batch size** in training
2. **Enable memory growth** (see above)
3. **Clear session** between experiments
4. **Monitor memory**: `nvidia-smi -l 1`

### GPU Not Detected

1. Check driver: `nvidia-smi`
2. Reinstall TensorFlow: `uv sync --reinstall`
3. Verify CUDA: `nvcc --version` (if installed separately)

### WSL2-Specific Issues

Ensure WSL2 has GPU passthrough enabled:

```powershell
# In Windows PowerShell (admin)
wsl --update
```

Check GPU is visible in WSL:
```bash
nvidia-smi
```

If GPU still not detected in WSL, ensure you have the latest NVIDIA Windows drivers.

---

## Related Documentation

- **[DYNAMIC_BATCH_SIZE.md](DYNAMIC_BATCH_SIZE.md)** - Dynamic batch size finder
- **[UV_SETUP.md](UV_SETUP.md)** - Package manager setup
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Experiment tracking
- **[CALLBACKS.md](CALLBACKS.md)** - Training optimization callbacks
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Notebook workflow

