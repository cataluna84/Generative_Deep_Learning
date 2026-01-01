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

## Batch Size Recommendations

| VRAM | Simple Datasets (MNIST/CIFAR) | Face Datasets (CelebA) |
|------|------------------------------|------------------------|
| 6GB | 512 | 128-256 |
| 8GB | 1024 | 256-384 |
| 12GB+ | 2048+ | 512+ |

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

- **[UV_SETUP.md](UV_SETUP.md)** - Package manager setup
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Experiment tracking
- **[CALLBACKS.md](CALLBACKS.md)** - Training optimization callbacks
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Notebook workflow
