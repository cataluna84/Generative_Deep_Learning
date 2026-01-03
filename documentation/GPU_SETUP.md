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

> [!TIP]
> **Dynamic Configuration (Recommended)**:
> Use `utils/gpu_utils.py` to automatically calculate the optimal batch size for your GPU VRAM:
> ```python
> from utils.gpu_utils import get_optimal_batch_size, get_gpu_vram_gb
> BATCH_SIZE = get_optimal_batch_size('cifar10', vram_gb=get_gpu_vram_gb())
> ```

### Available Model Profiles

| Profile | Use Case | Notebooks |
|---------|----------|-----------|
| `'cifar10'` | CIFAR-10/MNIST classification (32x32 RGB) | `02_01`, `02_02`, `02_03` |
| `'gan'` | Standard GANs (28x28 grayscale) | `04_01_gan_camel` |
| `'wgan'` | WGAN/WGANGP with gradient penalty | `04_02`, `04_03` |
| `'vae'` | VAE with 128x128 RGB images | `03_05_vae_faces` |
| `'ae'` | Simple Autoencoders | `03_01_autoencoder` |

### Reference Values by Profile and VRAM

| VRAM | `cifar10` | `gan` | `wgan` | `vae` | `ae` |
|------|-----------|-------|--------|-------|------|
| 4GB  | 512       | 256   | 128    | 64    | 128  |
| 6GB  | 1024      | 512   | 256    | 128   | 256  |
| 8GB  | 2048      | 1024  | 512    | 256   | 384  |
| 12GB | 4096      | 2048  | 1024   | 384   | 512  |
| 16GB | 8192      | 4096  | 2048   | 512   | 768  |
| 24GB | 16384     | 8192  | 4096   | 768   | 1024 |

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
