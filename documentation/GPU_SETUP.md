# GPU & CUDA Setup Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060+ | RTX 2070+ |
| VRAM | 6GB | 8GB+ |
| Driver | 525.x+ | Latest |
| CUDA | 12.x (bundled with TensorFlow) | - |

## Verified Configuration

This setup has been tested on:
- **GPU**: NVIDIA GeForce RTX 2070 (8GB VRAM)
- **Driver**: 591.44
- **OS**: Ubuntu 24.04.3 LTS (WSL2)
- **TensorFlow**: 2.20.0 (with bundled CUDA 12.9)

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

## Memory Management

### Enable Memory Growth

Add to the first cell of each notebook:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Limit GPU Memory

If running multiple notebooks:

```python
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
)
```

## Troubleshooting

### "Could not load dynamic library"

TensorFlow 2.20 bundles CUDA libraries. If errors persist:

```bash
# Verify CUDA installation
uv run python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
```

### Out of Memory (OOM) Errors

1. Reduce batch size in training
2. Enable memory growth (see above)
3. Clear session between experiments:
   ```python
   tf.keras.backend.clear_session()
   ```

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
