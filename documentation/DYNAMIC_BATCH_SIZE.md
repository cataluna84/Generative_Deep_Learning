# Dynamic Batch Size Finder

Automatically find the optimal batch size for your model and GPU using binary search with OOM detection.

## Overview

The dynamic batch size finder replaces static lookup tables with a runtime approach that:
1. Tests progressively larger batch sizes
2. Detects OOM errors automatically
3. Uses binary search to find the maximum safe batch size
4. Logs model parameters to console and W&B

## Quick Start

```python
from utils.gpu_utils import find_optimal_batch_size, calculate_adjusted_epochs

# Build your model first
model = create_my_model()

# Find optimal batch size
BATCH_SIZE = find_optimal_batch_size(
    model=model,
    input_shape=(28, 28, 1),
)

# Scale epochs to maintain training volume
EPOCHS = calculate_adjusted_epochs(
    reference_epochs=200,
    reference_batch=32,
    actual_batch=BATCH_SIZE,
)
```

## API Reference

### `find_optimal_batch_size()`

```python
def find_optimal_batch_size(
    model: tf.keras.Model,
    input_shape: Tuple[int, ...],
    min_batch_size: int = 2,
    max_batch_size: int = 4096,
    safety_factor: float = 0.9,
    verbose: bool = True,
    log_to_wandb: bool = True,
) -> int:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Model | Required | Compiled Keras model |
| `input_shape` | Tuple | Required | Shape without batch dim (H, W, C) |
| `min_batch_size` | int | 2 | Minimum batch to test |
| `max_batch_size` | int | 4096 | Maximum batch to test |
| `safety_factor` | float | 0.9 | Return batch × this factor |
| `verbose` | bool | True | Print progress |
| `log_to_wandb` | bool | True | Log to W&B if available |

**Returns:** Optimal batch size (int)

### `calculate_adjusted_epochs()`

```python
def calculate_adjusted_epochs(
    reference_epochs: int,
    reference_batch: int,
    actual_batch: int,
) -> int:
```

Scales epochs inversely with batch size to maintain equivalent training updates:

```
adjusted_epochs = reference_epochs × (reference_batch / actual_batch)
```

**Returns:** Adjusted epoch count (minimum 50)

## Example Output

```
════════════════════════════════════════════════════════════════
DYNAMIC BATCH SIZE FINDER
════════════════════════════════════════════════════════════════
Model Parameters: 1,234,567
Estimated Model Memory: 19.8 MB (weights + optimizer + gradients)
Input Shape: (28, 28, 1)
────────────────────────────────────────────────────────────────
Testing batch sizes...
  batch_size=   64 ✓
  batch_size=  128 ✓
  batch_size=  256 ✓
  batch_size=  512 ✓
  batch_size= 1024 ✗ OOM
Binary search: 512 - 1024
  batch_size=  768 ✓
────────────────────────────────────────────────────────────────
✓ Optimal batch size: 691 (768 × 0.9 safety)
════════════════════════════════════════════════════════════════
```

## How It Works

### Algorithm

1. **Exponential Growth**: Start at `min_batch_size`, double until OOM
2. **Binary Search**: Find exact boundary between success and OOM
3. **Safety Factor**: Reduce by 10% to avoid edge-case OOM
4. **Round to 32**: For GPU memory alignment efficiency

### Memory Components Captured

| Component | How Captured |
|-----------|--------------|
| Model Weights | Loaded when model is called |
| Optimizer State | Adam stores m & v (2× params) |
| Gradients | Computed during `tape.gradient()` |
| Activations | Stored during forward pass |
| Input Batch | Test batch we create |

## Integration with W&B

When W&B is active, the finder logs:

```python
wandb.log({
    "batch_finder/model_params": 1234567,
    "batch_finder/model_memory_mb": 19.8,
    "batch_finder/optimal_batch_size": 691,
})
wandb.config.update({
    "batch_size": 691,
    "batch_size_source": "dynamic_finder",
})
```

## Best Practices

1. **Call after model build**: The finder needs the model to test memory
2. **Use before W&B init**: So batch size is logged in config
3. **Run in fresh kernel**: For accurate memory measurement
4. **Set appropriate max**: Higher max = longer search time

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Takes too long | Reduce `max_batch_size` |
| Always returns min | Model may be too large for GPU |
| Inconsistent results | Restart kernel for fresh state |

## See Also

- [GPU_SETUP.md](GPU_SETUP.md) - GPU configuration guide
- [NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md) - Notebook workflow
- [CALLBACKS.md](CALLBACKS.md) - Training callbacks
