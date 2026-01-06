# GAN Training Triage Guide

This guide provides a systematic approach to diagnose and resolve GAN training instability, particularly mode collapse.

---

## Mode Collapse Symptoms

### Primary Indicators

| Metric | Mode Collapse | Healthy Training |
|--------|---------------|------------------|
| D Loss | 0.72-0.75 (stuck high) | ~0.69-0.71 (near ln(2), fluctuating) |
| D Accuracy | 50% exactly (stuck) | 50-70% (decreasing over time) |
| G Accuracy | 99%+ | 80-95% (gradual increase) |

**Example: Mode Collapsed Run**
```
Epoch 0:   D Loss: 0.746, D Acc: 50.0%, G Loss: 0.595, G Acc: 99.8%
Epoch 500: D Loss: 0.725, D Acc: 50.1%, G Loss: 0.545, G Acc: 99.8%
```

**Example: Healthy Training Run**
```
Epoch 0:   D Loss: 0.697, D Acc: 65.0%, G Loss: 0.681, G Acc: 100.0%
Epoch 500: D Loss: 0.708, D Acc: 53.3%, G Loss: 0.571, G Acc: 95.1%
```

---

## Triage Methodology

When GAN training shows inconsistent behavior (sometimes succeeds, sometimes fails), use controlled experiments to isolate the root cause.

### Step 1: Implement Seed Control

Add this code **before any model building**:

```python
# =============================================================================
# SEED CONTROL FOR REPRODUCIBILITY
# =============================================================================
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42  # Change this to test different initializations

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"✓ Random seed set to {SEED}")
```

### Step 2: Run Controlled Experiments

#### Plan A: Test Reproducibility First

| Run | Seed | Batch Size | Expected Outcome |
|-----|------|------------|------------------|
| 1 | 42 | 1024 | Record result (success/collapse) |
| 2 | 42 | 1024 | Should match Run 1 exactly |
| 3 | 42 | 1024 | Should match Run 1 exactly |

If all 3 runs give the **same result** → Random initialization is controlled.

#### Plan B: Isolate Batch Size vs Initialization

| Run | Seed | Batch Size | Purpose |
|-----|------|------------|---------|
| 1 | 42 | 256 | Baseline (known stable) |
| 2 | 42 | 512 | Mid-range test |
| 3 | 42 | 1024 | Target batch size |
| 4 | 123 | 1024 | Different seed |
| 5 | 456 | 1024 | Another seed |

### Step 3: Interpret Results

| Evidence | Root Cause | Solution |
|----------|------------|----------|
| All batch sizes work with seed 42, only some work with other seeds | **Random initialization** | Use known-good seed, or improve weight initialization |
| 256 always works but 1024 fails across seeds | **Batch size** | Reduce batch size, or scale LR with batch size |
| Smaller batch sizes more robust across seeds | **Both factors** | Large batch amplifies poor init; use smaller batch OR better init |

---

## Quick Reference

### Common Root Causes

1. **Batch size too large** - Reduces gradient noise, harder to escape bad minima
2. **Poor random initialization** - Unlucky starting weights
3. **Discriminator too weak** - Architectural imbalance
4. **Learning rate imbalance** - Improper D/G learning rate ratio
5. **Dropout too high** - Weakening discriminator excessively

### Practical Recommendations

**For Immediate Triage:**
1. Add seed control code to notebook
2. Run 2-3 times with same seed to verify reproducibility
3. Try different seeds with same batch size
4. Try different batch sizes with same seed

**For Long-Term Stability:**
1. Document working seeds in notebook configuration
2. Set maximum batch size cap if larger batches are unstable
3. Log all hyperparameters to W&B for reproducibility

---

## Implementation

The GAN notebook `04_01_gan_camel_train.ipynb` has been configured for triage experiments:

- **Fixed batch size**: `BATCH_SIZE = 1024`
- **Seed control**: `SEED = 42` (configurable)
- **Dynamic allocation removed** for controlled experiments

Change `SEED` value to test different initializations while keeping batch size constant.
