# Training Analysis Report: 0002_horses

**Generated**: 2026-01-12 02:23:50  
**Total Epochs**: 10000  
**Final D Loss**: -0.0768  
**Final G Loss**: -1.7327  
**W&B Run**: [View on W&B](https://wandb.ai/cataluna84/generative-deep-learning/runs/1ex40gx6)

---

## Training Verdict

| Metric | Value |
|--------|-------|
| **Stability** | ⚠️ UNSTABLE |
| **Quality** | Fair |
| **Score** | 3/5 indicators passed |
| **Recommendation** | Review training parameters and consider adjustments |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 512 |
| Epochs | 10000 |
| Lr Critic | 5e-05 |
| Lr Generator | 5e-05 |
| Optimizer | rmsprop |
| Z Dim | 100 |
| N Critic | 5 |
| Clip Threshold | 0.01 |
| Input Dim | (32, 32, 3) |
| Critic Filters | [32, 64, 128, 128] |
| Generator Filters | [128, 64, 32, 3] |

---

## Training Progression (Phase-wise Metrics)

| Phase | Epoch Range | D Loss (Start → End) | G Loss (Start → End) | Δ D/epoch | Δ G/epoch |
|-------|-------------|----------------------|----------------------|-----------|-----------|
| Warmup | 0-83 | 0.00 → 0.05 | -0.00 → -0.23 | 0.0007 | -0.0027 |
| Early | 83-833 | 0.05 → 0.39 | -0.24 → -6.16 | 0.0005 | -0.0079 |
| Mid | 833-3333 | 0.39 → 0.05 | -6.15 → -2.48 | -0.0001 | 0.0015 |
| Late | 3333-6666 | 0.05 → -0.05 | -2.48 → -2.05 | -0.0000 | 0.0001 |
| Final | 6666-10000 | -0.05 → -0.08 | -2.05 → -1.73 | -0.0000 | 0.0001 |

---

## Stability Indicators

| Indicator | Status | Observation |
|-----------|--------|-------------|
| Monotonicity | ✅ Good | D and G losses change smoothly without oscillations |
| Balance | ✅ Good | D_loss_real ≈ D_loss_fake throughout training (avg diff: 0.000) |
| Mode Collapse | ❌ Issue | Potential mode collapse: plateau of 865 epochs, 113 sudden changes |
| Gradient Signal | ❌ Issue | Critic lost discrimination (final D loss: -0.077) |
| Wasserstein Distance | ✅ Good | |G loss| grows steadily (0.23 → 1.73) |

### Interpretation

**Wasserstein Loss Understanding:**
- **D loss = E[critic(real)] - E[critic(fake)]**: Critic maximizes this
- **G loss = -E[critic(fake)]**: Generator minimizes this

**Expected WGAN Behavior:**
- D loss should be positive and gradually increasing
- |G loss| should increase as generator improves
- Real/Fake discrimination should remain balanced

---

## Notes

Baseline run with default hyperparameters

---

## Full Details

For complete metrics, loss curves, and generated images, see the W&B run dashboard.

[View Full Report on W&B](https://wandb.ai/cataluna84/generative-deep-learning/runs/1ex40gx6)
