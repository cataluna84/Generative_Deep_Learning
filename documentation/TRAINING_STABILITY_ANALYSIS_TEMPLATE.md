# Training Stability Analysis Template

Use this template to document training run analysis. Save as `analysis_report.md` in your run folder.

---

# Training Analysis Report: [RUN_ID]

**Generated**: [DATE]  
**Total Epochs**: [EPOCHS]  
**Final D Loss**: [D_LOSS]  
**Final G Loss**: [G_LOSS]  
**W&B Run**: [View on W&B](https://wandb.ai/USERNAME/PROJECT/runs/RUN_ID)

---

## Training Verdict

| Metric | Value |
|--------|-------|
| **Stability** | ✅ STABLE / ⚠️ UNSTABLE / ❌ COLLAPSED |
| **Quality** | Excellent / Good / Fair / Poor |
| **Score** | X/5 indicators passed |
| **Recommendation** | [Action to take] |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | [VALUE] |
| Epochs | [VALUE] |
| Lr Critic | [VALUE] |
| Lr Generator | [VALUE] |
| Optimizer | [VALUE] |
| Z Dim | [VALUE] |
| N Critic | [VALUE] |
| Clip Threshold | [VALUE] |
| Input Dim | [VALUE] |
| Critic Filters | [VALUE] |
| Generator Filters | [VALUE] |

---

## Training Progression (Phase-wise Metrics)

| Phase | Epoch Range | D Loss (Start → End) | G Loss (Start → End) | Δ D/epoch | Δ G/epoch |
|-------|-------------|----------------------|----------------------|-----------|-----------|
| Warmup | 0-100 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Early | 100-1000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Mid | 1000-4000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Late | 4000-8000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Final | 8000-TOTAL | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |

---

## Stability Indicators

| Indicator | Status | Observation |
|-----------|--------|-------------|
| Monotonicity | ✅/⚠️/❌ | D and G losses change smoothly / with oscillations |
| Balance | ✅/⚠️/❌ | D_loss_real ≈ D_loss_fake (avg diff: X.XXX) |
| Mode Collapse | ✅/⚠️/❌ | No sudden plateaus / Possible collapse detected |
| Gradient Signal | ✅/⚠️/❌ | Critic maintains / loses discrimination ability |
| Wasserstein Distance | ✅/⚠️/❌ | |G loss| grows steadily (X.XX → X.XX) |

### Interpretation

**Wasserstein Loss Understanding:**
- **D loss = E[critic(real)] - E[critic(fake)]**: Critic maximizes this
- **G loss = -E[critic(fake)]**: Generator minimizes this

**Expected WGAN Behavior:**
- D loss should be positive and gradually increasing
- |G loss| should increase as generator improves
- Real/Fake discrimination should remain balanced

---

## Quality Metrics (Final)

| Metric | Value | Assessment |
|--------|-------|------------|
| FID | [VALUE] | Excellent/Good/Fair/Poor |
| IS | [VALUE] ± [STD] | Excellent/Moderate/Limited |
| PixVar | [VALUE] | Good diversity / Mode collapse risk |

---

## Notes

[Add observations, hypotheses, or next steps here]

---

## Full Details

For complete metrics, loss curves, and generated images, see the W&B run dashboard.

[View Full Report on W&B](https://wandb.ai/USERNAME/PROJECT/runs/RUN_ID)
