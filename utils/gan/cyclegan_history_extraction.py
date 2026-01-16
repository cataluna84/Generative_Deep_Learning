
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

notebook_path = "notebooks/05_01_cyclegan_train.ipynb"
run_folder = "run/paint/0001_apple2orange/008"
report_path = os.path.join(run_folder, "008_analysis_report.md")

# Ensure run folder exists
if not os.path.exists(run_folder):
    os.makedirs(run_folder)

print(f"Reading notebook: {notebook_path}")
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find the training cell
# We look for a cell that contains "gan.train" in its source
history_lines = []
found_cell = False

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "gan.train" in source_str:
            print("Found training cell.")
            found_cell = True
            # Extract outputs
            for output in cell.get("outputs", []):
                if output.get("name") == "stdout":
                    history_lines.extend(output.get("text", []))
            break

if not found_cell:
    print("Training cell not found!")
    exit(1)

print(f"Extracted {len(history_lines)} lines of output.")

# Parse lines
# Format: [Epoch 99/100] [Batch 61/62] [D loss: 0.206820, acc: 76%] [G loss: 4.82674, adv: 1.28620, recon: 0.29340, id: 0.29340] time: ...
regex = r"\[Epoch (\d+)/(\d+)\] \[Batch (\d+)/(\d+)\] \[D loss: ([\d\.]+), acc: (\d+)%\] \[G loss: ([\d\.]+), adv: ([\d\.]+), recon: ([\d\.]+), id: ([\d\.]+)\]"

data = []
for line in history_lines:
    match = re.search(regex, line)
    if match:
        epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        batch = int(match.group(3))
        d_loss = float(match.group(5))
        d_acc = float(match.group(6))
        g_loss = float(match.group(7))
        g_adv = float(match.group(8))
        g_recon = float(match.group(9))
        g_id = float(match.group(10))
        
        data.append({
            "epoch": epoch,
            "batch": batch,
            "d_loss": d_loss,
            "d_acc": d_acc,
            "g_loss": g_loss,
            "g_adv": g_adv,
            "g_recon": g_recon,
            "g_id": g_id,
            "global_step": epoch * 62 + batch # Approx
        })

print(f"Parsed {len(data)} data points.")
df = pd.DataFrame(data)

if df.empty:
    print("No data parsed. Check regex.")
    # Print sample lines
    print("Sample lines:")
    for l in history_lines[-5:]:
        print(l.strip())
    exit(1)

# Save history
history_csv = os.path.join(run_folder, "training_history.csv")
df.to_csv(history_csv, index=False)
print(f"Saved history to {history_csv}")

# Generate Plots
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["d_loss"], label="D Loss", alpha=0.7)
plt.plot(df.index, df["g_loss"], label="G Loss", alpha=0.7)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Generator vs Discriminator Loss")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(run_folder, "loss_plot.png"))
print("Saved loss_plot.png")

# Generate Stability Metrics
# Monotonicity check (simple moving average trend)
d_loss_trend = df["d_loss"].rolling(window=20).mean()
monotonic = d_loss_trend.is_monotonic_decreasing or d_loss_trend.is_monotonic_increasing
print(f"D Loss Monotonic: {monotonic}")

# Generate Report Markdown
report = f"""# CycleGAN Training Report: Run 008

## Overview
- **Date**: 2026-01-16
- **Epochs**: {df['epoch'].max()}
- **Data Points**: {len(df)}

## Training Verdict
| Metric | Value |
|--------|-------|
| Stability | {'✅ Stable' if df['d_loss'].iloc[-1] < 1.0 else '⚠️ Unstable'} |
| D Loss End | {df['d_loss'].iloc[-1]:.4f} |
| G Loss End | {df['g_loss'].iloc[-1]:.4f} |

## Loss Visualization
![Loss Plot](loss_plot.png)

## Phase-wise Metrics
(Calculated from {len(df)} steps)

"""

with open(report_path, "w") as f:
    f.write(report)
print(f"Generated report at {report_path}")

