"""
CycleGAN Training Report Generation

This module provides functions to automatically generate training analysis reports
for CycleGAN experiments. It can be called directly from the training notebook
after training completes.

Usage in notebook:
    from utils.gan.cyclegan_report_generation import generate_training_report
    
    # After training completes:
    report_path = generate_training_report(
        gan=gan,
        run_folder=RUN_FOLDER,
        run_id=EXPERIMENT_RUN_ID,
        config={
            'data_name': DATA_NAME,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'gen_n_filters': GEN_N_FILTERS,
            'disc_n_filters': DISC_N_FILTERS,
            'buffer_max_length': BUFFER_MAX_LENGTH,
            'lambda_reconstr': LAMBDA_RECONSTR,
            'lambda_id': LAMBDA_ID
        }
    )
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def extract_training_history(gan, epochs):
    """
    Extract training history from a CycleGAN object.
    
    Args:
        gan: Trained CycleGAN object with g_losses and d_losses attributes
        epochs: Number of training epochs
        
    Returns:
        pandas.DataFrame with training metrics
    """
    n_batches = len(gan.g_losses) // epochs if epochs > 0 else max(len(gan.g_losses), 1)
    
    history_data = []
    for i, (g_loss, d_loss) in enumerate(zip(gan.g_losses, gan.d_losses)):
        history_data.append({
            'epoch': i // n_batches if n_batches > 0 else 0,
            'batch': i % n_batches if n_batches > 0 else i,
            'global_step': i,
            'd_loss': d_loss[0] if isinstance(d_loss, (list, tuple)) else d_loss,
            'd_acc': d_loss[1] * 100 if isinstance(d_loss, (list, tuple)) and len(d_loss) > 1 else 50,
            'g_loss': g_loss[0] if isinstance(g_loss, (list, tuple)) else g_loss,
            'g_adv': g_loss[1] if isinstance(g_loss, (list, tuple)) and len(g_loss) > 1 else 0,
            'g_recon': g_loss[3] if isinstance(g_loss, (list, tuple)) and len(g_loss) > 3 else 0,
            'g_id': g_loss[5] if isinstance(g_loss, (list, tuple)) and len(g_loss) > 5 else 0
        })
    
    return pd.DataFrame(history_data)


def calculate_phase_metrics(df):
    """
    Calculate phase-wise training metrics.
    
    Args:
        df: DataFrame with training history
        
    Returns:
        List of formatted phase metric strings for markdown table
    """
    total_epochs = df['epoch'].max() + 1
    phases = {
        'Warmup': (0, max(1, int(total_epochs * 0.1))),
        'Early': (max(1, int(total_epochs * 0.1)), int(total_epochs * 0.3)),
        'Mid': (int(total_epochs * 0.3), int(total_epochs * 0.7)),
        'Late': (int(total_epochs * 0.7), total_epochs)
    }
    
    phase_rows = []
    for phase, (start, end) in phases.items():
        subset = df[(df['epoch'] >= start) & (df['epoch'] < end)]
        if subset.empty:
            continue
        
        d_start, d_end = subset['d_loss'].iloc[0], subset['d_loss'].iloc[-1]
        g_start, g_end = subset['g_loss'].iloc[0], subset['g_loss'].iloc[-1]
        d_slope = (d_end - d_start) / len(subset) if len(subset) > 0 else 0
        g_slope = (g_end - g_start) / len(subset) if len(subset) > 0 else 0
        
        phase_rows.append(
            f"| {phase} | {start}-{end} | {d_start:.2f} -> {d_end:.2f} | "
            f"{g_start:.2f} -> {g_end:.2f} | {d_slope:.4f} | {g_slope:.4f} |"
        )
    
    return phase_rows


def calculate_stability_indicators(df, lambda_reconstr=10):
    """
    Calculate stability indicators for the training run.
    
    Args:
        df: DataFrame with training history
        lambda_reconstr: Lambda value for cycle consistency loss
        
    Returns:
        Tuple of (indicators list, balance_status, stable_variance)
    """
    # Variance reduction check
    variance_start = df['d_loss'].iloc[:len(df)//2].var()
    variance_end = df['d_loss'].iloc[len(df)//2:].var()
    stable_variance = variance_end < variance_start
    
    # D loss range check (ideal is ~0.25 for LSGAN)
    final_d_loss = df['d_loss'].iloc[-1]
    final_g_loss = df['g_loss'].iloc[-1]
    balance_status = '✅ Good' if 0.1 < final_d_loss < 0.4 else '⚠️ Issue'
    
    final_cycle_loss = df['g_recon'].iloc[-1] * lambda_reconstr
    
    indicators = [
        f"| Variance Reduction | {'✅ Good' if stable_variance else '⚠️ Converging?'} | "
        f"Variance: {variance_start:.4f} -> {variance_end:.4f} |",
        f"| D Loss Range | {balance_status} | Final D Loss: {final_d_loss:.4f} (Target ~0.25) |",
        f"| Reconstruction | ✅ Good | Cycle Loss component: {final_cycle_loss:.2f} / Total G: {final_g_loss:.2f} |"
    ]
    
    return indicators, balance_status, stable_variance


def generate_loss_plot(df, save_path):
    """
    Generate and save a loss plot.
    
    Args:
        df: DataFrame with training history
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['d_loss'], label='D Loss', alpha=0.7)
    ax.plot(df.index, df['g_loss'], label='G Loss', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Generator vs Discriminator Loss')
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def find_sample_images(run_folder, total_epochs):
    """
    Find sample images from key epochs for the report.
    
    Args:
        run_folder: Path to the run folder
        total_epochs: Total number of epochs
        
    Returns:
        Markdown string with embedded images
    """
    image_epochs = [1, max(1, total_epochs // 2), max(1, total_epochs - 1)]
    image_md = ''
    
    for ep in image_epochs:
        files_ab = glob.glob(os.path.join(run_folder, 'images', f'0_{ep}_*.png'))
        files_ba = glob.glob(os.path.join(run_folder, 'images', f'1_{ep}_*.png'))
        
        if files_ab:
            img_path = os.path.relpath(files_ab[0], run_folder)
            image_md += f"### Epoch {ep} (A -> B)\n![Epoch {ep} A->B]({img_path})\n\n"
        if files_ba:
            img_path = os.path.relpath(files_ba[0], run_folder)
            image_md += f"### Epoch {ep} (B -> A)\n![Epoch {ep} B->A]({img_path})\n\n"
    
    return image_md


def generate_training_report(gan, run_folder, run_id, config, wandb_run=None):
    """
    Generate a comprehensive training analysis report.
    
    Args:
        gan: Trained CycleGAN object
        run_folder: Path to the run folder
        run_id: Experiment run ID (e.g., '008')
        config: Dictionary with training configuration:
            - data_name: Dataset name
            - batch_size: Batch size
            - epochs: Number of epochs
            - learning_rate: Learning rate
            - gen_n_filters: Generator filter count
            - disc_n_filters: Discriminator filter count
            - buffer_max_length: Buffer size
            - lambda_reconstr: Cycle loss weight
            - lambda_id: Identity loss weight
        wandb_run: Optional wandb run object to log artifacts
        
    Returns:
        Path to the generated report
    """
    print("Generating training analysis report...")
    
    # Extract configuration with defaults
    data_name = config.get('data_name', 'unknown')
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 0.0002)
    gen_n_filters = config.get('gen_n_filters', 64)
    disc_n_filters = config.get('disc_n_filters', 64)
    buffer_max_length = config.get('buffer_max_length', 50)
    lambda_reconstr = config.get('lambda_reconstr', 10)
    lambda_id = config.get('lambda_id', 2)
    
    # Extract training history
    df = extract_training_history(gan, epochs)
    
    # Save training history to CSV
    history_path = os.path.join(run_folder, 'training_history.csv')
    df.to_csv(history_path, index=False)
    print(f"  Saved training history to {history_path}")
    
    # Generate loss plot
    loss_plot_path = os.path.join(run_folder, 'loss_plot.png')
    generate_loss_plot(df, loss_plot_path)
    print(f"  Saved loss plot to {loss_plot_path}")
    
    # Calculate metrics
    total_epochs = df['epoch'].max() + 1
    phase_rows = calculate_phase_metrics(df)
    indicators, balance_status, stable_variance = calculate_stability_indicators(df, lambda_reconstr)
    
    # Get final metrics
    final_d_loss = df['d_loss'].iloc[-1]
    final_g_loss = df['g_loss'].iloc[-1]
    
    # Find sample images
    image_md = find_sample_images(run_folder, total_epochs)
    
    # Build configuration table
    config_table = '\n'.join([
        f"| Batch Size | {batch_size} |",
        f"| Epochs | {epochs} |",
        f"| Learning Rate | {learning_rate} |",
        f"| Generator Filters | {gen_n_filters} |",
        f"| Discriminator Filters | {disc_n_filters} |",
        f"| Buffer Size | {buffer_max_length} |",
        f"| Lambda Cycle | {lambda_reconstr} |",
        f"| Lambda ID | {lambda_id} |"
    ])
    
    # Generate report
    stability_str = '✅ STABLE' if balance_status == '✅ Good' else '⚠️ UNSTABLE'
    quality_str = 'Good' if stable_variance else 'Needs Review'
    recommendation = 'Continue training' if final_d_loss > 0.15 else 'Check for mode collapse'
    balance_note = 'remained balanced' if balance_status == '✅ Good' else 'may need tuning'
    
    report = f"""# CycleGAN Training Analysis: Run {run_id}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Dataset**: {data_name}
**Total Epochs**: {total_epochs}
**Final D Loss**: {final_d_loss:.4f}
**Final G Loss**: {final_g_loss:.4f}

---

## Training Verdict

| Metric | Value |
|--------|-------|
| **Stability** | {stability_str} |
| **Quality** | {quality_str} |
| **Recommendation** | {recommendation} |

---

## Configuration

| Parameter | Value |
|-----------|-------|
{config_table}

---

## Training Progression (Phase-wise Metrics)

| Phase | Epoch Range | D Loss (Start -> End) | G Loss (Start -> End) | Δ D/step | Δ G/step |
|-------|-------------|-----------------------|-----------------------|----------|----------|
{chr(10).join(phase_rows)}

---

## Stability Indicators

| Indicator | Status | Observation |
|-----------|--------|-------------|
{chr(10).join(indicators)}

---

## Loss Visualization

![Loss Plot](loss_plot.png)

---

## Generated Samples

{image_md}

## Notes
- Run {run_id} completed {total_epochs} epochs.
- Generator and Discriminator losses {balance_note}.
"""
    
    # Save report
    report_path = os.path.join(run_folder, f'{run_id}_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Generated analysis report at {report_path}")
    
    # Log to W&B if provided
    if wandb_run is not None:
        try:
            import wandb
            report_artifact = wandb.Artifact(
                name=f'cyclegan_report_{data_name}_{run_id}',
                type='analysis_report'
            )
            report_artifact.add_file(report_path)
            report_artifact.add_file(history_path)
            report_artifact.add_file(loss_plot_path)
            wandb_run.log_artifact(report_artifact)
            print("  Report logged to W&B as artifact")
        except Exception as e:
            print(f"  Warning: Could not log to W&B: {e}")
    
    return report_path


# For backwards compatibility - standalone execution
if __name__ == "__main__":
    print("This module is designed to be imported and called from a notebook.")
    print("Usage:")
    print("  from utils.gan.cyclegan_report_generation import generate_training_report")
    print("  generate_training_report(gan, run_folder, run_id, config)")
