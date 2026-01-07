"""
Weights & Biases Utility Functions for Generative Deep Learning Notebooks

Usage:
    from wandb_utils import init_wandb, get_image_logger

    init_wandb("vae_experiment", {"epochs": 50})
    model.fit(..., callbacks=[get_image_logger()])
    wandb.finish()
"""

import os
from typing import Optional

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbCallback, WandbMetricsLogger


def init_wandb(
    name: str,
    config: Optional[dict] = None,
    project: str = "generative-deep-learning",
) -> wandb.run:
    """Initialize W&B run with standard configuration.
    
    Args:
        name: Experiment name (e.g., "vae_faces_v1")
        config: Hyperparameters dict
        project: W&B project name
        
    Returns:
        wandb.run object
    """

    if wandb.run is not None:
        print(f"Finishing existing run: {wandb.run.name}")
        wandb.finish()

    return wandb.init(
        project=project,
        name=name,
        config=config or {},
        save_code=True,
    )


def get_metrics_logger() -> WandbMetricsLogger:
    """Get standard metrics logger callback."""
    return WandbMetricsLogger()


def get_model_checkpoint(filepath: str = "model-best.keras") -> WandbCallback:
    """Get model checkpoint callback that saves to W&B.
    
    Args:
        filepath: Local path for checkpoint
        
    Returns:
        WandbCallback configured for checkpointing
    """
    return WandbCallback(
        save_model=True,
        monitor="val_loss",
        mode="min",
    )


def log_images(images: list, key: str = "generated_images") -> None:
    """Log a batch of images to W&B.
    
    Args:
        images: List of numpy arrays or PIL images
        key: W&B log key
    """
    wandb.log({key: [wandb.Image(img) for img in images]})


def log_model_summary(model: tf.keras.Model) -> None:
    """Log model architecture summary."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    wandb.log({"model_summary": wandb.Html("<pre>" + "\n".join(summary_lines) + "</pre>")})


class GeneratedImageCallback(tf.keras.callbacks.Callback):
    """Callback to log generated images during training (for VAE/GAN)."""
    
    def __init__(
        self,
        generator_fn,
        num_images: int = 16,
        log_every_n_epochs: int = 5,
    ):
        """
        Args:
            generator_fn: Function that returns generated images
            num_images: Number of images to log
            log_every_n_epochs: Frequency of logging
        """
        super().__init__()
        self.generator_fn = generator_fn
        self.num_images = num_images
        self.log_every_n_epochs = log_every_n_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every_n_epochs == 0:
            images = self.generator_fn(self.num_images)
            log_images(images, key=f"generated_epoch_{epoch+1}")


# =============================================================================
# GAN ANALYSIS LOGGING FUNCTIONS
# =============================================================================

def log_phase_metrics(phase_metrics: dict) -> None:
    """
    Log phase-wise metrics as a W&B Table.
    
    Args:
        phase_metrics: Dict from stability_analysis.compute_phase_metrics()
    """
    columns = [
        "Phase", "Epoch Range", "D Loss (Start → End)", 
        "G Loss (Start → End)", "Δ D/epoch", "Δ G/epoch"
    ]
    data = []
    
    for phase_name, metrics in phase_metrics.items():
        epoch_range = f"{metrics['epoch_range'][0]}-{metrics['epoch_range'][1]}"
        d_range = f"{metrics['d_loss_start']:.2f} → {metrics['d_loss_end']:.2f}"
        g_range = f"{metrics['g_loss_start']:.2f} → {metrics['g_loss_end']:.2f}"
        d_delta = f"{metrics['d_delta_per_epoch']:.4f}"
        g_delta = f"{metrics['g_delta_per_epoch']:.4f}"
        
        data.append([phase_name.capitalize(), epoch_range, d_range, g_range, d_delta, g_delta])
    
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"phase_metrics": table})


def log_stability_analysis(analysis: dict) -> None:
    """
    Log stability analysis results to W&B summary.
    
    Args:
        analysis: Dict from stability_analysis.analyze_training_run()
    """
    # Log indicators to summary
    for indicator_name, (passed, observation) in analysis['indicators'].items():
        wandb.summary[f"stability_{indicator_name}"] = "✅ Pass" if passed else "❌ Fail"
        wandb.summary[f"stability_{indicator_name}_note"] = observation
    
    # Log verdict to summary
    verdict = analysis['verdict']
    wandb.summary["training_stability"] = verdict['stability']
    wandb.summary["training_quality"] = verdict['quality']
    wandb.summary["recommendation"] = verdict['recommendation']
    wandb.summary["stability_score"] = f"{verdict['passed']}/{verdict['total']}"
    
    # Log final metrics
    if analysis['final_d_loss'] is not None:
        wandb.summary["final_d_loss"] = analysis['final_d_loss']
    if analysis['final_g_loss'] is not None:
        wandb.summary["final_g_loss"] = analysis['final_g_loss']
    wandb.summary["total_epochs"] = analysis['total_epochs']


def log_stability_table(analysis: dict) -> None:
    """
    Log stability indicators as a W&B Table for visualization.
    
    Args:
        analysis: Dict from stability_analysis.analyze_training_run()
    """
    columns = ["Indicator", "Status", "Observation"]
    data = []
    
    for indicator_name, (passed, observation) in analysis['indicators'].items():
        status = "✅ Good" if passed else "❌ Issue"
        # Convert snake_case to Title Case
        display_name = indicator_name.replace('_', ' ').title()
        data.append([display_name, status, observation])
    
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"stability_indicators": table})


def log_training_report(
    config: dict,
    analysis: dict,
    run_folder: str = None,
    notes: str = None
) -> None:
    """
    Log complete training analysis report to W&B.
    
    This function logs all analysis data in a structured format:
    - Phase metrics table
    - Stability indicators table
    - Verdict summary
    - Optional notes
    
    Args:
        config: Training configuration dict
        analysis: Dict from stability_analysis.analyze_training_run()
        run_folder: Path to run folder (for reference)
        notes: Optional observation notes
    """
    # Log phase metrics as table
    log_phase_metrics(analysis['phase_metrics'])
    
    # Log stability indicators as table
    log_stability_table(analysis)
    
    # Log all indicators and verdict to summary
    log_stability_analysis(analysis)
    
    # Log run folder reference if provided
    if run_folder:
        wandb.summary["run_folder"] = run_folder
    
    # Log additional notes if provided
    if notes:
        wandb.summary["notes"] = notes
    
    print("✓ Training report logged to W&B")


def log_gan_losses(
    d_losses: list,
    g_losses: list,
    log_frequency: int = 10
) -> None:
    """
    Log GAN losses to W&B history (for plotting).
    
    Args:
        d_losses: List of (d_total, d_real, d_fake) tuples
        g_losses: List of generator loss values
        log_frequency: Log every N epochs to reduce data volume
    """
    for epoch in range(0, len(d_losses), log_frequency):
        d_total, d_real, d_fake = d_losses[epoch]
        g_loss = g_losses[epoch]
        
        wandb.log({
            "epoch": epoch,
            "d_loss": d_total,
            "d_loss_real": d_real,
            "d_loss_fake": d_fake,
            "g_loss": g_loss,
            "wasserstein_distance": abs(g_loss)
        }, step=epoch)


def log_epoch_metrics(epoch: int, metrics: dict) -> None:
    """
    Log comprehensive epoch metrics to W&B.
    
    This function logs all metrics collected by utils.gan.metrics
    for a single training epoch.
    
    Args:
        epoch: Current epoch number.
        metrics: Dictionary from collect_epoch_metrics() containing:
            - d_loss, d_loss_real, d_loss_fake, g_loss
            - wasserstein_distance, dg_ratio
            - critic_weight_mean, critic_weight_std
            - generator_weight_mean, generator_weight_std
            - clip_ratio
            - epoch_time
            - loss_variance (optional)
    
    Example:
        >>> from utils.gan.metrics import collect_epoch_metrics
        >>> metrics = collect_epoch_metrics(critic, generator, d_loss, g_loss, ...)
        >>> log_epoch_metrics(epoch, metrics)
    """
    log_data = {
        "epoch": epoch,
        # Losses
        "d_loss": metrics.get('d_loss', 0),
        "d_loss_real": metrics.get('d_loss_real', 0),
        "d_loss_fake": metrics.get('d_loss_fake', 0),
        "g_loss": metrics.get('g_loss', 0),
        "wasserstein_distance": metrics.get('wasserstein_distance', 0),
        "dg_ratio": metrics.get('dg_ratio', 0),
        # Weights
        "critic_weight_mean": metrics.get('critic_weight_mean', 0),
        "critic_weight_std": metrics.get('critic_weight_std', 0),
        "generator_weight_mean": metrics.get('generator_weight_mean', 0),
        "generator_weight_std": metrics.get('generator_weight_std', 0),
        # Stability
        "clip_ratio": metrics.get('clip_ratio', 0),
        # Timing
        "epoch_time": metrics.get('epoch_time', 0),
    }
    
    # Optional metrics
    if 'loss_variance' in metrics:
        log_data['loss_variance'] = metrics['loss_variance']
    
    wandb.log(log_data, step=epoch)


def log_quality_metrics(epoch: int, quality: dict) -> None:
    """
    Log quality metrics (FID, IS) to W&B.
    
    Call this every N epochs (e.g., 100) when quality metrics are computed.
    
    Args:
        epoch: Current epoch number.
        quality: Dictionary from collect_quality_metrics() containing:
            - fid_score
            - inception_score_mean, inception_score_std
            - pixel_variance
            - image_stats (dict)
    
    Example:
        >>> if epoch % 100 == 0:
        ...     quality = collect_quality_metrics(real, fake)
        ...     log_quality_metrics(epoch, quality)
    """
    log_data = {"epoch": epoch}
    
    # FID Score
    if 'fid_score' in quality and quality['fid_score'] >= 0:
        log_data['fid_score'] = quality['fid_score']
    
    # Inception Score
    if 'inception_score_mean' in quality:
        log_data['inception_score_mean'] = quality['inception_score_mean']
        log_data['inception_score_std'] = quality.get('inception_score_std', 0)
    
    # Pixel Variance
    if 'pixel_variance' in quality:
        log_data['pixel_variance'] = quality['pixel_variance']
    
    # Image Stats
    if 'image_stats' in quality:
        stats = quality['image_stats']
        log_data['generated_image_mean'] = stats.get('mean', 0)
        log_data['generated_image_std'] = stats.get('std', 0)
    
    wandb.log(log_data, step=epoch)
    print(f"✓ Quality metrics logged at epoch {epoch}")


