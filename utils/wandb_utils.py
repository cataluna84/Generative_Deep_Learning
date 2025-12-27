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
