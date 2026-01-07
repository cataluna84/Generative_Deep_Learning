"""
GAN Training Metrics Module.

This module provides functions for computing training metrics during GAN training,
including gradient norms, weight statistics, and other monitoring metrics.

Module Location:
    utils/gan/metrics.py

Key Metrics:
    - Gradient Norms: Measure gradient health for critic and generator
    - Weight Statistics: Track weight distributions per layer
    - Clip Ratio: Monitor percentage of weights being clipped (WGAN)
    - Loss Statistics: Rolling variance and derived metrics

Usage:
    from utils.gan.metrics import (
        compute_gradient_norm,
        compute_weight_stats,
        compute_clip_ratio,
        compute_loss_variance,
    )

    # During training
    critic_grad = compute_gradient_norm(critic, real_images, labels)
    weight_stats = compute_weight_stats(generator)

Author:
    Generated with comprehensive PEP-8 documentation.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


# =============================================================================
# GRADIENT METRICS
# =============================================================================

def compute_gradient_norm(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Optional[callable] = None
) -> float:
    """
    Compute the L2 norm of gradients for a model.

    This metric helps monitor gradient health during training.
    High values may indicate exploding gradients, while very low
    values may indicate vanishing gradients.

    Args:
        model: Keras model to compute gradients for.
        x: Input data (batch of images).
        y: Target labels.
        loss_fn: Optional custom loss function. Uses model's compiled
            loss if not provided.

    Returns:
        Float representing the L2 norm of all gradients.

    Example:
        >>> grad_norm = compute_gradient_norm(critic, real_images, labels)
        >>> print(f"Critic gradient norm: {grad_norm:.4f}")
    """
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        if loss_fn is not None:
            loss = loss_fn(y, predictions)
        else:
            loss = model.compiled_loss(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Compute L2 norm of all gradients
    total_norm = 0.0
    for grad in gradients:
        if grad is not None:
            total_norm += tf.reduce_sum(tf.square(grad))

    return float(tf.sqrt(total_norm))


def compute_gradient_norm_simple(model: tf.keras.Model) -> Dict[str, float]:
    """
    Compute gradient-related statistics from model weights.

    A simplified alternative that doesn't require forward pass.
    Uses weight magnitudes as a proxy for gradient health.

    Args:
        model: Keras model to analyze.

    Returns:
        Dictionary with weight statistics that correlate with gradient health.
    """
    weight_norms = []

    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            weight_norms.append(np.linalg.norm(w))

    return {
        'mean_weight_norm': float(np.mean(weight_norms)) if weight_norms else 0.0,
        'max_weight_norm': float(np.max(weight_norms)) if weight_norms else 0.0,
        'min_weight_norm': float(np.min(weight_norms)) if weight_norms else 0.0,
    }


# =============================================================================
# WEIGHT STATISTICS
# =============================================================================

def compute_weight_stats(model: tf.keras.Model) -> Dict[str, float]:
    """
    Compute statistics of model weights across all layers.

    Monitoring weight statistics helps detect:
    - Vanishing weights (very small mean/std)
    - Exploding weights (very large mean/std)
    - Dead neurons (zero weights)

    Args:
        model: Keras model to analyze.

    Returns:
        Dictionary containing:
            - mean: Global mean of all weights
            - std: Global standard deviation
            - min: Minimum weight value
            - max: Maximum weight value
            - zero_ratio: Percentage of weights that are zero

    Example:
        >>> stats = compute_weight_stats(generator)
        >>> print(f"Weight mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    """
    all_weights = []

    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            all_weights.extend(w.flatten())

    if not all_weights:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'zero_ratio': 0.0,
        }

    weights_array = np.array(all_weights)

    return {
        'mean': float(np.mean(weights_array)),
        'std': float(np.std(weights_array)),
        'min': float(np.min(weights_array)),
        'max': float(np.max(weights_array)),
        'zero_ratio': float(np.sum(weights_array == 0) / len(weights_array)),
    }


def compute_layer_weight_stats(model: tf.keras.Model) -> Dict[str, Dict[str, float]]:
    """
    Compute weight statistics per layer.

    Provides detailed per-layer analysis useful for debugging
    specific layers that may be problematic.

    Args:
        model: Keras model to analyze.

    Returns:
        Dictionary mapping layer names to their weight statistics.

    Example:
        >>> layer_stats = compute_layer_weight_stats(critic)
        >>> for layer, stats in layer_stats.items():
        ...     print(f"{layer}: mean={stats['mean']:.4f}")
    """
    layer_stats = {}

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        all_weights = []
        for w in weights:
            all_weights.extend(w.flatten())

        if all_weights:
            weights_array = np.array(all_weights)
            layer_stats[layer.name] = {
                'mean': float(np.mean(weights_array)),
                'std': float(np.std(weights_array)),
                'count': len(weights_array),
            }

    return layer_stats


# =============================================================================
# CLIP RATIO (WGAN-SPECIFIC)
# =============================================================================

def compute_clip_ratio(
    model: tf.keras.Model,
    clip_threshold: float
) -> float:
    """
    Compute the percentage of weights at the clipping threshold.

    In WGAN, weights are clipped to [-c, c] after each update.
    A high clip ratio indicates the model is pushing against
    the Lipschitz constraint, which may limit capacity.

    Args:
        model: Keras model (typically the critic).
        clip_threshold: The clipping threshold value (c).

    Returns:
        Float between 0.0 and 1.0 representing the ratio of
        weights at the clipping boundary.

    Example:
        >>> clip_ratio = compute_clip_ratio(critic, clip_threshold=0.01)
        >>> print(f"Clip ratio: {clip_ratio:.2%}")
    """
    clipped_count = 0
    total_count = 0

    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            flat_weights = w.flatten()
            total_count += len(flat_weights)

            # Count weights at the boundaries
            at_upper = np.sum(np.abs(flat_weights - clip_threshold) < 1e-6)
            at_lower = np.sum(np.abs(flat_weights + clip_threshold) < 1e-6)
            clipped_count += at_upper + at_lower

    if total_count == 0:
        return 0.0

    return float(clipped_count / total_count)


# =============================================================================
# LOSS STATISTICS
# =============================================================================

def compute_loss_variance(
    losses: List[float],
    window_size: int = 100
) -> float:
    """
    Compute the rolling variance of losses.

    High variance indicates unstable training, while very low
    variance may indicate the model has converged or is stuck.

    Args:
        losses: List of loss values from training history.
        window_size: Number of recent losses to consider.

    Returns:
        Float representing the variance of recent losses.

    Example:
        >>> variance = compute_loss_variance(gan.d_losses, window_size=50)
        >>> print(f"Recent loss variance: {variance:.6f}")
    """
    if len(losses) < 2:
        return 0.0

    recent_losses = losses[-window_size:]
    return float(np.var(recent_losses))


def compute_dg_ratio(d_loss: float, g_loss: float) -> float:
    """
    Compute the discriminator-generator loss ratio.

    This ratio helps monitor the balance between D and G.
    An ideal ratio maintains healthy competition.

    Args:
        d_loss: Discriminator (critic) loss value.
        g_loss: Generator loss value.

    Returns:
        Float representing D loss divided by |G loss|.
        Returns 0.0 if G loss is zero.

    Example:
        >>> ratio = compute_dg_ratio(d_loss=0.5, g_loss=-10.0)
        >>> print(f"D/G ratio: {ratio:.4f}")
    """
    if abs(g_loss) < 1e-10:
        return 0.0

    return abs(d_loss / g_loss)


def compute_wasserstein_distance(g_loss: float) -> float:
    """
    Extract Wasserstein distance from generator loss.

    In WGAN, the generator loss represents the negative
    Wasserstein distance. This function returns the absolute value.

    Args:
        g_loss: Generator loss value (typically negative).

    Returns:
        Float representing the Wasserstein distance (positive).

    Example:
        >>> w_dist = compute_wasserstein_distance(g_loss=-50.0)
        >>> print(f"Wasserstein distance: {w_dist:.2f}")
    """
    return abs(g_loss)


# =============================================================================
# TIMING METRICS
# =============================================================================

class EpochTimer:
    """
    Timer class for measuring epoch duration.

    Provides a context manager interface for easy timing of
    training epochs.

    Attributes:
        start_time: Timestamp when timing started.
        elapsed: Duration in seconds after stopping.

    Example:
        >>> timer = EpochTimer()
        >>> timer.start()
        >>> # ... training code ...
        >>> timer.stop()
        >>> print(f"Epoch took {timer.elapsed:.2f}s")
    """

    def __init__(self):
        """Initialize the timer."""
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
        return self.elapsed

    def __enter__(self) -> 'EpochTimer':
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# COMPREHENSIVE METRICS COLLECTION
# =============================================================================

def collect_epoch_metrics(
    critic: tf.keras.Model,
    generator: tf.keras.Model,
    d_loss: Tuple[float, float, float],
    g_loss: float,
    clip_threshold: float,
    epoch_time: float,
    loss_history: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Collect all training metrics for a single epoch.

    This is the main entry point for collecting metrics during training.
    It computes all easy and medium-effort metrics in a single call.

    Args:
        critic: The critic (discriminator) model.
        generator: The generator model.
        d_loss: Tuple of (total, real, fake) discriminator losses.
        g_loss: Generator loss value.
        clip_threshold: WGAN weight clipping threshold.
        epoch_time: Time taken for this epoch (seconds).
        loss_history: Optional list of previous D losses for variance.

    Returns:
        Dictionary containing all computed metrics:
            - d_loss, d_loss_real, d_loss_fake, g_loss
            - wasserstein_distance, dg_ratio
            - critic_weight_mean, critic_weight_std
            - generator_weight_mean, generator_weight_std
            - clip_ratio
            - epoch_time
            - loss_variance (if history provided)

    Example:
        >>> metrics = collect_epoch_metrics(
        ...     critic=gan.critic,
        ...     generator=gan.generator,
        ...     d_loss=(0.05, 0.04, 0.06),
        ...     g_loss=-10.0,
        ...     clip_threshold=0.01,
        ...     epoch_time=1.5
        ... )
    """
    # Basic losses
    metrics = {
        'd_loss': d_loss[0],
        'd_loss_real': d_loss[1],
        'd_loss_fake': d_loss[2],
        'g_loss': g_loss,
        'wasserstein_distance': compute_wasserstein_distance(g_loss),
        'dg_ratio': compute_dg_ratio(d_loss[0], g_loss),
    }

    # Weight statistics
    critic_stats = compute_weight_stats(critic)
    metrics['critic_weight_mean'] = critic_stats['mean']
    metrics['critic_weight_std'] = critic_stats['std']

    generator_stats = compute_weight_stats(generator)
    metrics['generator_weight_mean'] = generator_stats['mean']
    metrics['generator_weight_std'] = generator_stats['std']

    # Clip ratio
    metrics['clip_ratio'] = compute_clip_ratio(critic, clip_threshold)

    # Timing
    metrics['epoch_time'] = epoch_time

    # Loss variance
    if loss_history is not None and len(loss_history) > 1:
        d_losses = [d[0] for d in loss_history]
        metrics['loss_variance'] = compute_loss_variance(d_losses)

    return metrics


def format_verbose_output(
    epoch: int,
    total_epochs: int,
    metrics: Dict[str, Any]
) -> str:
    """
    Format metrics as verbose console output.

    Provides a detailed, readable display of all training metrics
    for the current epoch.

    Args:
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
        metrics: Dictionary of computed metrics.

    Returns:
        Formatted string for console output.

    Example:
        >>> output = format_verbose_output(100, 6000, metrics)
        >>> print(output)
    """
    separator = "═" * 75
    line = "─" * 75

    output = f"""
{separator}
Epoch {epoch}/{total_epochs} [{metrics.get('epoch_time', 0):.2f}s]
{line}
  Losses     │ D: {metrics['d_loss']:.4f} (R:{metrics['d_loss_real']:.4f} F:{metrics['d_loss_fake']:.4f})  G: {metrics['g_loss']:.3f}  W-dist: {metrics['wasserstein_distance']:.2f}
  Weights    │ Critic μ:{metrics['critic_weight_mean']:.4f} σ:{metrics['critic_weight_std']:.4f}  Gen μ:{metrics['generator_weight_mean']:.4f} σ:{metrics['generator_weight_std']:.4f}
  Stability  │ D/G Ratio: {metrics['dg_ratio']:.4f}  Clip%: {metrics['clip_ratio']:.1%}"""

    if 'loss_variance' in metrics:
        output += f"  Var: {metrics['loss_variance']:.6f}"

    output += f"\n{separator}"

    return output
