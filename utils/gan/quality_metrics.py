"""
GAN Quality Metrics Module.

This module provides functions for computing image quality metrics
for GAN-generated images, including FID and Inception Score.

Module Location:
    utils/gan/quality_metrics.py

Key Metrics:
    - FID (Fréchet Inception Distance): Measures similarity to real images
    - Inception Score (IS): Measures quality and diversity
    - Pixel Variance: Simple diversity metric
    - Image Statistics: Basic statistical analysis of generated images

Note:
    FID and IS require significant computation and use pre-trained
    Inception v3 models. Run these every N epochs (e.g., 100) rather
    than every epoch.

Usage:
    from utils.gan.quality_metrics import (
        compute_fid_score,
        compute_inception_score,
        compute_pixel_variance,
    )

    # Every 100 epochs
    if epoch % 100 == 0:
        fid = compute_fid_score(real_images, fake_images)
        is_mean, is_std = compute_inception_score(fake_images)

Dependencies:
    - tensorflow
    - scipy
    - numpy

Author:
    Generated with comprehensive PEP-8 documentation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import tensorflow as tf
    from scipy import linalg
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =============================================================================
# INCEPTION MODEL (LAZY LOADING)
# =============================================================================

# Global variable to cache the Inception model
_INCEPTION_MODEL = None


def _get_inception_model():
    """
    Lazily load and cache the Inception v3 model.

    The model is loaded only once and cached for subsequent calls.
    Uses the pool_3 layer output for feature extraction.

    Returns:
        Keras Model for feature extraction.

    Raises:
        ImportError: If TensorFlow is not available.
    """
    global _INCEPTION_MODEL

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for quality metrics.")

    if _INCEPTION_MODEL is None:
        print("Loading Inception v3 model for quality metrics...")
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=(299, 299, 3)
        )
        _INCEPTION_MODEL = base_model
        print("✓ Inception v3 model loaded")

    return _INCEPTION_MODEL


def _preprocess_images_for_inception(
    images: np.ndarray,
    target_size: Tuple[int, int] = (299, 299)
) -> np.ndarray:
    """
    Preprocess images for Inception v3 model.

    Resizes images and applies Inception-specific preprocessing.

    Args:
        images: Numpy array of images with shape (N, H, W, C).
                Values should be in range [0, 1] or [-1, 1].
        target_size: Target size for resizing (height, width).

    Returns:
        Preprocessed images ready for Inception model.
    """
    # Ensure values are in [0, 1]
    if images.min() < 0:
        images = (images + 1) / 2

    # Resize images
    resized = tf.image.resize(images, target_size)

    # Apply Inception preprocessing (scales to [-1, 1])
    preprocessed = tf.keras.applications.inception_v3.preprocess_input(
        resized * 255
    )

    return preprocessed.numpy()


# =============================================================================
# FID SCORE
# =============================================================================

def compute_fid_score(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    batch_size: int = 50
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between real and fake images.

    FID measures the similarity between the distribution of generated
    images and real images. Lower FID indicates higher quality.

    The FID is computed as:
        FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r·Σ_f))

    Where μ and Σ are the mean and covariance of Inception features.

    Args:
        real_images: Array of real images (N, H, W, C).
        fake_images: Array of generated images (N, H, W, C).
        batch_size: Batch size for feature extraction.

    Returns:
        FID score as a float. Lower is better.
        Returns -1.0 if computation fails.

    Example:
        >>> fid = compute_fid_score(real_images, generated_images)
        >>> print(f"FID Score: {fid:.2f}")

    Note:
        - Minimum recommended sample size: 10,000 images
        - Typical good scores: < 50 for 32x32 images
        - Images are resized to 299x299 for Inception
    """
    try:
        model = _get_inception_model()
    except ImportError as e:
        print(f"⚠ Cannot compute FID: {e}")
        return -1.0

    # Preprocess images
    real_preprocessed = _preprocess_images_for_inception(real_images)
    fake_preprocessed = _preprocess_images_for_inception(fake_images)

    # Extract features in batches
    real_features = []
    fake_features = []

    for i in range(0, len(real_preprocessed), batch_size):
        batch = real_preprocessed[i:i + batch_size]
        features = model.predict(batch, verbose=0)
        real_features.append(features)

    for i in range(0, len(fake_preprocessed), batch_size):
        batch = fake_preprocessed[i:i + batch_size]
        features = model.predict(batch, verbose=0)
        fake_features.append(features)

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu_real - mu_fake
    diff_squared = np.dot(diff, diff)

    # Compute sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_squared + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


# =============================================================================
# INCEPTION SCORE
# =============================================================================

def compute_inception_score(
    images: np.ndarray,
    n_splits: int = 10,
    batch_size: int = 50
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS) for generated images.

    IS measures both quality and diversity of generated images.
    Higher scores indicate better quality and diversity.

    The IS is computed as:
        IS = exp(E[KL(p(y|x) || p(y))])

    Where p(y|x) is the conditional class probability and p(y) is
    the marginal class probability.

    Args:
        images: Array of generated images (N, H, W, C).
        n_splits: Number of splits for computing mean and std.
        batch_size: Batch size for prediction.

    Returns:
        Tuple of (mean IS, std IS) computed across splits.
        Returns (0.0, 0.0) if computation fails.

    Example:
        >>> is_mean, is_std = compute_inception_score(generated_images)
        >>> print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

    Note:
        - Typical good scores: 3-5 for CIFAR-10
        - IS does not compare to real images
        - Biased towards ImageNet classes
    """
    try:
        # Use InceptionV3 with classification head for IS
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights='imagenet',
            input_shape=(299, 299, 3)
        )
    except Exception as e:
        print(f"⚠ Cannot compute IS: {e}")
        return (0.0, 0.0)

    # Preprocess images
    preprocessed = _preprocess_images_for_inception(images)

    # Get predictions in batches
    all_preds = []
    for i in range(0, len(preprocessed), batch_size):
        batch = preprocessed[i:i + batch_size]
        preds = model.predict(batch, verbose=0)
        all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)

    # Compute IS for each split
    split_scores = []
    n_images = len(preds)
    split_size = n_images // n_splits

    for i in range(n_splits):
        start = i * split_size
        end = start + split_size
        part = preds[start:end]

        # Compute KL divergence
        py = np.mean(part, axis=0)
        scores = []

        for j in range(len(part)):
            pyx = part[j]
            # Add small epsilon for numerical stability
            kl = np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10)))
            scores.append(np.exp(kl))

        split_scores.append(np.mean(scores))

    return (float(np.mean(split_scores)), float(np.std(split_scores)))


# =============================================================================
# SIMPLE QUALITY METRICS
# =============================================================================

def compute_pixel_variance(images: np.ndarray) -> float:
    """
    Compute pixel variance across generated images.

    A simple diversity metric. Low variance may indicate mode collapse
    (all images look similar).

    Args:
        images: Array of images (N, H, W, C).

    Returns:
        Mean pixel variance across all images.

    Example:
        >>> var = compute_pixel_variance(generated_images)
        >>> print(f"Pixel variance: {var:.4f}")
    """
    # Compute variance for each image, then average
    per_image_var = np.var(images, axis=(1, 2, 3))
    return float(np.mean(per_image_var))


def compute_image_stats(images: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics of generated images.

    Provides a quick overview of image characteristics.

    Args:
        images: Array of images (N, H, W, C).

    Returns:
        Dictionary containing:
            - mean: Global pixel mean
            - std: Global pixel standard deviation
            - min: Minimum pixel value
            - max: Maximum pixel value
            - variance: Mean per-image variance

    Example:
        >>> stats = compute_image_stats(generated_images)
        >>> print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    """
    return {
        'mean': float(np.mean(images)),
        'std': float(np.std(images)),
        'min': float(np.min(images)),
        'max': float(np.max(images)),
        'variance': compute_pixel_variance(images),
    }


# =============================================================================
# COMPREHENSIVE QUALITY COLLECTION
# =============================================================================

def collect_quality_metrics(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    compute_fid: bool = True,
    compute_is: bool = True,
    fid_sample_size: int = 1000
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Collect all quality metrics for generated images.

    This is the main entry point for quality evaluation.
    Call every N epochs (e.g., 100) to monitor generation quality.

    Args:
        real_images: Array of real images.
        fake_images: Array of generated images.
        compute_fid: Whether to compute FID (expensive).
        compute_is: Whether to compute IS (expensive).
        fid_sample_size: Number of images to use for FID.

    Returns:
        Dictionary of quality metrics:
            - fid_score: FID value (lower is better)
            - inception_score_mean: IS mean
            - inception_score_std: IS std
            - pixel_variance: Diversity metric
            - image_stats: Basic statistics

    Example:
        >>> if epoch % 100 == 0:
        ...     quality = collect_quality_metrics(real, fake)
        ...     print(f"FID: {quality['fid_score']:.2f}")
    """
    metrics = {}

    # Basic stats (always compute)
    metrics['pixel_variance'] = compute_pixel_variance(fake_images)
    metrics['image_stats'] = compute_image_stats(fake_images)

    # FID (expensive, optional)
    if compute_fid:
        # Sample images if too many
        if len(real_images) > fid_sample_size:
            idx = np.random.choice(len(real_images), fid_sample_size, replace=False)
            real_sample = real_images[idx]
        else:
            real_sample = real_images

        if len(fake_images) > fid_sample_size:
            idx = np.random.choice(len(fake_images), fid_sample_size, replace=False)
            fake_sample = fake_images[idx]
        else:
            fake_sample = fake_images

        metrics['fid_score'] = compute_fid_score(real_sample, fake_sample)
    else:
        metrics['fid_score'] = -1.0

    # Inception Score (expensive, optional)
    if compute_is:
        is_mean, is_std = compute_inception_score(fake_images)
        metrics['inception_score_mean'] = is_mean
        metrics['inception_score_std'] = is_std
    else:
        metrics['inception_score_mean'] = 0.0
        metrics['inception_score_std'] = 0.0

    return metrics


def format_quality_output(metrics: Dict[str, Any]) -> str:
    """
    Format quality metrics as a readable string.

    Args:
        metrics: Dictionary from collect_quality_metrics().

    Returns:
        Formatted string for console output.
    """
    output = "  Quality    │"

    if 'fid_score' in metrics and metrics['fid_score'] >= 0:
        output += f" FID: {metrics['fid_score']:.1f}"

    if 'inception_score_mean' in metrics and metrics['inception_score_mean'] > 0:
        output += f"  IS: {metrics['inception_score_mean']:.2f}±{metrics['inception_score_std']:.2f}"

    if 'pixel_variance' in metrics:
        output += f"  PixVar: {metrics['pixel_variance']:.4f}"

    return output
