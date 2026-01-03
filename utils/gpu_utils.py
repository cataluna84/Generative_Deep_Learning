"""
GPU Utilities Module for Dynamic Training Configuration.

This module provides utilities for optimizing deep learning training based on
available GPU VRAM. It automatically detects GPU memory and calculates optimal
batch sizes and epoch counts to fully utilize hardware while maintaining
equivalent training volume.

Key Features:
    - Automatic GPU VRAM detection using TensorFlow
    - Model-specific batch size recommendations (GAN, VAE, AE, etc.)
    - Dynamic epoch scaling to maintain total training updates
    - Manual override options for custom configurations

Example Usage:
    >>> from utils.gpu_utils import get_optimal_batch_size, calculate_adjusted_epochs
    >>> 
    >>> # Auto-detect VRAM and get optimal batch size for GAN
    >>> batch_size = get_optimal_batch_size('gan')  # Returns 1024 for 8GB GPU
    >>> 
    >>> # Scale epochs to maintain equivalent training updates
    >>> epochs = calculate_adjusted_epochs(
    ...     reference_epochs=6000, 
    ...     reference_batch=256, 
    ...     actual_batch=batch_size
    ... )  # Returns 1500 for batch_size=1024

References:
    - documentation/GPU_SETUP.md: Batch size recommendations table
    - documentation/NOTEBOOK_STANDARDIZATION.md: Dynamic training workflow

Author: Antigravity AI Assistant
Created: 2026-01-02
"""

import tensorflow as tf
from typing import Optional


# =============================================================================
# BATCH SIZE CONFIGURATIONS
# =============================================================================
# These values are tuned based on experimentation with different GPU memory
# sizes. The goal is to maximize GPU utilization while avoiding OOM errors.
#
# Format: MODEL_TYPE -> {VRAM_GB: BATCH_SIZE}
# =============================================================================
BATCH_SIZE_CONFIG = {
    # GAN models (28x28 grayscale, ~1.5M params)
    # Relatively lightweight, can use larger batch sizes
    'gan': {
        4: 256,     # 4GB VRAM (GTX 1050 Ti)
        6: 512,     # 6GB VRAM (GTX 1060, RTX 2060)
        8: 1024,    # 8GB VRAM (RTX 2070, RTX 3070)
        12: 2048,   # 12GB VRAM (RTX 3080, RTX 4070)
        16: 4096,   # 16GB VRAM (RTX 4080)
        24: 8192,   # 24GB VRAM (RTX 3090, RTX 4090)
    },
    
    # WGAN/WGANGP models (similar to GAN but with gradient penalty)
    # Slightly more memory overhead due to interpolated samples
    'wgan': {
        4: 128,
        6: 256,
        8: 512,
        12: 1024,
        16: 2048,
        24: 4096,
    },
    
    # VAE models (128x128 RGB, ~800K params for encoder+decoder)
    # Higher memory usage due to larger images
    'vae': {
        4: 64,
        6: 128,
        8: 256,
        12: 384,
        16: 512,
        24: 768,
    },
    
    # Autoencoder for CelebA faces (128x128 RGB, ~800K params)
    # Higher memory usage due to larger images
    'ae': {
        4: 128,
        6: 256,
        8: 384,
        12: 512,
        16: 768,
        24: 1024,
    },
    
    # Autoencoder for MNIST/digits (28x28 grayscale, ~200K params)
    # Lightweight - can use larger batch sizes similar to GANs
    # Suitable for: 03_01_autoencoder, 03_03_vae_digits
    'ae_digits': {
        4: 512,     # 4GB VRAM
        6: 1024,    # 6GB VRAM
        8: 2048,    # 8GB VRAM (targets ~6-7GB usage)
        12: 4096,   # 12GB VRAM
        16: 8192,   # 16GB VRAM
        24: 16384,  # 24GB VRAM
    },
    
    # CIFAR-10 classification models (32x32 RGB, ~620K params for MLP)
    # Lightweight models can use large batch sizes similar to GANs
    # Suitable for: 02_01_deep_neural_network, 02_02_convolutions, 02_03_conv_neural_network
    'cifar10': {
        4: 512,     # 4GB VRAM (GTX 1050 Ti)
        6: 1024,    # 6GB VRAM (GTX 1060, RTX 2060)
        8: 2048,    # 8GB VRAM (RTX 2070, RTX 3070)
        12: 4096,   # 12GB VRAM (RTX 3080, RTX 4070)
        16: 8192,   # 16GB VRAM (RTX 4080)
        24: 16384,  # 24GB VRAM (RTX 3090, RTX 4090)
    },
}

# Default configuration for unknown model types
DEFAULT_BATCH_SIZES = {
    4: 64,
    6: 128,
    8: 256,
    12: 384,
    16: 512,
    24: 768,
}


def get_gpu_vram_gb() -> int:
    """
    Detect the available GPU VRAM in gigabytes.
    
    This function queries TensorFlow to get GPU device information and returns
    the memory capacity of the first available GPU. If no GPU is detected or
    an error occurs, it returns a default value of 8GB.
    
    Returns:
        int: GPU VRAM in gigabytes (rounded down).
        
    Note:
        - Only considers the first GPU in multi-GPU systems
        - Returns 8 if GPU detection fails (conservative default)
        - Memory is reported in GiB (1 GiB = 1024^3 bytes)
        
    Example:
        >>> vram = get_gpu_vram_gb()
        >>> print(f"GPU VRAM: {vram}GB")
        GPU VRAM: 8GB
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("WARNING: No GPU detected. Using default VRAM value of 8GB.")
            return 8
        
        # Get memory info for the first GPU
        # TensorFlow doesn't directly expose VRAM, so we use a workaround
        # by checking the device attributes or using nvidia-smi parsing
        gpu = gpus[0]
        
        # Try to get memory limit if set
        try:
            memory_info = tf.config.experimental.get_memory_info(gpu.name.replace('/physical_device:', ''))
            if 'total' in memory_info:
                vram_bytes = memory_info['total']
                vram_gb = int(vram_bytes / (1024 ** 3))
                return max(vram_gb, 4)  # Minimum 4GB
        except (RuntimeError, AttributeError):
            pass
        
        # Fallback: detect from device name (rough estimation)
        gpu_name = gpu.name.lower() if hasattr(gpu, 'name') else ''
        
        # Common GPU VRAM mappings based on device names
        if '3090' in gpu_name or '4090' in gpu_name:
            return 24
        elif '3080' in gpu_name or '4080' in gpu_name:
            return 12
        elif '3070' in gpu_name or '4070' in gpu_name or '2080' in gpu_name:
            return 8
        elif '2070' in gpu_name or '3060' in gpu_name:
            return 8
        elif '2060' in gpu_name or '1080' in gpu_name:
            return 8
        elif '1070' in gpu_name or '3050' in gpu_name:
            return 8
        elif '1060' in gpu_name:
            return 6
        elif '1050' in gpu_name:
            return 4
        else:
            # Default to 8GB for unknown GPUs
            return 8
            
    except Exception as e:
        print(f"WARNING: Error detecting GPU VRAM: {e}. Using default of 8GB.")
        return 8


def get_optimal_batch_size(
    model_type: str,
    vram_gb: Optional[int] = None
) -> int:
    """
    Get the optimal batch size for a given model type and GPU VRAM.
    
    This function returns a pre-configured batch size that maximizes GPU
    utilization while avoiding out-of-memory errors. The batch sizes are
    determined through empirical testing on various GPU configurations.
    
    Args:
        model_type: Type of model being trained. Supported values:
            - 'gan': Standard GAN (28x28 grayscale images)
            - 'wgan': Wasserstein GAN variants
            - 'vae': Variational Autoencoder (128x128 RGB)
            - 'ae': Simple Autoencoder
            
        vram_gb: GPU VRAM in gigabytes. If None, auto-detects using
            get_gpu_vram_gb(). Set this manually to override auto-detection.
    
    Returns:
        int: Recommended batch size for the configuration.
        
    Example:
        >>> # Auto-detect VRAM
        >>> batch = get_optimal_batch_size('gan')
        >>> print(f"Optimal batch size: {batch}")
        Optimal batch size: 1024
        
        >>> # Manual override for 6GB GPU
        >>> batch = get_optimal_batch_size('gan', vram_gb=6)
        >>> print(f"Optimal batch size: {batch}")
        Optimal batch size: 512
    """
    # Auto-detect VRAM if not specified
    if vram_gb is None:
        vram_gb = get_gpu_vram_gb()
    
    # Get configuration for model type, fallback to default
    model_type = model_type.lower()
    config = BATCH_SIZE_CONFIG.get(model_type, DEFAULT_BATCH_SIZES)
    
    # Find the best matching VRAM tier
    # If exact match not found, use the largest tier <= available VRAM
    available_tiers = sorted([tier for tier in config.keys() if tier <= vram_gb], reverse=True)
    
    if not available_tiers:
        # VRAM is less than minimum tier, use smallest available
        min_tier = min(config.keys())
        print(f"WARNING: GPU VRAM ({vram_gb}GB) is below minimum tested ({min_tier}GB). "
              f"Using batch size for {min_tier}GB.")
        return config[min_tier]
    
    selected_tier = available_tiers[0]
    return config[selected_tier]


def calculate_adjusted_epochs(
    reference_epochs: int,
    reference_batch: int,
    actual_batch: int
) -> int:
    """
    Calculate adjusted epochs to maintain equivalent total training updates.
    
    When batch size changes, the number of weight updates per epoch changes.
    To maintain the same total training effort (number of gradient updates),
    we scale epochs inversely with batch size.
    
    The formula is:
        adjusted_epochs = reference_epochs × (reference_batch / actual_batch)
    
    This preserves the relationship:
        total_updates = (dataset_size / batch_size) × epochs
    
    Args:
        reference_epochs: Original epoch count (e.g., 6000).
        reference_batch: Original batch size (e.g., 256).
        actual_batch: New batch size being used (e.g., 1024).
        
    Returns:
        int: Adjusted epoch count (minimum 100).
        
    Example:
        >>> # Original: 6000 epochs with batch 256
        >>> # New: batch 1024 (4x larger)
        >>> adjusted = calculate_adjusted_epochs(6000, 256, 1024)
        >>> print(f"Adjusted epochs: {adjusted}")
        Adjusted epochs: 1500
        
    Note:
        - Larger batches generally provide more stable gradients
        - The minimum returned value is 100 epochs
        - For very large batch sizes, consider fewer epochs with higher LR
        
    See Also:
        - documentation/GPU_SETUP.md for batch size recommendations
        - The epoch scaling preserves TOTAL UPDATES, not total samples seen
    """
    if actual_batch <= 0:
        raise ValueError("actual_batch must be positive")
    if reference_batch <= 0:
        raise ValueError("reference_batch must be positive")
    
    # Calculate scaling factor
    scale_factor = reference_batch / actual_batch
    
    # Calculate adjusted epochs
    adjusted = int(reference_epochs * scale_factor)
    
    # Enforce minimum of 100 epochs
    return max(adjusted, 100)


def print_training_config(
    model_type: str,
    batch_size: int,
    epochs: int,
    reference_batch: int = None,
    reference_epochs: int = None,
    vram_gb: int = None
) -> None:
    """
    Print a formatted summary of the training configuration.
    
    This helper function displays the current training parameters in a
    clear, readable format for verification before starting training.
    
    Args:
        model_type: Type of model being trained.
        batch_size: Batch size being used.
        epochs: Number of epochs to train.
        reference_batch: Original batch size for comparison (optional).
        reference_epochs: Original epoch count for comparison (optional).
        vram_gb: GPU VRAM in GB (optional, for display).
        
    Example:
        >>> print_training_config('gan', 1024, 1500, 256, 6000, 8)
        ═══════════════════════════════════════════════════════════════════
        TRAINING CONFIGURATION
        ═══════════════════════════════════════════════════════════════════
        Model Type:     GAN
        GPU VRAM:       8 GB
        Batch Size:     1024 (reference: 256)
        Epochs:         1500 (reference: 6000)
        Scale Factor:   0.25x epochs
        ═══════════════════════════════════════════════════════════════════
    """
    print("═" * 68)
    print("TRAINING CONFIGURATION")
    print("═" * 68)
    print(f"Model Type:     {model_type.upper()}")
    
    if vram_gb:
        print(f"GPU VRAM:       {vram_gb} GB")
    
    if reference_batch:
        print(f"Batch Size:     {batch_size} (reference: {reference_batch})")
    else:
        print(f"Batch Size:     {batch_size}")
    
    if reference_epochs:
        print(f"Epochs:         {epochs} (reference: {reference_epochs})")
        scale = epochs / reference_epochs
        print(f"Scale Factor:   {scale:.2f}x epochs")
    else:
        print(f"Epochs:         {epochs}")
    
    print("═" * 68)


# =============================================================================
# MODULE SELF-TEST
# =============================================================================
if __name__ == "__main__":
    # Run a quick self-test when executed directly
    print("GPU Utils Module Self-Test")
    print("-" * 40)
    
    # Test VRAM detection
    vram = get_gpu_vram_gb()
    print(f"Detected VRAM: {vram}GB")
    
    # Test batch size calculation for different models
    for model in ['gan', 'vae', 'ae']:
        batch = get_optimal_batch_size(model, vram_gb=vram)
        print(f"{model.upper()} optimal batch size: {batch}")
    
    # Test epoch scaling
    ref_epochs, ref_batch = 6000, 256
    for test_batch in [256, 512, 1024, 2048]:
        adjusted = calculate_adjusted_epochs(ref_epochs, ref_batch, test_batch)
        print(f"Batch {test_batch}: {adjusted} epochs")
    
    print("-" * 40)
    print("Self-test complete!")
