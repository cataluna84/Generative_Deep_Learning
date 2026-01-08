"""
GPU Utilities Module for Dynamic Training Configuration.

This module provides utilities for optimizing deep learning training based on
available GPU VRAM. It automatically detects GPU memory and dynamically
calculates optimal batch sizes using binary search with OOM detection.

Key Features:
    - Automatic GPU VRAM detection using TensorFlow
    - Dynamic batch size finder using binary search + OOM detection
    - Parameter logging to console and W&B
    - Epoch scaling to maintain total training updates

Example Usage:
    >>> from utils.gpu_utils import find_optimal_batch_size, get_gpu_vram_gb
    >>>
    >>> # Build your model first
    >>> model = create_my_model()
    >>>
    >>> # Find optimal batch size dynamically
    >>> batch_size = find_optimal_batch_size(
    ...     model=model,
    ...     input_shape=(28, 28, 1),
    ... )
    >>>
    >>> # Scale epochs to maintain equivalent training updates
    >>> epochs = calculate_adjusted_epochs(
    ...     reference_epochs=200,
    ...     reference_batch=32,
    ...     actual_batch=batch_size
    ... )

References:
    - documentation/GPU_SETUP.md: GPU configuration guide
    - documentation/NOTEBOOK_STANDARDIZATION.md: Dynamic training workflow

Created: 2026-01-02
Updated: 2026-01-04 (Added dynamic batch finder, removed static tables)
"""

import tensorflow as tf
import numpy as np
import gc
from typing import Optional, Tuple, Dict, Any

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# GPU DETECTION
# =============================================================================

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
        gpu = gpus[0]
        
        # Try to get memory limit if set
        try:
            # Build device name for memory info query
            device_name = gpu.name.replace('/physical_device:', '')
            memory_info = tf.config.experimental.get_memory_info(device_name)
            if 'total' in memory_info:
                vram_bytes = memory_info['total']
                vram_gb = int(vram_bytes / (1024 ** 3))
                return max(vram_gb, 4)  # Minimum 4GB
        except (RuntimeError, AttributeError, KeyError):
            pass
        
        return 8
            
    except Exception as e:
        print(f"WARNING: Error detecting GPU VRAM: {e}. Using default of 8GB.")
        return 8


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory info in MB:
            - current_mb: Currently allocated memory
            - peak_mb: Peak memory usage
    """
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        return {
            'current_mb': info.get('current', 0) / (1024 ** 2),
            'peak_mb': info.get('peak', 0) / (1024 ** 2),
        }
    except Exception:
        return {'current_mb': 0, 'peak_mb': 0}


# =============================================================================
# DYNAMIC BATCH SIZE FINDER
# =============================================================================

def find_optimal_batch_size(
    model: tf.keras.Model,
    input_shape: Tuple[int, ...],
    min_batch_size: int = 2,
    max_batch_size: int = 4096,
    safety_factor: float = 0.9,
    verbose: bool = True,
    log_to_wandb: bool = True,
) -> int:
    """
    Find the optimal batch size using binary search with OOM detection.
    
    This function tests progressively larger batch sizes until an OOM error
    occurs, then uses binary search to find the largest batch size that fits.
    
    Args:
        model: Compiled Keras model to test. Must be callable.
        input_shape: Shape of a single input sample (H, W, C) without batch dim.
        min_batch_size: Minimum batch size to test. Default: 2.
        max_batch_size: Maximum batch size to test. Default: 4096.
        safety_factor: Fraction of max batch to return (0.9 = 90%). Default: 0.9.
        verbose: Print progress messages. Default: True.
        log_to_wandb: Log results to W&B if available. Default: True.
        
    Returns:
        int: Optimal batch size that fits in GPU memory.
        
    Example:
        >>> model = create_vae_model()
        >>> optimal_batch = find_optimal_batch_size(model, (28, 28, 1))
        ════════════════════════════════════════════════════════════════
        DYNAMIC BATCH SIZE FINDER
        ════════════════════════════════════════════════════════════════
        Model Parameters: 1,234,567
        Estimated Model Memory: 19.8 MB (weights + optimizer + gradients)
        Input Shape: (28, 28, 1)
        ────────────────────────────────────────────────────────────────
        Testing batch sizes...
          batch_size=    2 ✓
          ...
          batch_size= 2048 ✗ OOM
        ────────────────────────────────────────────────────────────────
        ✓ Optimal batch size: 1382
        ════════════════════════════════════════════════════════════════
    """
    # =========================================================================
    # Step 1: Log model information
    # =========================================================================
    model_info = _get_model_info(model, input_shape)
    
    if verbose:
        print("═" * 64)
        print("DYNAMIC BATCH SIZE FINDER")
        print("═" * 64)
        print(f"Model Parameters: {model_info['params']:,}")
        print(f"Estimated Model Memory: {model_info['memory_mb']:.1f} MB "
              "(weights + optimizer + gradients)")
        print(f"Input Shape: {input_shape}")
        print("─" * 64)
        print("Testing batch sizes...")
    
    # Log to W&B if available
    if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "batch_finder/model_params": model_info['params'],
            "batch_finder/model_memory_mb": model_info['memory_mb'],
            "batch_finder/input_shape": str(input_shape),
        })
    
    # =========================================================================
    # Step 2: Exponential growth to find upper bound
    # =========================================================================
    last_success = min_batch_size
    first_oom = None
    batch_size = min_batch_size
    
    while batch_size <= max_batch_size:
        success = _test_batch_size(model, input_shape, batch_size, verbose)
        
        if success:
            last_success = batch_size
            batch_size *= 2
        else:
            first_oom = batch_size
            break
    
    # If no OOM occurred, return max tested with safety factor
    if first_oom is None:
        result = int(last_success * safety_factor)
        result = _round_to_multiple(result, 32)
        
        if verbose:
            print("─" * 64)
            print(f"✓ No OOM detected. Using: {result}")
            print("═" * 64)
        
        _log_result_to_wandb(result, model_info, log_to_wandb)
        return result
    
    # =========================================================================
    # Step 3: Binary search between last_success and first_oom
    # =========================================================================
    if verbose:
        print(f"Binary search: {last_success} - {first_oom}")
    
    low, high = last_success, first_oom
    
    while high - low > low * 0.1:  # Stop when within 10%
        mid = (low + high) // 2
        
        # Round to nice numbers for efficiency
        if mid > 64:
            mid = (mid // 32) * 32  # Round to multiple of 32
        
        if mid <= low or mid >= high:
            break
            
        success = _test_batch_size(model, input_shape, mid, verbose)
        
        if success:
            low = mid
        else:
            high = mid
    
    # =========================================================================
    # Step 4: Apply safety factor and return
    # =========================================================================
    raw_result = low
    result = int(raw_result * safety_factor)
    result = max(min_batch_size, result)
    
    # Round down to multiple of 32 for GPU efficiency
    if result > 64:
        result = (result // 32) * 32
    
    if verbose:
        print("─" * 64)
        print(f"✓ Optimal batch size: {result} "
              f"({raw_result} × {safety_factor} safety)")
        print("═" * 64)
    
    _log_result_to_wandb(result, model_info, log_to_wandb)
    
    return result


# =============================================================================
# EPOCH SCALING
# =============================================================================

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
    
    Args:
        reference_epochs: Original epoch count (e.g., 200).
        reference_batch: Original batch size (e.g., 32).
        actual_batch: New batch size being used (e.g., 1024).
        
    Returns:
        int: Adjusted epoch count (minimum 100).
        
    Example:
        >>> # Original: 200 epochs with batch 32
        >>> # New: batch 1024 (32x larger)
        >>> adjusted = calculate_adjusted_epochs(200, 32, 1024)
        >>> print(f"Adjusted epochs: {adjusted}")
        Adjusted epochs: 100
    """
    if actual_batch <= 0:
        raise ValueError("actual_batch must be positive")
    if reference_batch <= 0:
        raise ValueError("reference_batch must be positive")
    
    # Calculate scaling factor
    scale_factor = reference_batch / actual_batch
    
    # Calculate adjusted epochs
    adjusted = int(reference_epochs * scale_factor)
    
    # Enforce minimum of 100 epochs to ensure sufficient training iterations
    # for model convergence. With large batch sizes, the number of gradient
    # updates per epoch decreases significantly, so a higher minimum ensures
    # the model sees enough training iterations to learn effectively.
    return max(adjusted, 100)


# =============================================================================
# CONFIGURATION PRINTING
# =============================================================================

def print_training_config(
    batch_size: int,
    epochs: int,
    model_params: int = None,
    reference_batch: int = None,
    reference_epochs: int = None,
    vram_gb: int = None
) -> None:
    """
    Print a formatted summary of the training configuration.
    
    Args:
        batch_size: Batch size being used.
        epochs: Number of epochs to train.
        model_params: Number of model parameters (optional).
        reference_batch: Original batch size for comparison (optional).
        reference_epochs: Original epoch count for comparison (optional).
        vram_gb: GPU VRAM in GB (optional, for display).
        
    Example:
        >>> print_training_config(1024, 50, model_params=1234567, vram_gb=8)
    """
    print("═" * 64)
    print("TRAINING CONFIGURATION")
    print("═" * 64)
    
    if vram_gb:
        print(f"GPU VRAM:       {vram_gb} GB")
    
    if model_params:
        print(f"Model Params:   {model_params:,}")
    
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
    
    print("═" * 64)


# =============================================================================
# HELPER FUNCTIONS (PRIVATE)
# =============================================================================

def _get_model_info(model: tf.keras.Model, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Extract model information for logging."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    
    # Memory estimation:
    # - Weights: params × 4 bytes (float32)
    # - Gradients: params × 4 bytes
    # - Adam optimizer: params × 8 bytes (m and v)
    # Total: params × 16 bytes
    bytes_per_param = 16
    memory_bytes = total_params * bytes_per_param
    memory_mb = memory_bytes / (1024 ** 2)
    
    return {
        'params': total_params,
        'trainable_params': trainable_params,
        'memory_mb': memory_mb,
        'input_shape': input_shape,
    }


def _test_batch_size(
    model: tf.keras.Model,
    input_shape: Tuple[int, ...],
    batch_size: int,
    verbose: bool = True
) -> bool:
    """Test if a batch size fits in GPU memory."""
    try:
        # Create test batch
        test_input = tf.random.normal([batch_size] + list(input_shape))
        
        # Run forward + backward pass (simulates training)
        with tf.GradientTape() as tape:
            output = model(test_input, training=True)
            
            # Handle different output types
            if isinstance(output, (list, tuple)):
                loss = tf.reduce_mean(tf.cast(output[0], tf.float32))
            else:
                loss = tf.reduce_mean(tf.cast(output, tf.float32))
        
        # Compute gradients
        if model.trainable_variables:
            _ = tape.gradient(loss, model.trainable_variables)
        
        # Clean up
        del test_input, output, loss
        gc.collect()
        
        if verbose:
            print(f"  batch_size={batch_size:5d} ✓")
        return True
        
    except tf.errors.ResourceExhaustedError:
        if verbose:
            print(f"  batch_size={batch_size:5d} ✗ OOM")
        gc.collect()
        return False
        
    except Exception as e:
        if verbose:
            print(f"  batch_size={batch_size:5d} ✗ Error: {type(e).__name__}")
        gc.collect()
        return False


def _round_to_multiple(n: int, multiple: int) -> int:
    """Round down to nearest multiple for GPU efficiency."""
    if n < multiple:
        return n
    return (n // multiple) * multiple


def _log_result_to_wandb(
    batch_size: int,
    model_info: Dict[str, Any],
    log_to_wandb: bool
) -> None:
    """Log final results to W&B if available."""
    if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "batch_finder/optimal_batch_size": batch_size,
        })
        # Update config
        wandb.config.update({
            "batch_size": batch_size,
            "batch_size_source": "dynamic_finder",
            "model_params": model_info['params'],
        }, allow_val_change=True)


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("GPU Utils Module - Self Test")
    print("-" * 40)
    
    # Test VRAM detection
    vram = get_gpu_vram_gb()
    print(f"Detected VRAM: {vram}GB")
    
    # Create a SMALL test model (avoid Flatten which creates huge param count)
    print("\nCreating small test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),  # Much smaller than Flatten
        tf.keras.layers.Dense(10),
    ])
    print(f"Model params: {model.count_params():,}")
    
    # Test the dynamic batch finder with reduced max for quick testing
    print("\nTesting dynamic batch finder...")
    optimal = find_optimal_batch_size(
        model=model,
        input_shape=(28, 28, 1),
        min_batch_size=64,
        max_batch_size=512,  # Reduced for quick test
        verbose=True,
        log_to_wandb=False,
    )
    
    # Test epoch scaling
    print("\nTesting epoch scaling...")
    for test_batch in [32, 64, 128, 256, 512, 1024]:
        adjusted = calculate_adjusted_epochs(200, 32, test_batch)
        print(f"  Batch {test_batch}: {adjusted} epochs")
    
    print("-" * 40)
    print("Self-test complete!")

