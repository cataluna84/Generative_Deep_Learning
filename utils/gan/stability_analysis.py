"""
Stability Analysis Utility for GAN Training.

This module provides automated analysis of GAN training stability by
examining loss curves and computing key indicators.

Usage:
    from utils.stability_analysis import analyze_training_run
    
    analysis = analyze_training_run(d_losses, g_losses)
    print(analysis['verdict'])
"""

import numpy as np
from typing import Dict, List, Tuple, Any


# =============================================================================
# PHASE CONFIGURATION
# =============================================================================
# Default phase ranges for 6000 epoch training
DEFAULT_PHASES = {
    'warmup': (0, 50),
    'early': (50, 500),
    'mid': (500, 2000),
    'late': (2000, 4000),
    'final': (4000, 6000)
}


def compute_phase_metrics(
    d_losses: List[Tuple[float, float, float]],
    g_losses: List[float],
    phases: Dict[str, Tuple[int, int]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics for each training phase.
    
    Args:
        d_losses: List of (d_loss_total, d_loss_real, d_loss_fake) tuples
        g_losses: List of generator loss values
        phases: Dict mapping phase names to (start_epoch, end_epoch)
        
    Returns:
        Dict with phase names as keys, containing metrics for each phase
    """
    if phases is None:
        # Scale phases to match actual training length
        total_epochs = len(d_losses)
        scale = total_epochs / 6000
        phases = {
            name: (int(start * scale), int(end * scale))
            for name, (start, end) in DEFAULT_PHASES.items()
        }
    
    phase_metrics = {}
    
    for phase_name, (start, end) in phases.items():
        # Clamp to valid range
        start = max(0, start)
        end = min(len(d_losses), end)
        
        if start >= end:
            continue
            
        # Extract phase data
        d_phase = d_losses[start:end]
        g_phase = g_losses[start:end]
        
        # Compute metrics
        d_totals = [d[0] for d in d_phase]
        d_reals = [d[1] for d in d_phase]
        d_fakes = [d[2] for d in d_phase]
        
        phase_metrics[phase_name] = {
            'epoch_range': (start, end),
            'd_loss_start': d_totals[0],
            'd_loss_end': d_totals[-1],
            'd_loss_mean': np.mean(d_totals),
            'g_loss_start': g_phase[0],
            'g_loss_end': g_phase[-1],
            'g_loss_mean': np.mean(g_phase),
            'd_delta_per_epoch': (d_totals[-1] - d_totals[0]) / len(d_totals),
            'g_delta_per_epoch': (g_phase[-1] - g_phase[0]) / len(g_phase),
            'real_fake_balance': np.mean(np.abs(
                np.array(d_reals) - np.array(d_fakes)
            ))
        }
    
    return phase_metrics


def check_monotonicity(
    losses: List[float],
    window: int = 100,
    threshold: float = 0.3
) -> Tuple[bool, str]:
    """
    Check if losses change smoothly without excessive oscillations.
    
    Args:
        losses: List of loss values
        window: Window size for computing local variance
        threshold: Max allowed ratio of local variance to global change
        
    Returns:
        Tuple of (is_stable, observation_text)
    """
    if len(losses) < window * 2:
        return True, "Insufficient data for monotonicity check"
    
    # Compute local variance in sliding windows
    losses_arr = np.array(losses)
    local_vars = []
    
    for i in range(0, len(losses) - window, window // 2):
        local_vars.append(np.std(losses_arr[i:i + window]))
    
    avg_local_var = np.mean(local_vars)
    global_change = abs(losses[-1] - losses[0])
    
    # Stable if local variance is small relative to global change
    if global_change == 0:
        is_stable = avg_local_var < 0.1
    else:
        is_stable = (avg_local_var / global_change) < threshold
    
    if is_stable:
        observation = "D and G losses change smoothly without oscillations"
    else:
        observation = f"High local variance detected (ratio: {avg_local_var/max(global_change, 0.001):.2f})"
    
    return is_stable, observation


def check_balance(
    d_losses: List[Tuple[float, float, float]],
    threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if critic's real/fake discrimination remains balanced.
    
    Args:
        d_losses: List of (d_loss_total, d_loss_real, d_loss_fake) tuples
        threshold: Max allowed average difference between real/fake scores
        
    Returns:
        Tuple of (is_balanced, observation_text)
    """
    if not d_losses:
        return True, "No loss data available"
    
    # Compute average real/fake difference
    diffs = [abs(d[1] - d[2]) for d in d_losses]
    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    is_balanced = avg_diff < threshold
    
    if is_balanced:
        observation = f"D_loss_real ≈ D_loss_fake throughout training (avg diff: {avg_diff:.3f})"
    else:
        observation = f"Imbalanced discrimination detected (avg diff: {avg_diff:.3f}, max: {max_diff:.3f})"
    
    return is_balanced, observation


def detect_mode_collapse(
    g_losses: List[float],
    plateau_threshold: float = 0.001,
    plateau_length: int = 200
) -> Tuple[bool, str]:
    """
    Detect potential mode collapse from generator loss patterns.
    
    Mode collapse often manifests as sudden plateaus or dramatic drops
    in generator loss.
    
    Args:
        g_losses: List of generator loss values
        plateau_threshold: Max change rate to consider a plateau
        plateau_length: Min consecutive epochs of plateau to flag
        
    Returns:
        Tuple of (no_collapse_detected, observation_text)
    """
    if len(g_losses) < plateau_length:
        return True, "Insufficient data for mode collapse detection"
    
    g_arr = np.array(g_losses)
    
    # Check for extended plateaus
    diffs = np.abs(np.diff(g_arr))
    plateau_count = 0
    max_plateau = 0
    
    for diff in diffs:
        if diff < plateau_threshold:
            plateau_count += 1
            max_plateau = max(max_plateau, plateau_count)
        else:
            plateau_count = 0
    
    has_plateau = max_plateau >= plateau_length
    
    # Check for sudden dramatic changes (>50% of total range in 10 epochs)
    total_range = abs(g_arr[-1] - g_arr[0])
    sudden_changes = 0
    
    for i in range(len(g_arr) - 10):
        local_change = abs(g_arr[i + 10] - g_arr[i])
        if total_range > 0 and (local_change / total_range) > 0.5:
            sudden_changes += 1
    
    has_sudden_change = sudden_changes > 0
    
    no_collapse = not (has_plateau or has_sudden_change)
    
    if no_collapse:
        observation = "No sudden plateaus or repetitive outputs observed"
    else:
        issues = []
        if has_plateau:
            issues.append(f"plateau of {max_plateau} epochs")
        if has_sudden_change:
            issues.append(f"{sudden_changes} sudden changes")
        observation = f"Potential mode collapse: {', '.join(issues)}"
    
    return no_collapse, observation


def check_gradient_signal(
    d_losses: List[Tuple[float, float, float]]
) -> Tuple[bool, str]:
    """
    Check if critic maintains discrimination ability (healthy gradients).
    
    The critic should maintain positive D loss, indicating it can still
    distinguish real from fake samples.
    
    Args:
        d_losses: List of (d_loss_total, d_loss_real, d_loss_fake) tuples
        
    Returns:
        Tuple of (is_healthy, observation_text)
    """
    if not d_losses:
        return True, "No loss data available"
    
    d_totals = [d[0] for d in d_losses]
    
    # Check final D loss is positive (critic still discriminating)
    final_d = d_totals[-1] if d_totals else 0
    
    # Check D loss is not saturating (all same value)
    d_std = np.std(d_totals[-100:]) if len(d_totals) > 100 else np.std(d_totals)
    
    is_healthy = final_d > 0 and d_std > 0.001
    
    if is_healthy:
        observation = "Critic maintains discrimination ability"
    else:
        if final_d <= 0:
            observation = f"Critic lost discrimination (final D loss: {final_d:.3f})"
        else:
            observation = f"Critic saturated (std: {d_std:.6f})"
    
    return is_healthy, observation


def check_wasserstein_distance(
    g_losses: List[float]
) -> Tuple[bool, str]:
    """
    Check if Wasserstein distance (|G loss|) is increasing as expected.
    
    In WGAN, the absolute value of generator loss should generally
    increase as training progresses.
    
    Args:
        g_losses: List of generator loss values
        
    Returns:
        Tuple of (is_increasing, observation_text)
    """
    if len(g_losses) < 100:
        return True, "Insufficient data"
    
    # Compare early vs late absolute values
    early_abs = abs(np.mean(g_losses[:100]))
    late_abs = abs(np.mean(g_losses[-100:]))
    
    is_increasing = late_abs > early_abs
    
    if is_increasing:
        observation = f"|G loss| grows steadily ({early_abs:.2f} → {late_abs:.2f})"
    else:
        observation = f"|G loss| not increasing as expected ({early_abs:.2f} → {late_abs:.2f})"
    
    return is_increasing, observation


def generate_verdict(indicators: Dict[str, Tuple[bool, str]]) -> Dict[str, Any]:
    """
    Generate overall training verdict from stability indicators.
    
    Args:
        indicators: Dict of indicator_name -> (passed, observation)
        
    Returns:
        Dict with verdict, quality assessment, and recommendation
    """
    passed_count = sum(1 for passed, _ in indicators.values() if passed)
    total_count = len(indicators)
    
    # Determine stability status
    if passed_count == total_count:
        stability = "✅ STABLE"
        quality = "Excellent"
        recommendation = "Continue with current hyperparameters or experiment with variations"
    elif passed_count >= total_count - 1:
        stability = "✅ STABLE"
        quality = "Good"
        recommendation = "Consider investigating the flagged indicator"
    elif passed_count >= total_count // 2:
        stability = "⚠️ UNSTABLE"
        quality = "Fair"
        recommendation = "Review training parameters and consider adjustments"
    else:
        stability = "❌ FAILED"
        quality = "Poor"
        recommendation = "Significant issues detected - review architecture and hyperparameters"
    
    return {
        'stability': stability,
        'quality': quality,
        'recommendation': recommendation,
        'passed': passed_count,
        'total': total_count
    }


def analyze_training_run(
    d_losses: List[Tuple[float, float, float]],
    g_losses: List[float],
    phases: Dict[str, Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Perform complete stability analysis on a training run.
    
    This is the main entry point for stability analysis.
    
    Args:
        d_losses: List of (d_loss_total, d_loss_real, d_loss_fake) tuples
        g_losses: List of generator loss values
        phases: Optional custom phase ranges
        
    Returns:
        Dict containing:
            - phase_metrics: Per-phase metrics
            - indicators: Stability indicators with pass/fail and observations
            - verdict: Overall assessment
    """
    # Compute phase-wise metrics
    phase_metrics = compute_phase_metrics(d_losses, g_losses, phases)
    
    # Extract D totals for monotonicity check
    d_totals = [d[0] for d in d_losses]
    
    # Run all stability checks
    indicators = {
        'monotonicity': check_monotonicity(d_totals),
        'balance': check_balance(d_losses),
        'mode_collapse': detect_mode_collapse(g_losses),
        'gradient_signal': check_gradient_signal(d_losses),
        'wasserstein_distance': check_wasserstein_distance(g_losses)
    }
    
    # Generate verdict
    verdict = generate_verdict(indicators)
    
    return {
        'phase_metrics': phase_metrics,
        'indicators': indicators,
        'verdict': verdict,
        'total_epochs': len(d_losses),
        'final_d_loss': d_losses[-1][0] if d_losses else None,
        'final_g_loss': g_losses[-1] if g_losses else None
    }
