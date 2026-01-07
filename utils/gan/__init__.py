"""
GAN Utilities Package.

This package provides utility modules for GAN training, analysis, and evaluation.

Modules:
    stability_analysis: Automated stability analysis for GAN training runs
    report_generator: Markdown report generation for training analysis
    metrics: Core training metrics (gradient norms, weight stats, timing)
    quality_metrics: Image quality metrics (FID, Inception Score, diversity)

Example Usage:
    # Stability Analysis
    from utils.gan.stability_analysis import analyze_training_run
    analysis = analyze_training_run(d_losses, g_losses)

    # Report Generation
    from utils.gan.report_generator import generate_run_report
    report_path = generate_run_report(run_folder, config, d_losses, g_losses)

    # Training Metrics
    from utils.gan.metrics import collect_epoch_metrics, format_verbose_output
    metrics = collect_epoch_metrics(critic, generator, d_loss, g_loss, ...)

    # Quality Metrics (every N epochs)
    from utils.gan.quality_metrics import collect_quality_metrics
    quality = collect_quality_metrics(real_images, fake_images)

Author:
    Generated with comprehensive PEP-8 documentation.
"""

# Stability Analysis
from utils.gan.stability_analysis import (
    analyze_training_run,
    compute_phase_metrics,
    generate_verdict,
)

# Report Generation
from utils.gan.report_generator import generate_run_report

# Training Metrics
from utils.gan.metrics import (
    collect_epoch_metrics,
    format_verbose_output,
    compute_weight_stats,
    compute_clip_ratio,
    compute_loss_variance,
    compute_dg_ratio,
    compute_wasserstein_distance,
    EpochTimer,
)

# Quality Metrics
from utils.gan.quality_metrics import (
    collect_quality_metrics,
    compute_fid_score,
    compute_inception_score,
    compute_pixel_variance,
    compute_image_stats,
    format_quality_output,
)

__all__ = [
    # Stability
    'analyze_training_run',
    'compute_phase_metrics',
    'generate_verdict',
    # Reports
    'generate_run_report',
    # Metrics
    'collect_epoch_metrics',
    'format_verbose_output',
    'compute_weight_stats',
    'compute_clip_ratio',
    'compute_loss_variance',
    'compute_dg_ratio',
    'compute_wasserstein_distance',
    'EpochTimer',
    # Quality
    'collect_quality_metrics',
    'compute_fid_score',
    'compute_inception_score',
    'compute_pixel_variance',
    'compute_image_stats',
    'format_quality_output',
]
