"""
Report Generator for GAN Training Analysis.

This module generates markdown analysis reports for each training run,
saved to the run folder for documentation and version control.

Module Location:
    utils/gan/report_generator.py

Usage:
    from utils.gan.report_generator import generate_run_report

    report_path = generate_run_report(
        run_folder="run/gan/0002_horses",
        config={...},
        d_losses=gan.d_losses,
        g_losses=gan.g_losses,
        wandb_url="https://wandb.ai/..."
    )

Author:
    Generated with comprehensive PEP-8 documentation.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Import stability analysis from the same package
try:
    from utils.gan.stability_analysis import analyze_training_run
except ImportError:
    # Handle case when running from different directory
    from stability_analysis import analyze_training_run


def format_phase_metrics_table(phase_metrics: Dict[str, Dict]) -> str:
    """
    Format phase metrics as a markdown table.
    
    Args:
        phase_metrics: Dict from analyze_training_run()
        
    Returns:
        Markdown table string
    """
    header = "| Phase | Epoch Range | D Loss (Start → End) | G Loss (Start → End) | Δ D/epoch | Δ G/epoch |"
    separator = "|-------|-------------|----------------------|----------------------|-----------|-----------|"
    rows = []
    
    for phase_name, metrics in phase_metrics.items():
        epoch_range = f"{metrics['epoch_range'][0]}-{metrics['epoch_range'][1]}"
        d_range = f"{metrics['d_loss_start']:.2f} → {metrics['d_loss_end']:.2f}"
        g_range = f"{metrics['g_loss_start']:.2f} → {metrics['g_loss_end']:.2f}"
        d_delta = f"{metrics['d_delta_per_epoch']:.4f}"
        g_delta = f"{metrics['g_delta_per_epoch']:.4f}"
        
        rows.append(f"| {phase_name.capitalize()} | {epoch_range} | {d_range} | {g_range} | {d_delta} | {g_delta} |")
    
    return "\n".join([header, separator] + rows)


def format_stability_table(indicators: Dict[str, Tuple[bool, str]]) -> str:
    """
    Format stability indicators as a markdown table.
    
    Args:
        indicators: Dict from analyze_training_run()
        
    Returns:
        Markdown table string
    """
    header = "| Indicator | Status | Observation |"
    separator = "|-----------|--------|-------------|"
    rows = []
    
    for indicator_name, (passed, observation) in indicators.items():
        status = "✅ Good" if passed else "❌ Issue"
        display_name = indicator_name.replace('_', ' ').title()
        rows.append(f"| {display_name} | {status} | {observation} |")
    
    return "\n".join([header, separator] + rows)


def format_config_table(config: Dict[str, Any]) -> str:
    """
    Format configuration as a markdown table.
    
    Args:
        config: Training configuration dict
        
    Returns:
        Markdown table string
    """
    header = "| Parameter | Value |"
    separator = "|-----------|-------|"
    rows = []
    
    for key, value in config.items():
        # Format value appropriately
        if isinstance(value, float):
            formatted_value = f"{value:.6g}"
        elif isinstance(value, list):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        display_key = key.replace('_', ' ').title()
        rows.append(f"| {display_key} | {formatted_value} |")
    
    return "\n".join([header, separator] + rows)


def generate_run_report(
    run_folder: str,
    config: Dict[str, Any],
    d_losses: List[Tuple[float, float, float]],
    g_losses: List[float],
    wandb_url: str = None,
    notes: str = None
) -> str:
    """
    Generate a complete analysis report markdown file.
    
    Args:
        run_folder: Path to run folder (e.g., "run/gan/0002_horses")
        config: Training configuration dict
        d_losses: List of (d_loss_total, d_loss_real, d_loss_fake) tuples
        g_losses: List of generator loss values
        wandb_url: Optional W&B run URL
        notes: Optional observation notes
        
    Returns:
        Path to generated report file
    """
    # Run stability analysis
    analysis = analyze_training_run(d_losses, g_losses)
    
    # Extract verdict
    verdict = analysis['verdict']
    
    # Generate report timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = os.path.basename(run_folder)
    
    # Build report content
    report = f"""# Training Analysis Report: {run_id}

**Generated**: {timestamp}  
**Total Epochs**: {analysis['total_epochs']}  
**Final D Loss**: {analysis['final_d_loss']:.4f}  
**Final G Loss**: {analysis['final_g_loss']:.4f}  
"""
    
    # Add W&B link if provided
    if wandb_url:
        report += f"**W&B Run**: [View on W&B]({wandb_url})\n"
    
    report += "\n---\n\n"
    
    # Overall Verdict Section
    report += f"""## Training Verdict

| Metric | Value |
|--------|-------|
| **Stability** | {verdict['stability']} |
| **Quality** | {verdict['quality']} |
| **Score** | {verdict['passed']}/{verdict['total']} indicators passed |
| **Recommendation** | {verdict['recommendation']} |

---

"""
    
    # Configuration Section
    report += "## Configuration\n\n"
    report += format_config_table(config)
    report += "\n\n---\n\n"
    
    # Phase-wise Metrics Section
    report += "## Training Progression (Phase-wise Metrics)\n\n"
    report += format_phase_metrics_table(analysis['phase_metrics'])
    report += "\n\n---\n\n"
    
    # Stability Analysis Section
    report += "## Stability Indicators\n\n"
    report += format_stability_table(analysis['indicators'])
    report += "\n\n"
    
    # Add interpretation
    report += """### Interpretation

**Wasserstein Loss Understanding:**
- **D loss = E[critic(real)] - E[critic(fake)]**: Critic maximizes this
- **G loss = -E[critic(fake)]**: Generator minimizes this

**Expected WGAN Behavior:**
- D loss should be positive and gradually increasing
- |G loss| should increase as generator improves
- Real/Fake discrimination should remain balanced

---

"""
    
    # Notes Section
    if notes:
        report += f"## Notes\n\n{notes}\n\n---\n\n"
    
    # Footer with W&B reference
    report += """## Full Details

For complete metrics, loss curves, and generated images, see the W&B run dashboard.
"""
    
    if wandb_url:
        report += f"\n[View Full Report on W&B]({wandb_url})\n"
    
    # Write report to file
    report_path = os.path.join(run_folder, "analysis_report.md")
    
    # Ensure run folder exists
    os.makedirs(run_folder, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✓ Analysis report saved to: {report_path}")
    
    return report_path
