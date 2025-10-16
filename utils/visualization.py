"""
Visualization utilities for beat prediction results.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(
        input_sequence: np.ndarray,
        target_sequence: np.ndarray,
        predicted_sequence: np.ndarray,
        period: float,
        dt: float,
        save_path: Optional[str] = None,
        show: bool = True
) -> None:
    """
    Plot input, target, and predicted sequences.

    Args:
        input_sequence: Input beat sequence
        target_sequence: Target (ground truth) sequence
        predicted_sequence: Model predictions
        period: Period used for this sequence
        dt: Time step in seconds
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Create time axis
    n_timesteps = len(input_sequence)
    time = np.arange(n_timesteps) * dt

    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot input
    axes[0].plot(time, input_sequence, 'b-', label='Input', linewidth=1.5)
    axes[0].set_ylabel('Input', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'Beat Prediction - Period: {period:.3f}s', fontsize=12)

    # Plot target and prediction
    axes[1].plot(time, target_sequence, 'k--', label='Target', linewidth=1.5)
    axes[1].plot(time, predicted_sequence, 'r-', label='Output', alpha=0.8, linewidth=2)
    axes[1].plot(time, input_sequence, 'b--', label='Input', linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel('Target and Pred', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    # Show figure
    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_vs_period(
        periods: List[float],
        mse_values: List[float],
        save_path: Optional[str] = None,
        show: bool = True
) -> None:
    """
    Plot MSE vs period to show model performance across different periods.

    Args:
        periods: List of test periods
        mse_values: MSE for each period
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot MSE vs period
    ax.plot(periods, mse_values, 'bo-', linewidth=2, markersize=8)

    # Labels and title
    ax.set_xlabel('Period (s)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Model Performance vs Beat Period', fontsize=14)

    # Grid
    ax.grid(True, alpha=0.3)

    # Add value labels on points
    for period, mse in zip(periods, mse_values):
        ax.annotate(f'{mse:.4f}',
                    xy=(period, mse),
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=8,
                    ha='center')

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    # Show figure
    if show:
        plt.show()
    else:
        plt.close()


def create_test_report(
        test_results: Dict[str, Any],
        save_path: str
) -> None:
    """
    Create a simple text report of test results.

    Args:
        test_results: Dictionary with test results
        save_path: Path to save report
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Beat Prediction Test Results\n")
        f.write("=" * 60 + "\n\n")

        # Overall statistics
        f.write("Overall Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean MSE: {test_results['mean_mse']:.6f}\n")
        f.write(f"Std MSE: {test_results['std_mse']:.6f}\n")
        f.write(f"Min MSE: {test_results['min_mse']:.6f}\n")
        f.write(f"Max MSE: {test_results['max_mse']:.6f}\n\n")

        # Performance by period
        f.write("Performance by Period:\n")
        f.write("-" * 30 + "\n")
        for period, mse in zip(test_results['periods'], test_results['mse_per_period']):
            f.write(f"Period {period:.3f}s: MSE = {mse:.6f}\n")

        f.write("\n" + "=" * 60 + "\n")
