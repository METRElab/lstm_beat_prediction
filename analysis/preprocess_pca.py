"""
Preprocess hidden and cell states using PCA for visualization.
Fits individual PCA for each period to capture its specific dynamics.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import pickle
from typing import Dict, Tuple


def fit_pca_for_period(
        states: np.ndarray,
        n_components: int = 3
) -> Tuple[np.ndarray, PCA, np.ndarray]:
    """
    Fit PCA for a single period's states.

    Args:
        states: States array of shape (timesteps, hidden_size)
        n_components: Number of PCA components

    Returns:
        reduced_states: Transformed states (timesteps, n_components)
        pca_model: Fitted PCA model
        explained_variance: Variance explained by each component
    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(states)

    return reduced, pca, pca.explained_variance_ratio_


def process_checkpoint(
        states_file: Path,
        output_dir: Path,
        n_components: int = 3
) -> Dict:
    """
    Process a single checkpoint file, fitting PCA for each period.

    Args:
        states_file: Path to epoch_X_states.npz file
        output_dir: Directory to save processed data
        n_components: Number of PCA components

    Returns:
        Metadata about the processing
    """
    # Load states
    data = np.load(states_file)
    hidden_states = data['hidden_states']  # (n_periods, timesteps, num_layers, hidden_size)
    cell_states = data['cell_states']
    periods = data['periods']
    mse_per_period = data['mse_per_period']
    epoch = data['epoch'].item()

    n_periods, timesteps, num_layers, hidden_size = hidden_states.shape

    # Prepare arrays for reduced states
    h_t_pca = np.zeros((n_periods, timesteps, n_components))
    c_t_pca = np.zeros((n_periods, timesteps, n_components))
    h_t_explained = np.zeros((n_periods, n_components))
    c_t_explained = np.zeros((n_periods, n_components))

    # Store PCA models for each period
    h_pca_models = []
    c_pca_models = []

    print(f"Processing epoch {epoch}...")

    for period_idx in range(n_periods):
        period = periods[period_idx]

        # Extract states for this period
        # Squeeze out the layer dimension since we have only 1 layer
        h_states = hidden_states[period_idx, :, 0, :]  # (timesteps, hidden_size)
        c_states = cell_states[period_idx, :, 0, :]  # (timesteps, hidden_size)

        # Fit PCA for hidden states
        h_reduced, h_pca, h_var = fit_pca_for_period(h_states, n_components)
        h_t_pca[period_idx] = h_reduced
        h_t_explained[period_idx] = h_var
        h_pca_models.append(h_pca)

        # Fit PCA for cell states
        c_reduced, c_pca, c_var = fit_pca_for_period(c_states, n_components)
        c_t_pca[period_idx] = c_reduced
        c_t_explained[period_idx] = c_var
        c_pca_models.append(c_pca)

        print(f"  Period {period:.3f}s: h_t variance explained: {h_var.sum():.3f}, "
              f"c_t variance explained: {c_var.sum():.3f}")

    # Save reduced states
    output_file = output_dir / f'epoch_{epoch}_pca_states.npz'
    np.savez(
        output_file,
        h_t_pca=h_t_pca,  # (n_periods, timesteps, 3)
        c_t_pca=c_t_pca,  # (n_periods, timesteps, 3)
        periods=periods,  # (n_periods,)
        mse_per_period=mse_per_period,  # (n_periods,)
        h_t_explained_variance=h_t_explained,  # (n_periods, 3)
        c_t_explained_variance=c_t_explained,  # (n_periods, 3)
        epoch=epoch,
        n_components=n_components
    )

    # Save PCA models for potential future use
    models_file = output_dir / f'epoch_{epoch}_pca_models.pkl'
    with open(models_file, 'wb') as f:
        pickle.dump({
            'h_pca_models': h_pca_models,
            'c_pca_models': c_pca_models,
            'periods': periods
        }, f)

    print(f"Saved PCA results to {output_file}")

    return {
        'epoch': epoch,
        'n_periods': n_periods,
        'mean_h_variance_explained': h_t_explained.sum(axis=1).mean(),
        'mean_c_variance_explained': c_t_explained.sum(axis=1).mean()
    }


def main():
    """
    Main function to process all checkpoint files.
    """
    parser = argparse.ArgumentParser(description='Preprocess states with PCA')
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--n_components', type=int, default=3,
                        help='Number of PCA components (default: 3)')

    args = parser.parse_args()

    # Setup paths
    experiment_path = Path(args.experiment_path)
    states_dir = experiment_path / 'state_analysis'

    if not states_dir.exists():
        print(f"Error: State analysis directory not found: {states_dir}")
        return

    # Create output directory
    output_dir = states_dir / 'pca_preprocessed'
    output_dir.mkdir(exist_ok=True)

    # Find all state files
    state_files = sorted(states_dir.glob('epoch_*_states.npz'))

    if not state_files:
        print("No state files found")
        return

    print(f"Found {len(state_files)} checkpoint files to process")
    print(f"Using {args.n_components} PCA components\n")

    # Process each checkpoint
    metadata = []
    for state_file in state_files:
        meta = process_checkpoint(state_file, output_dir, args.n_components)
        metadata.append(meta)

    # Save metadata
    metadata_file = output_dir / 'pca_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'n_components': args.n_components,
            'n_checkpoints': len(metadata),
            'checkpoints': metadata
        }, f, indent=2)

    print(f"\nPCA preprocessing complete!")
    print(f"Results saved in {output_dir}")


if __name__ == '__main__':
    main()
