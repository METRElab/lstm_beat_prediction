"""
Preprocess hidden and cell states using PCA for visualization.
Fits a single PCA for all periods within each epoch to create a shared space.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import pickle
from typing import Dict, Tuple


def fit_pca_shared_space(
        all_states: np.ndarray,
        n_components: int = 3
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on concatenated states from all periods to create shared space.

    Args:
        all_states: Concatenated states array of shape (total_timesteps, hidden_size)
                   where total_timesteps = timesteps * n_periods
        n_components: Number of PCA components

    Returns:
        pca_model: Fitted PCA model
        explained_variance: Variance explained by each component
    """
    # Fit PCA on all periods together
    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    return pca, pca.explained_variance_ratio_


def process_checkpoint(
        states_file: Path,
        output_dir: Path,
        n_components: int = 3
) -> Dict:
    """
    Process a single checkpoint file, fitting one PCA for all periods together.

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

    print(f"Processing epoch {epoch}...")
    print(f"  Concatenating {n_periods} periods × {timesteps} timesteps = {n_periods * timesteps} total timesteps")

    # Reshape and concatenate all periods for hidden states
    # From (n_periods, timesteps, 1, hidden_size) to (n_periods * timesteps, hidden_size)
    h_states_all = []
    c_states_all = []

    for period_idx in range(n_periods):
        # Extract states for this period (squeeze out layer dimension)
        h_states = hidden_states[period_idx, :, 0, :]  # (timesteps, hidden_size)
        c_states = cell_states[period_idx, :, 0, :]    # (timesteps, hidden_size)

        h_states_all.append(h_states)
        c_states_all.append(c_states)

    # Concatenate all periods along time axis
    h_states_concatenated = np.vstack(h_states_all)  # (n_periods * timesteps, hidden_size)
    c_states_concatenated = np.vstack(c_states_all)  # (n_periods * timesteps, hidden_size)

    print(f"  Concatenated shape: {h_states_concatenated.shape}")

    # Fit PCA on concatenated data (shared space for all periods)
    h_pca_model, h_explained_var = fit_pca_shared_space(h_states_concatenated, n_components)
    c_pca_model, c_explained_var = fit_pca_shared_space(c_states_concatenated, n_components)

    print(f"  Shared PCA - h_t variance explained: {h_explained_var.sum():.3f}, "
          f"c_t variance explained: {c_explained_var.sum():.3f}")

    # Transform each period using the shared PCA
    h_t_pca = np.zeros((n_periods, timesteps, n_components))
    c_t_pca = np.zeros((n_periods, timesteps, n_components))
    h_t_explained = np.zeros((n_periods, n_components))
    c_t_explained = np.zeros((n_periods, n_components))

    for period_idx in range(n_periods):
        period = periods[period_idx]

        # Extract states for this period
        h_states = hidden_states[period_idx, :, 0, :]  # (timesteps, hidden_size)
        c_states = cell_states[period_idx, :, 0, :]    # (timesteps, hidden_size)

        # Transform using shared PCA
        h_reduced = h_pca_model.transform(h_states)  # (timesteps, n_components)
        c_reduced = c_pca_model.transform(c_states)  # (timesteps, n_components)

        h_t_pca[period_idx] = h_reduced
        c_t_pca[period_idx] = c_reduced

        # Store the shared explained variance for each period (same for all)
        h_t_explained[period_idx] = h_explained_var
        c_t_explained[period_idx] = c_explained_var

        # Calculate per-period variance in the shared space (for information)
        h_var_in_shared = np.var(h_reduced, axis=0)
        c_var_in_shared = np.var(c_reduced, axis=0)

        print(f"    Period {period:.3f}s: h_t variance in shared space: {h_var_in_shared.sum():.3f}, "
              f"c_t variance in shared space: {c_var_in_shared.sum():.3f}")

    # Save reduced states
    output_file = output_dir / f'epoch_{epoch}_pca_states.npz'
    np.savez(
        output_file,
        h_t_pca=h_t_pca,                    # (n_periods, timesteps, 3)
        c_t_pca=c_t_pca,                    # (n_periods, timesteps, 3)
        periods=periods,                     # (n_periods,)
        mse_per_period=mse_per_period,      # (n_periods,)
        h_t_explained_variance=h_t_explained, # (n_periods, 3) - same values repeated
        c_t_explained_variance=c_t_explained, # (n_periods, 3) - same values repeated
        epoch=epoch,
        n_components=n_components,
        pca_type='shared_per_epoch'         # Flag to indicate shared PCA space
    )

    # Save PCA models (one per epoch, not per period)
    models_file = output_dir / f'epoch_{epoch}_pca_models.pkl'
    with open(models_file, 'wb') as f:
        pickle.dump({
            'h_pca_model': h_pca_model,  # Single model for all periods
            'c_pca_model': c_pca_model,  # Single model for all periods
            'periods': periods,
            'explained_variance_h': h_explained_var,
            'explained_variance_c': c_explained_var
        }, f)

    print(f"Saved PCA results to {output_file}")

    return {
        'epoch': int(epoch),
        'n_periods': int(n_periods),
        'shared_h_variance_explained': float(h_explained_var.sum()),
        'shared_c_variance_explained': float(c_explained_var.sum())
    }


def main():
    """
    Main function to process all checkpoint files.
    """
    parser = argparse.ArgumentParser(description='Preprocess states with shared PCA')
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
    print(f"Using {args.n_components} PCA components")
    print(f"PCA mode: Shared space for all periods within each epoch\n")

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
            'pca_type': 'shared_per_epoch',
            'checkpoints': metadata
        }, f, indent=2)

    print(f"\nPCA preprocessing complete!")
    print(f"All periods within each epoch now share the same PCA space")
    print(f"Results saved in {output_dir}")


if __name__ == '__main__':
    main()
