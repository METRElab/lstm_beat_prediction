"""
Save PCA models for each epoch of experiments.
Works with both legacy gaussian_noise and sync_continuation tasks.
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from sklearn.decomposition import PCA
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import BeatPredictionLSTM
from data.generators import create_batch_from_config, generate_test_sequences_from_config


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger for the preprocessing."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_checkpoint_epochs(checkpoint_dir: Path) -> List[int]:
    """Get list of available checkpoint epochs."""
    epochs = []
    for file in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', file.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def generate_test_data_for_pca(
    config: Dict[str, Any],
    n_periods: int,
    trials_per_period: int
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate test data for PCA computation using config-based generation.

    Args:
        config: Experiment configuration
        n_periods: Number of test periods
        trials_per_period: Trials per period

    Returns:
        inputs: (n_trials, timesteps) array
        phase_infos: List of phase info dicts (for sync_continuation) or empty list
    """
    # Create test config
    test_config = config.copy()
    test_config['testing'] = {
        'n_test_periods': n_periods,
        'test_trials_per_period': trials_per_period
    }

    # Use smaller batch for memory efficiency
    test_config['training'] = test_config.get('training', {})
    test_config['training']['batch_size'] = 1

    # Generate test sequences
    test_data = generate_test_sequences_from_config(test_config)

    all_inputs = []
    all_phase_infos = []

    task_type = config['task'].get('task_type', 'legacy')

    for period_data in test_data['sequences']:
        for trial_idx in range(len(period_data['inputs'])):
            all_inputs.append(period_data['inputs'][trial_idx])

            # Get phase info if available
            if task_type == 'sync_continuation' and 'phase_infos' in period_data:
                all_phase_infos.append(period_data['phase_infos'][trial_idx])

    inputs = np.stack(all_inputs)

    return inputs, all_phase_infos


def collect_states_for_epoch(
    model: BeatPredictionLSTM,
    inputs: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on inputs and collect hidden and cell states.

    Args:
        model: LSTM model
        inputs: Input sequences (n_trials, timesteps)
        device: Torch device

    Returns:
        hidden_states: (n_trials * timesteps, hidden_size)
        cell_states: (n_trials * timesteps, hidden_size)
    """
    model.eval()

    all_hidden = []
    all_cell = []

    with torch.no_grad():
        for input_seq in inputs:
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)

            hidden = None
            hidden_states = []
            cell_states = []

            for t in range(input_tensor.shape[1]):
                x_t = input_tensor[:, t:t + 1, :]
                output_t, hidden = model.forward(x_t, hidden)

                h_t, c_t = hidden
                hidden_states.append(h_t.squeeze().cpu().numpy())
                cell_states.append(c_t.squeeze().cpu().numpy())

            all_hidden.extend(hidden_states)
            all_cell.extend(cell_states)

    hidden_array = np.array(all_hidden)
    cell_array = np.array(all_cell)

    return hidden_array, cell_array


def process_experiment(
    exp_path: Path,
    analysis_config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Process all epochs for a single experiment.

    Args:
        exp_path: Path to experiment directory
        analysis_config: Analysis configuration
        logger: Logger instance
    """
    logger.info(f"Processing experiment: {exp_path.name}")

    # Load experiment config
    config_path = exp_path / 'config.json'
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    task_type = config['task'].get('task_type', 'legacy')
    logger.info(f"Task type: {task_type}")

    # Get available checkpoints
    checkpoint_dir = exp_path / 'checkpoints'
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoint_dir}")
        return

    epochs = get_checkpoint_epochs(checkpoint_dir)
    logger.info(f"Found {len(epochs)} checkpoints")

    # Create PCA models directory
    pca_dir = exp_path / 'pca_models'
    pca_dir.mkdir(exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = BeatPredictionLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model'].get('dropout', 0.0)
    ).to(device)

    # Generate test data
    logger.info("Generating test data...")
    n_periods = analysis_config['analysis']['n_test_periods']
    trials_per_period = analysis_config['analysis']['trials_per_tempo']

    inputs, phase_infos = generate_test_data_for_pca(config, n_periods, trials_per_period)
    logger.info(f"Generated {inputs.shape[0]} sequences of length {inputs.shape[1]}")

    # Process each epoch
    for epoch in epochs:
        logger.info(f"Processing epoch {epoch}")

        # Load checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Collect states
        logger.info("  Collecting states...")
        hidden_states, cell_states = collect_states_for_epoch(model, inputs, device)
        logger.info(f"  Collected states shape: {hidden_states.shape}")

        # Fit PCA for hidden states
        logger.info("  Fitting PCA for hidden states...")
        pca_h = PCA(n_components=analysis_config['analysis']['n_components'])
        pca_h.fit(hidden_states)
        variance_h = pca_h.explained_variance_ratio_.sum()
        logger.info(f"  Hidden PCA variance explained: {variance_h:.3f}")

        # Fit PCA for cell states
        logger.info("  Fitting PCA for cell states...")
        pca_c = PCA(n_components=analysis_config['analysis']['n_components'])
        pca_c.fit(cell_states)
        variance_c = pca_c.explained_variance_ratio_.sum()
        logger.info(f"  Cell PCA variance explained: {variance_c:.3f}")

        # Save PCA models
        pca_h_path = pca_dir / f'epoch_{epoch}_pca_h.pkl'
        with open(pca_h_path, 'wb') as f:
            pickle.dump(pca_h, f)

        pca_c_path = pca_dir / f'epoch_{epoch}_pca_c.pkl'
        with open(pca_c_path, 'wb') as f:
            pickle.dump(pca_c, f)

        logger.info(f"  Saved PCA models for epoch {epoch}")

    # Save metadata
    metadata = {
        'task_type': task_type,
        'epochs_processed': epochs,
        'n_components': analysis_config['analysis']['n_components'],
        'trials_per_tempo': analysis_config['analysis']['trials_per_tempo'],
        'n_test_periods': analysis_config['analysis']['n_test_periods']
    }

    metadata_path = pca_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Completed processing {exp_path.name}")


def main():
    """Main function to process experiments."""
    parser = argparse.ArgumentParser(description='Save PCA models for experiments')
    parser.add_argument('--experiment_path', type=str, default=None,
                        help='Path to specific experiment directory')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory containing multiple experiments')
    parser.add_argument('--analysis_config', type=str, default='config/pca_analysis_config.json',
                        help='Path to analysis configuration')

    args = parser.parse_args()

    logger = setup_logger('pca_preprocessing')

    # Load analysis config
    analysis_config_path = Path(args.analysis_config)
    if analysis_config_path.exists():
        with open(analysis_config_path, 'r') as f:
            analysis_config = json.load(f)
    else:
        # Default config
        analysis_config = {
            'analysis': {
                'n_test_periods': 9,
                'trials_per_tempo': 3,
                'n_components': 3
            }
        }
        logger.warning(f"Analysis config not found, using defaults: {analysis_config}")

    logger.info(f"Analysis configuration: {analysis_config}")

    # Determine which experiments to process
    if args.experiment_path:
        exp_paths = [Path(args.experiment_path)]
    elif args.base_dir:
        base_dir = Path(args.base_dir)
        exp_paths = sorted([d for d in base_dir.iterdir() if d.is_dir() and (d / 'config.json').exists()])
    else:
        logger.error("Must specify either --experiment_path or --base_dir")
        return

    logger.info(f"Found {len(exp_paths)} experiments to process")

    for exp_path in exp_paths:
        try:
            process_experiment(exp_path, analysis_config, logger)
        except Exception as e:
            logger.error(f"Failed to process {exp_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("PCA preprocessing complete!")


if __name__ == '__main__':
    main()
