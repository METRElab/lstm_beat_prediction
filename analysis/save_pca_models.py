"""
Save PCA models for each epoch of experiments with gaussian noise.
This script computes PCA on concatenated states from multiple trials
and saves the PCA models for later use in interactive visualization.
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

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import BeatPredictionLSTM
from data.generators import generate_beat_sequence_gaussian_with_noise


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger for the preprocessing."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_experiment_name(folder_name: str) -> Tuple[float, float]:
    """
    Parse phase and jitter noise levels from experiment folder name.

    Args:
        folder_name: Name like '20251109_160331_p0.005_j0.01'

    Returns:
        (phase_noise, jitter_noise) tuple
    """
    match = re.search(r'p([\d.]+)_j([\d.]+)', folder_name)
    if match:
        phase = float(match.group(1))
        jitter = float(match.group(2))
        return phase, jitter
    return None, None


def get_checkpoint_epochs(checkpoint_dir: Path) -> List[int]:
    """
    Get list of available checkpoint epochs.

    Args:
        checkpoint_dir: Path to checkpoints directory

    Returns:
        Sorted list of epoch numbers
    """
    epochs = []
    for file in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', file.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def generate_test_data_for_pca(
        config: Dict[str, Any],
        analysis_config: Dict[str, Any],
        phase_noise: float,
        jitter_noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple trials of test data for PCA computation.

    Args:
        config: Experiment configuration
        analysis_config: Analysis configuration
        phase_noise: Phase noise level for this experiment
        jitter_noise: Jitter noise level for this experiment

    Returns:
        inputs: (n_total_timesteps, 1) concatenated inputs
        targets: (n_total_timesteps, 1) concatenated targets
    """
    task = config['task']
    n_periods = analysis_config['analysis']['n_test_periods']
    trials_per_tempo = analysis_config['analysis']['trials_per_tempo']

    # Generate test periods
    test_periods = np.linspace(task['min_period'], task['max_period'], n_periods)

    all_inputs = []
    all_targets = []

    for period in test_periods:
        for _ in range(trials_per_tempo):
            # Generate sequence with noise
            input_seq, target_seq, _ = generate_beat_sequence_gaussian_with_noise(
                period=period,
                phase_noise_std=phase_noise,
                jitter_std=jitter_noise,
                min_n_pulses=task['noise_params']['min_n_pulses'],
                max_n_pulses=task['noise_params']['max_n_pulses'],
                pulse_width=task['pulse_width'],
                pulse_height=task['pulse_height'],
                dt=task['dt'],
                sequence_length=task['sequence_length'],
                baseline_value=task['baseline_value'],
                output_offset=task.get('output_offset', -0.5),
                gaussian_length=task.get('gaussian_params', {}).get('gaussian_length'),
                gaussian_sigma=task.get('gaussian_params', {}).get('gaussian_sigma'),
                gaussian_max_height=task.get('gaussian_params', {}).get('gaussian_max_height'),
                skip_first_n=task.get('skip_first_n', 2)
            )

            all_inputs.append(input_seq)
            all_targets.append(target_seq)

    # Stack all sequences
    inputs = np.stack(all_inputs)  # (n_trials, timesteps)
    targets = np.stack(all_targets)

    return inputs, targets


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
            # Prepare input tensor (1, timesteps, 1)
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)

            # Process sequence step by step
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

    # Convert to arrays
    hidden_array = np.array(all_hidden)  # (n_trials * timesteps, hidden_size)
    cell_array = np.array(all_cell)

    return hidden_array, cell_array


def process_single_experiment(
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

    # Parse noise levels from folder name
    phase_noise, jitter_noise = parse_experiment_name(exp_path.name)
    if phase_noise is None:
        logger.warning(f"Could not parse noise levels from {exp_path.name}")
        return

    logger.info(f"  Phase noise: {phase_noise}, Jitter noise: {jitter_noise}")

    # Load experiment config
    config_path = exp_path / 'config.json'
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get available checkpoints
    checkpoint_dir = exp_path / 'checkpoints'
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoint_dir}")
        return

    epochs = get_checkpoint_epochs(checkpoint_dir)
    logger.info(f"  Found {len(epochs)} checkpoints")

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
        dropout=config['model']['dropout']
    ).to(device)

    # Generate test data once (same data for all epochs)
    logger.info("  Generating test data...")
    inputs, _ = generate_test_data_for_pca(config, analysis_config, phase_noise, jitter_noise)
    logger.info(f"    Generated {inputs.shape[0]} sequences of length {inputs.shape[1]}")

    # Process each epoch
    for epoch in epochs:
        logger.info(f"  Processing epoch {epoch}")

        # Load checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Collect states
        logger.info("    Collecting states...")
        hidden_states, cell_states = collect_states_for_epoch(model, inputs, device)
        logger.info(f"    Collected states shape: {hidden_states.shape}")

        # Fit PCA for hidden states
        logger.info("    Fitting PCA for hidden states...")
        pca_h = PCA(n_components=analysis_config['analysis']['n_components'])
        pca_h.fit(hidden_states)
        variance_h = pca_h.explained_variance_ratio_.sum()
        logger.info(f"    Hidden PCA variance explained: {variance_h:.3f}")

        # Fit PCA for cell states
        logger.info("    Fitting PCA for cell states...")
        pca_c = PCA(n_components=analysis_config['analysis']['n_components'])
        pca_c.fit(cell_states)
        variance_c = pca_c.explained_variance_ratio_.sum()
        logger.info(f"    Cell PCA variance explained: {variance_c:.3f}")

        # Save PCA models
        pca_h_path = pca_dir / f'epoch_{epoch}_pca_h.pkl'
        with open(pca_h_path, 'wb') as f:
            pickle.dump(pca_h, f)

        pca_c_path = pca_dir / f'epoch_{epoch}_pca_c.pkl'
        with open(pca_c_path, 'wb') as f:
            pickle.dump(pca_c, f)

        logger.info(f"    Saved PCA models for epoch {epoch}")

    # Save metadata
    metadata = {
        'phase_noise': phase_noise,
        'jitter_noise': jitter_noise,
        'epochs_processed': epochs,
        'n_components': analysis_config['analysis']['n_components'],
        'trials_per_tempo': analysis_config['analysis']['trials_per_tempo'],
        'n_test_periods': analysis_config['analysis']['n_test_periods']
    }

    metadata_path = pca_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Completed processing {exp_path.name}")


def main():
    """Main function to process all experiments."""
    parser = argparse.ArgumentParser(description='Save PCA models for gaussian noise experiments')
    parser.add_argument('--base_dir', type=str, default='experiments/gaussian_noise',
                        help='Base directory containing all experiments')
    parser.add_argument('--analysis_config', type=str, default='config/pca_analysis_config.json',
                        help='Path to analysis configuration')
    parser.add_argument('--specific_experiment', type=str, default=None,
                        help='Process only this specific experiment (folder name)')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('pca_preprocessing')

    # Load analysis config
    with open(args.analysis_config, 'r') as f:
        analysis_config = json.load(f)

    logger.info(f"Analysis configuration:")
    logger.info(f"  Trials per tempo: {analysis_config['analysis']['trials_per_tempo']}")
    logger.info(f"  Number of test periods: {analysis_config['analysis']['n_test_periods']}")
    logger.info(f"  PCA components: {analysis_config['analysis']['n_components']}")

    # Get list of experiments to process
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return

    if args.specific_experiment:
        exp_dirs = [base_dir / args.specific_experiment]
        if not exp_dirs[0].exists():
            logger.error(f"Specific experiment not found: {exp_dirs[0]}")
            return
    else:
        # Find all experiment directories with pattern
        exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and '_p' in d.name and '_j' in d.name])

    logger.info(f"Found {len(exp_dirs)} experiments to process")

    # Process each experiment
    for exp_dir in exp_dirs:
        try:
            process_single_experiment(exp_dir, analysis_config, logger)
        except Exception as e:
            logger.error(f"Failed to process {exp_dir.name}: {str(e)}")
            continue

    logger.info("PCA preprocessing complete!")


if __name__ == '__main__':
    main()
