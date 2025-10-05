"""
Main entry point for LSTM beat prediction training and testing.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np
import torch

from models import BeatPredictionLSTM
from training import Trainer
from data import generate_test_sequences
from utils import setup_logger
from utils import plot_predictions, plot_accuracy_vs_period, create_test_report


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_model(config: Dict[str, Any]) -> BeatPredictionLSTM:
    """
    Create LSTM model from config.

    Args:
        config: Configuration dictionary

    Returns:
        LSTM model instance
    """
    model = BeatPredictionLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout']
    )
    return model


def train(config_path: str) -> None:
    """
    Run training with given configuration.

    Args:
        config_path: Path to configuration file
    """
    # Load config
    config = load_config(config_path)

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_path = Path(config['paths']['experiment_dir']) / timestamp
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Update config with experiment path
    config['paths']['experiment_path'] = str(experiment_path)

    # Save updated config in experiment folder
    save_config(config, experiment_path / 'config.json')

    # Setup logging
    logger = setup_logger(
        name='training',
        log_file=str(experiment_path / 'logs' / 'training.log')
    )

    logger.info(f"Starting experiment: {experiment_path}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = Trainer(config, model, str(experiment_path), logger)

    # Train model
    history = trainer.train()

    # Save training history
    history_path = experiment_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Results saved to: {experiment_path}")


def test(config_path: str, checkpoint_name: str = 'best_model.pth') -> None:
    """
    Test a trained model.

    Args:
        config_path: Path to configuration file (from trained experiment)
        checkpoint_name: Name of checkpoint to load
    """
    # Load config
    config = load_config(config_path)

    # Get experiment path from config
    experiment_path = Path(config['paths']['experiment_path'])
    if not experiment_path.exists():
        print(f"Error: Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    # Setup logging (append to existing log)
    logger = setup_logger(
        name='testing',
        log_file=str(experiment_path / 'logs' / 'testing.log')
    )

    logger.info(f"Testing model from: {experiment_path}")

    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Create model
    model = create_model(config)

    # Load checkpoint
    checkpoint_path = experiment_path / 'checkpoints' / checkpoint_name
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Generate test data
    logger.info("Generating test sequences...")
    test_data = generate_test_sequences(
        n_periods=config['testing']['n_test_periods'],
        min_period=config['task']['min_period'],
        max_period=config['task']['max_period'],
        trials_per_period=config['testing']['test_trials_per_period'],
        n_pulses=config['task']['n_pulses'],
        pulse_width=config['task']['pulse_width'],
        pulse_height=config['task']['pulse_height'],
        dt=config['task']['dt'],
        sequence_length=config['task']['sequence_length'],
        baseline_value=config['task']['baseline_value']
    )

    # Test model
    logger.info("Testing model...")
    mse_per_period = []
    all_mse = []

    figures_dir = experiment_path / 'figures'
    figures_dir.mkdir(exist_ok=True)

    for i, period_data in enumerate(test_data['sequences']):
        period = period_data['period']
        inputs = period_data['inputs']  # (trials, timesteps)
        targets = period_data['targets']

        # Convert to tensors and add channel dimension
        inputs_tensor = torch.FloatTensor(inputs).unsqueeze(-1).to(device)
        targets_tensor = torch.FloatTensor(targets).unsqueeze(-1).to(device)

        # Get predictions
        with torch.no_grad():
            predictions, _ = model(inputs_tensor)

        # Calculate MSE
        mse = torch.mean((predictions - targets_tensor) ** 2).item()
        mse_per_period.append(mse)
        all_mse.append(mse)

        # Plot first trial for each period
        pred_numpy = predictions[0, :, 0].cpu().numpy()
        plot_predictions(
            input_sequence=inputs[0],
            target_sequence=targets[0],
            predicted_sequence=pred_numpy,
            period=period,
            dt=config['task']['dt'],
            save_path=str(figures_dir / f'test_period_{period:.2f}.png'),
            show=False
        )

        logger.info(f"Period {period:.3f}s: MSE = {mse:.6f}")

    # Calculate overall statistics
    test_results = {
        'periods': test_data['periods'],
        'mse_per_period': mse_per_period,
        'mean_mse': float(np.mean(all_mse)),
        'std_mse': float(np.std(all_mse)),
        'min_mse': float(np.min(all_mse)),
        'max_mse': float(np.max(all_mse))
    }

    # Plot accuracy vs period
    plot_accuracy_vs_period(
        periods=test_data['periods'],
        mse_values=mse_per_period,
        save_path=str(figures_dir / 'accuracy_vs_period.png'),
        show=False
    )

    # Save test results
    results_path = experiment_path / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    # Create test report
    create_test_report(
        test_results=test_results,
        save_path=str(experiment_path / 'test_report.txt')
    )

    logger.info(f"Testing complete. Results saved to: {experiment_path}")
    logger.info(f"Overall MSE: {test_results['mean_mse']:.6f} ± {test_results['std_mse']:.6f}")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description='LSTM Beat Prediction')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Checkpoint name for testing (default: best_model.pth)')

    args = parser.parse_args()

    if args.train and args.test:
        print("Error: Cannot specify both --train and --test")
        sys.exit(1)

    if not args.train and not args.test:
        print("Error: Must specify either --train or --test")
        sys.exit(1)

    if args.train:
        train(args.config)
    elif args.test:
        test(args.config, args.checkpoint)


if __name__ == '__main__':
    main()