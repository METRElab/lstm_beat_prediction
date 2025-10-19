"""
State analyzer for post-training analysis of LSTM hidden and cell states.
"""

from typing import Dict, Any, List, Tuple
import json
import logging
from pathlib import Path
import numpy as np
import torch
import re

from models.lstm_model import BeatPredictionLSTM
from data.generators import generate_test_sequences_from_config


class StateAnalyzer:
    """
    Analyzes hidden and cell states across different checkpoints.
    """

    def __init__(
        self,
        experiment_path: str,
        analysis_config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Initialize state analyzer.

        Args:
            experiment_path: Path to experiment directory
            analysis_config: Analysis-specific configuration
            logger: Logger instance
        """
        self.experiment_path = Path(experiment_path)
        self.analysis_config = analysis_config
        self.logger = logger

        # Load experiment config
        config_path = self.experiment_path / 'config.json'
        with open(config_path, 'r') as f:
            self.experiment_config = json.load(f)

        # Override ITI settings for analysis
        self.min_iti = analysis_config['analysis']['min_iti']
        self.mean_iti = self.min_iti  # Set equal for consistent padding

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = self.experiment_path / 'state_analysis'
        self.output_dir.mkdir(exist_ok=True)

    def get_checkpoint_files(self) -> List[Tuple[int, Path]]:
        """
        Get all checkpoint files sorted by epoch.

        Returns:
            List of (epoch, checkpoint_path) tuples
        """
        checkpoint_dir = self.experiment_path / 'checkpoints'
        checkpoint_files = []

        for file in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
            # Extract epoch number from filename
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', file.name)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, file))

        # Sort by epoch
        checkpoint_files.sort(key=lambda x: x[0])

        self.logger.info(f"Found {len(checkpoint_files)} checkpoints")
        return checkpoint_files

    def create_model(self) -> BeatPredictionLSTM:
        """
        Create model instance from experiment config.

        Returns:
            LSTM model
        """
        config = self.experiment_config
        model = BeatPredictionLSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            output_size=config['model']['output_size'],
            dropout=config['model']['dropout']
        )
        return model

    def generate_test_periods(self) -> np.ndarray:
        """
        Generate evenly spaced test periods.

        Returns:
            Array of test periods
        """
        n_periods = self.analysis_config['analysis']['n_test_periods']
        min_period = self.experiment_config['task']['min_period']
        max_period = self.experiment_config['task']['max_period']

        return np.linspace(min_period, max_period, n_periods)

    def run_sequence_step_by_step(
        self,
        model: BeatPredictionLSTM,
        input_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray]]:
        """
        Run LSTM step by step to collect hidden and cell states.

        Args:
            model: LSTM model
            input_sequence: Input tensor (1, timesteps, 1)

        Returns:
            outputs: Model predictions (1, timesteps, 1)
            all_h_t: List of hidden states at each timestep
                    Each element shape: (num_layers, hidden_size)
            all_c_t: List of cell states at each timestep
                    Each element shape: (num_layers, hidden_size)
        """
        sequence_length = input_sequence.shape[1]
        outputs = []
        all_h_t = []
        all_c_t = []

        # Initialize with None - will be initialized in forward function
        hidden = None

        # Process sequence step by step
        for t in range(sequence_length):
            # Get input at time t (shape: 1, 1, 1)
            x_t = input_sequence[:, t:t+1, :]

            # Forward through model
            output_t, hidden = model.forward(x_t, hidden)

            # Extract h_t and c_t from hidden tuple
            h_t, c_t = hidden

            # Store outputs and states
            outputs.append(output_t)
            all_h_t.append(h_t.squeeze(1).cpu().numpy())  # Remove batch dim
            all_c_t.append(c_t.squeeze(1).cpu().numpy())  # Remove batch dim

        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)

        return outputs, all_h_t, all_c_t

    def analyze_all_checkpoints(self) -> None:
        """
        Analyze all checkpoints and save results.

        Saved data format in .npz files:
        - hidden_states: Array of shape (n_periods, timesteps, num_layers, hidden_size)
                        Hidden states for each period, timestep, and layer
        - cell_states: Array of shape (n_periods, timesteps, num_layers, hidden_size)
                      Cell states for each period, timestep, and layer
        - network_outputs: Array of shape (n_periods, timesteps)
                          Network output predictions for each period and timestep
        - periods: Array of shape (n_periods,)
                  The period value (in seconds) for each sequence
        - mse_per_period: Array of shape (n_periods,)
                         MSE loss for each period
        - epoch: Scalar, the epoch number of this checkpoint

        The first dimension corresponds to different test periods, allowing
        analysis of how states and outputs differ across different beat frequencies.
        """
        # Get all checkpoints
        checkpoint_files = self.get_checkpoint_files()
        if not checkpoint_files:
            self.logger.error("No checkpoints found")
            return

        # Create model
        model = self.create_model()
        model.to(self.device)

        # Generate test periods
        test_periods = self.generate_test_periods()

        # Prepare config for test data generation
        test_config = self.experiment_config.copy()

        # Override with analysis settings
        test_config['testing'] = {
            'n_test_periods': self.analysis_config['analysis']['n_test_periods'],
            'test_trials_per_period': self.analysis_config['analysis']['trials_per_period']
        }

        # Override ITI to be fixed (mean_iti = min_iti)
        test_config['task']['min_iti'] = self.min_iti
        test_config['task']['mean_iti'] = self.min_iti

        # Generate test sequences using the config-based generator
        self.logger.info("Generating test sequences...")
        test_data = generate_test_sequences_from_config(test_config)

        # Store accuracy progression
        accuracy_progression = {}

        # Process each checkpoint
        for epoch, checkpoint_path in checkpoint_files:
            self.logger.info(f"Analyzing checkpoint epoch {epoch}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Store states and MSE for this checkpoint
            hidden_states = []  # Will store states for each period
            cell_states = []    # Will store states for each period
            network_outputs = []  # Will store network outputs for each period
            mse_per_period = []

            # Test each period
            for i, period_data in enumerate(test_data['sequences']):
                period = period_data['period']
                # Get first trial (since trials_per_period = 1)
                input_seq = period_data['inputs'][0]  # Shape: (timesteps,)
                target_seq = period_data['targets'][0]  # Shape: (timesteps,)

                # Convert to tensors (batch_size=1)
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                target_tensor = torch.FloatTensor(target_seq).unsqueeze(0).unsqueeze(-1).to(self.device)

                # Run step by step
                with torch.no_grad():
                    outputs, h_t_list, c_t_list = self.run_sequence_step_by_step(model, input_tensor)

                # Calculate MSE
                mse = torch.mean((outputs - target_tensor) ** 2).item()
                mse_per_period.append(mse)

                # Stack states across time
                # h_t_list is a list of timesteps, each element has shape (num_layers, hidden_size)
                # After stacking: (timesteps, num_layers, hidden_size)
                h_states = np.stack(h_t_list, axis=0)
                c_states = np.stack(c_t_list, axis=0)

                # Save network output (squeeze to remove batch and feature dims)
                output_array = outputs.squeeze().cpu().numpy()  # Shape: (timesteps,)

                hidden_states.append(h_states)
                cell_states.append(c_states)
                network_outputs.append(output_array)

            # Convert lists to arrays
            # Shape: (n_periods, timesteps, num_layers, hidden_size)
            hidden_states = np.array(hidden_states)
            cell_states = np.array(cell_states)
            # Shape: (n_periods, timesteps)
            network_outputs = np.array(network_outputs)

            # Save states for this checkpoint with detailed data
            # Each saved file contains:
            # - States for all test periods
            # - Network outputs for all test periods
            # - The period value for each test sequence
            # - MSE performance for each period
            save_path = self.output_dir / f'epoch_{epoch}_states.npz'
            np.savez(
                save_path,
                hidden_states=hidden_states,  # Shape: (n_periods, timesteps, num_layers, hidden_size)
                cell_states=cell_states,      # Shape: (n_periods, timesteps, num_layers, hidden_size)
                network_outputs=network_outputs,  # Shape: (n_periods, timesteps)
                periods=test_data['periods'],  # Shape: (n_periods,) - period values in seconds
                mse_per_period=mse_per_period, # Shape: (n_periods,) - MSE for each period
                epoch=epoch                    # Scalar - epoch number
            )
            self.logger.info(f"Saved states and outputs to {save_path}")

            # Store accuracy progression with periods
            accuracy_progression[str(epoch)] = {
                'periods': test_data['periods'],
                'mse_per_period': mse_per_period
            }

            # Log MSE info
            mean_mse = np.mean(mse_per_period)
            self.logger.info(f"Epoch {epoch}: Mean MSE = {mean_mse:.6f}")

        # Save accuracy progression
        accuracy_path = self.output_dir / 'accuracy_progression.json'
        with open(accuracy_path, 'w') as f:
            json.dump(accuracy_progression, f, indent=2)
        self.logger.info(f"Saved accuracy progression to {accuracy_path}")

        # Save analysis metadata
        metadata = {
            'experiment_path': str(self.experiment_path),
            'n_checkpoints_analyzed': len(checkpoint_files),
            'epochs_analyzed': [epoch for epoch, _ in checkpoint_files],
            'test_periods': test_data['periods'],
            'analysis_config': self.analysis_config,
            'target_type': test_config['task'].get('target_type', 'rectangle')
        }

        metadata_path = self.output_dir / 'analysis_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Analysis complete. Results saved to {self.output_dir}")
