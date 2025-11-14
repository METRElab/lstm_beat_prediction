"""
Model runner utility for loading and running LSTM models with state collection.
"""

from pathlib import Path
import json
import pickle
import re
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import BeatPredictionLSTM


class ModelRunner:
    """Handles model loading, execution, and PCA transformation."""

    def __init__(self, experiment_path: Path):
        """
        Initialize model runner with experiment path.

        Args:
            experiment_path: Path to experiment directory
        """
        self.experiment_path = Path(experiment_path)
        self.load_config()
        self.setup_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.current_epoch = None

    def load_config(self) -> None:
        """Load experiment configuration."""
        config_path = self.experiment_path / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def setup_model(self) -> None:
        """Initialize model architecture."""
        model_config = self.config['model']
        self.model_arch = BeatPredictionLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout=model_config.get('dropout', 0.0)
        )
        self.hidden_size = model_config['hidden_size']
        self.num_layers = model_config['num_layers']

    def get_available_epochs(self) -> List[int]:
        """
        Get list of available checkpoint epochs.

        Returns:
            Sorted list of epoch numbers
        """
        checkpoint_dir = self.experiment_path / 'checkpoints'
        if not checkpoint_dir.exists():
            return []

        epochs = []
        for file in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', file.name)
            if match:
                epochs.append(int(match.group(1)))

        return sorted(epochs)

    def load_checkpoint(self, epoch: int) -> None:
        """
        Load model checkpoint for specific epoch.

        Args:
            epoch: Epoch number to load
        """
        checkpoint_path = self.experiment_path / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create fresh model and load weights
        self.model = BeatPredictionLSTM(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            output_size=self.config['model']['output_size'],
            dropout=self.config['model'].get('dropout', 0.0)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.current_epoch = epoch

    def load_pca_models(self, epoch: int) -> Tuple[Any, Any]:
        """
        Load PCA models for specific epoch.

        Args:
            epoch: Epoch number

        Returns:
            (pca_h, pca_c) - PCA models for hidden and cell states
        """
        pca_dir = self.experiment_path / 'pca_models'

        pca_h_path = pca_dir / f'epoch_{epoch}_pca_h.pkl'
        pca_c_path = pca_dir / f'epoch_{epoch}_pca_c.pkl'

        if not pca_h_path.exists() or not pca_c_path.exists():
            raise FileNotFoundError(f"PCA models not found for epoch {epoch}")

        with open(pca_h_path, 'rb') as f:
            pca_h = pickle.load(f)

        with open(pca_c_path, 'rb') as f:
            pca_c = pickle.load(f)

        return pca_h, pca_c

    def generate_input_from_times(
            self,
            pulse_times: List[float],
            sequence_length: float,
            dt: float,
            pulse_width: float,
            pulse_height: float
    ) -> np.ndarray:
        """
        Generate input tensor from pulse time array.

        Args:
            pulse_times: List of pulse onset times in seconds
            sequence_length: Total sequence length in seconds
            dt: Time step
            pulse_width: Width of each pulse
            pulse_height: Height of each pulse

        Returns:
            Input sequence array
        """
        n_timesteps = int(sequence_length / dt)
        input_seq = np.zeros(n_timesteps)

        pulse_samples = int(pulse_width / dt)

        for pulse_time in pulse_times:
            pulse_start_idx = int(pulse_time / dt)
            pulse_end_idx = pulse_start_idx + pulse_samples

            # Add pulse if within bounds
            if pulse_start_idx >= 0 and pulse_end_idx <= n_timesteps:
                input_seq[pulse_start_idx:pulse_end_idx] = pulse_height
            elif pulse_start_idx < n_timesteps and pulse_start_idx >= 0:
                # Partial pulse at the end
                input_seq[pulse_start_idx:min(pulse_end_idx, n_timesteps)] = pulse_height

        return input_seq

    def run_with_states(
            self,
            input_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model step-by-step and collect states.

        Args:
            input_tensor: Input sequence (timesteps,)

        Returns:
            output: Model output (timesteps,)
            hidden_states: Hidden states (timesteps, hidden_size)
            cell_states: Cell states (timesteps, hidden_size)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint first.")

        # Prepare input
        input_torch = torch.FloatTensor(input_tensor).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Run step by step
        outputs = []
        hidden_list = []
        cell_list = []

        hidden = None

        with torch.no_grad():
            for t in range(input_torch.shape[1]):
                x_t = input_torch[:, t:t + 1, :]
                output_t, hidden = self.model.forward(x_t, hidden)

                h_t, c_t = hidden

                outputs.append(output_t.squeeze().cpu().numpy())
                # For single layer, squeeze the layer dimension
                hidden_list.append(h_t.squeeze(0).squeeze(0).cpu().numpy())
                cell_list.append(c_t.squeeze(0).squeeze(0).cpu().numpy())

        output = np.array(outputs)
        hidden_states = np.array(hidden_list)
        cell_states = np.array(cell_list)

        return output, hidden_states, cell_states

    def transform_states(
            self,
            states: np.ndarray,
            pca_model: Any
    ) -> np.ndarray:
        """
        Apply PCA transformation to states.

        Args:
            states: States array (timesteps, hidden_size)
            pca_model: Fitted PCA model

        Returns:
            Transformed states (timesteps, 3)
        """
        return pca_model.transform(states)
