"""
State analyzer for post-training analysis of LSTM hidden and cell states.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
import logging
from pathlib import Path
import numpy as np
import torch
import re

from models.lstm_model import BeatPredictionLSTM
from models.feedback_lstm import FeedbackLSTM
from models.feedback import create_feedback_components
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

        # Check task type
        self.task_type = self.experiment_config['task'].get('task_type', 'legacy')
        self.logger.info(f"Task type: {self.task_type}")

        # Check feedback configuration
        self.feedback_config = self.experiment_config.get('feedback', {})
        self.feedback_enabled = self.feedback_config.get('enabled', False)
        if self.feedback_enabled:
            self.logger.info(f"Feedback enabled: threshold={self.feedback_config.get('threshold')}, delay={self.feedback_config.get('delay')}s")

        # Override ITI settings for analysis (only for legacy tasks)
        if self.task_type != 'sync_continuation':
            self.min_iti = analysis_config['analysis'].get('min_iti', 2.0)
            self.mean_iti = self.min_iti

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
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', file.name)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, file))

        checkpoint_files.sort(key=lambda x: x[0])

        self.logger.info(f"Found {len(checkpoint_files)} checkpoints")
        return checkpoint_files

    def create_model(self):
        """
        Create model instance from experiment config.

        Returns FeedbackLSTM when feedback is enabled, otherwise BeatPredictionLSTM.

        Returns:
            LSTM model (BeatPredictionLSTM or FeedbackLSTM)
        """
        config = self.experiment_config

        if self.feedback_enabled:
            model = FeedbackLSTM(
                input_size=config['model']['input_size'],
                hidden_size=config['model']['hidden_size'],
                num_layers=config['model']['num_layers'],
                output_size=config['model']['output_size'],
                dropout=config['model']['dropout'],
                feedback_config=self.feedback_config
            )
        else:
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
            all_c_t: List of cell states at each timestep
        """
        sequence_length = input_sequence.shape[1]
        outputs = []
        all_h_t = []
        all_c_t = []

        hidden = None

        for t in range(sequence_length):
            x_t = input_sequence[:, t:t+1, :]
            output_t, hidden = model.forward(x_t, hidden)

            h_t, c_t = hidden

            outputs.append(output_t)
            all_h_t.append(h_t.squeeze(1).cpu().numpy())
            all_c_t.append(c_t.squeeze(1).cpu().numpy())

        outputs = torch.cat(outputs, dim=1)

        return outputs, all_h_t, all_c_t

    def run_sequence_with_feedback(
        self,
        model: FeedbackLSTM,
        input_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Run LSTM step by step with feedback loop to collect hidden/cell states and feedback signal.

        Args:
            model: FeedbackLSTM model
            input_sequence: Input tensor (1, timesteps, 1)

        Returns:
            outputs: Model predictions (1, timesteps, 1)
            all_h_t: List of hidden states at each timestep
            all_c_t: List of cell states at each timestep
            feedback_signal: Feedback signal array (timesteps,)
        """
        dt = self.experiment_config['task']['dt']
        sequence_length = input_sequence.shape[1]
        batch_size = 1

        # Initialize feedback components
        buffer, pulse_gen = create_feedback_components(
            config=self.feedback_config,
            dt=dt,
            batch_size=batch_size,
            device=self.device
        )

        outputs = []
        all_h_t = []
        all_c_t = []
        feedback_list = []

        hidden = None

        for t in range(sequence_length):
            x_t = input_sequence[:, t:t+1, :]

            # Get delayed feedback from buffer
            if t == 0:
                delayed_feedback = torch.zeros(batch_size, 1, device=self.device)
            else:
                delayed_feedback = buffer.buffer[0]

            # Combine input with feedback (additive)
            combined_input = x_t + delayed_feedback.unsqueeze(1)

            # Forward through model's base lstm
            if hasattr(model, 'lstm'):
                output_t, hidden = model.lstm(combined_input, hidden)
            else:
                output_t, hidden = model.forward(combined_input, hidden)

            h_t, c_t = hidden

            outputs.append(output_t)
            all_h_t.append(h_t.squeeze(1).cpu().numpy())
            all_c_t.append(c_t.squeeze(1).cpu().numpy())

            # Check threshold and generate feedback pulse
            feedback_pulse = pulse_gen.step(output_t.squeeze(1))
            buffer.push(feedback_pulse)
            feedback_list.append(feedback_pulse.squeeze().cpu().numpy())

        outputs = torch.cat(outputs, dim=1)
        feedback_signal = np.array(feedback_list)

        return outputs, all_h_t, all_c_t, feedback_signal

    def analyze_all_checkpoints(self) -> None:
        """
        Analyze all checkpoints and save results.
        """
        checkpoint_files = self.get_checkpoint_files()
        if not checkpoint_files:
            self.logger.error("No checkpoints found")
            return

        model = self.create_model()
        model.to(self.device)

        # Prepare config for test data generation
        test_config = self.experiment_config.copy()

        # Override with analysis settings
        test_config['testing'] = {
            'n_test_periods': self.analysis_config['analysis']['n_test_periods'],
            'test_trials_per_period': self.analysis_config['analysis'].get('trials_per_period', 1)
        }

        # For legacy tasks, override ITI
        if self.task_type != 'sync_continuation':
            test_config['task']['min_iti'] = self.min_iti
            test_config['task']['mean_iti'] = self.min_iti

        # Generate test sequences
        self.logger.info("Generating test sequences...")
        test_data = generate_test_sequences_from_config(test_config)

        # Check if this is sync_continuation task
        is_sync_continuation = self.task_type == 'sync_continuation'

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
            hidden_states = []
            cell_states = []
            network_outputs = []
            network_inputs = []
            target_outputs = []
            feedback_signals = []  # For feedback-enabled models
            mse_per_period = []
            all_phase_infos = []

            # Test each period
            for i, period_data in enumerate(test_data['sequences']):
                period = period_data['period']
                input_seq = period_data['inputs'][0]
                target_seq = period_data['targets'][0]

                # Store input and target
                network_inputs.append(input_seq)  # ADD THIS
                target_outputs.append(target_seq)  # ADD THIS

                # Get phase info if available (sync_continuation task)
                phase_info = None
                if is_sync_continuation and 'phase_infos' in period_data:
                    phase_info = period_data['phase_infos'][0]
                    all_phase_infos.append(phase_info)

                # Convert to tensors
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                target_tensor = torch.FloatTensor(target_seq).unsqueeze(0).unsqueeze(-1).to(self.device)

                # Run step by step (with or without feedback)
                with torch.no_grad():
                    if self.feedback_enabled:
                        outputs, h_t_list, c_t_list, feedback_signal = self.run_sequence_with_feedback(model, input_tensor)
                        feedback_signals.append(feedback_signal)
                    else:
                        outputs, h_t_list, c_t_list = self.run_sequence_step_by_step(model, input_tensor)

                # Calculate MSE (respecting phase masking for sync_continuation)
                if is_sync_continuation and phase_info is not None:
                    attention_end = phase_info.get('attention_end_idx', 0)
                    skipped_sync_end = phase_info.get('skipped_sync_end_idx', attention_end)
                    continuation_end = phase_info.get('continuation_end_idx', outputs.shape[1])

                    mask = torch.zeros_like(outputs, dtype=torch.bool)
                    mask[:, skipped_sync_end:continuation_end, :] = True

                    if mask.sum() > 0:
                        mse = torch.mean((outputs[mask] - target_tensor[mask]) ** 2).item()
                    else:
                        mse = torch.mean((outputs - target_tensor) ** 2).item()
                else:
                    mse = torch.mean((outputs - target_tensor) ** 2).item()

                mse_per_period.append(mse)

                # Stack states
                h_states = np.stack(h_t_list, axis=0)
                c_states = np.stack(c_t_list, axis=0)

                output_array = outputs.squeeze().cpu().numpy()

                hidden_states.append(h_states)
                cell_states.append(c_states)
                network_outputs.append(output_array)

            # Convert to arrays
            hidden_states = np.array(hidden_states)
            cell_states = np.array(cell_states)
            network_outputs = np.array(network_outputs)
            network_inputs = np.array(network_inputs)
            target_outputs = np.array(target_outputs)

            # Prepare save data
            save_data = {
                'hidden_states': hidden_states,
                'cell_states': cell_states,
                'network_outputs': network_outputs,
                'network_inputs': network_inputs,
                'target_outputs': target_outputs,
                'periods': test_data['periods'],
                'mse_per_period': mse_per_period,
                'epoch': epoch,
                'task_type': self.task_type,
                'feedback_enabled': self.feedback_enabled
            }

            # Add feedback signals if feedback is enabled
            if self.feedback_enabled and feedback_signals:
                save_data['feedback_signals'] = np.array(feedback_signals)

            # Add phase infos for sync_continuation
            if is_sync_continuation and all_phase_infos:
                save_data['phase_infos'] = self._serialize_phase_infos(all_phase_infos)

            # Save states
            save_path = self.output_dir / f'epoch_{epoch}_states.npz'
            np.savez(save_path, **save_data)
            self.logger.info(f"Saved states to {save_path}")

            # Store accuracy progression
            accuracy_progression[str(epoch)] = {
                'periods': test_data['periods'],
                'mse_per_period': mse_per_period
            }

            mean_mse = np.mean(mse_per_period)
            self.logger.info(f"Epoch {epoch}: Mean MSE = {mean_mse:.6f}")

        # Save accuracy progression
        accuracy_path = self.output_dir / 'accuracy_progression.json'
        with open(accuracy_path, 'w') as f:
            json.dump(accuracy_progression, f, indent=2)

        # Save analysis metadata
        metadata = {
            'experiment_path': str(self.experiment_path),
            'n_checkpoints_analyzed': len(checkpoint_files),
            'epochs_analyzed': [epoch for epoch, _ in checkpoint_files],
            'test_periods': test_data['periods'],
            'analysis_config': self.analysis_config,
            'task_type': self.task_type
        }

        metadata_path = self.output_dir / 'analysis_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Analysis complete. Results saved to {self.output_dir}")

    def _serialize_phase_infos(self, phase_infos: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Serialize phase_infos list to arrays for numpy saving.

        Args:
            phase_infos: List of phase info dicts

        Returns:
            Dictionary of arrays
        """
        if not phase_infos:
            return {}

        # Extract common keys
        keys_to_save = [
            'n_attention', 'n_sync', 'n_continuation',
            'attention_end_idx', 'sync_end_idx', 'continuation_end_idx',
            'skipped_sync_end_idx', 'period'
        ]

        result = {}
        for key in keys_to_save:
            values = []
            for pi in phase_infos:
                if key in pi:
                    values.append(pi[key])
                else:
                    values.append(0)  # Default value
            result[f'phase_{key}'] = np.array(values)

        return result
