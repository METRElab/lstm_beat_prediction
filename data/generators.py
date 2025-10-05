"""
Data generation functions for beat prediction task.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import torch


def generate_beat_sequence(
        period: float,
        n_pulses: int,
        pulse_width: float,
        pulse_height: float,
        dt: float,
        sequence_length: float,
        baseline_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single beat sequence with its target.

    Args:
        period: Time between beats in seconds
        n_pulses: Number of pulses in the sequence
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of the pulse
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        baseline_value: Value when no pulse is present

    Returns:
        input_sequence: 1D array of input values
        target_sequence: 1D array of target values (shifted earlier by pulse_width)
    """
    n_timesteps = int(sequence_length / dt)

    # Initialize sequences with baseline
    input_sequence = np.ones(n_timesteps) * baseline_value
    target_sequence = np.ones(n_timesteps) * baseline_value

    # Calculate number of samples for pulse width
    pulse_samples = int(pulse_width / dt)

    # Start time for first pulse (add some initial delay)
    start_delay = period * 0.5

    # Generate input pulses
    for i in range(n_pulses):
        pulse_start_time = start_delay + i * period
        pulse_start_idx = int(pulse_start_time / dt)
        pulse_end_idx = pulse_start_idx + pulse_samples

        # Add pulse to input if within bounds
        if pulse_end_idx < n_timesteps:
            input_sequence[pulse_start_idx:pulse_end_idx] = pulse_height

    # Generate target (prediction) - same as input but shifted earlier
    for i in range(n_pulses):
        pulse_start_time = start_delay + i * period
        # Target appears one pulse_width earlier
        target_start_time = pulse_start_time - pulse_width
        target_start_idx = int(target_start_time / dt)
        target_end_idx = target_start_idx + pulse_samples

        # Add pulse to target if within bounds
        if target_start_idx >= 0 and target_end_idx < n_timesteps:
            target_sequence[target_start_idx:target_end_idx] = pulse_height

    return input_sequence, target_sequence


def create_batch(
        batch_size: int,
        min_period: float,
        max_period: float,
        n_pulses: int,
        pulse_width: float,
        pulse_height: float,
        dt: float,
        sequence_length: float,
        baseline_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Create a batch of sequences with random periods.

    Args:
        batch_size: Number of sequences in batch
        min_period: Minimum period in seconds
        max_period: Maximum period in seconds
        n_pulses: Number of pulses per sequence
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of pulses
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        baseline_value: Baseline value when no pulse

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences (batch_size, sequence_length, 1)
        periods: List of periods used for each sequence
    """
    n_timesteps = int(sequence_length / dt)

    # Initialize batch tensors
    inputs = np.zeros((batch_size, n_timesteps, 1))
    targets = np.zeros((batch_size, n_timesteps, 1))
    periods = []

    for i in range(batch_size):
        # Random period for this sequence
        period = np.random.uniform(min_period, max_period)
        periods.append(period)

        # Generate sequence
        input_seq, target_seq = generate_beat_sequence(
            period=period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            baseline_value=baseline_value
        )

        # Add to batch (expand dims for LSTM input)
        inputs[i, :, 0] = input_seq
        targets[i, :, 0] = target_seq

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods


def generate_test_sequences(
        n_periods: int,
        min_period: float,
        max_period: float,
        trials_per_period: int,
        n_pulses: int,
        pulse_width: float,
        pulse_height: float,
        dt: float,
        sequence_length: float,
        baseline_value: float = 0.0
) -> Dict[str, Any]:
    """
    Generate test sequences with evenly spaced periods.

    Args:
        n_periods: Number of different periods to test
        min_period: Minimum period
        max_period: Maximum period
        trials_per_period: Number of trials for each period
        n_pulses: Number of pulses per sequence
        pulse_width: Width of each pulse
        pulse_height: Height of pulses
        dt: Time step
        sequence_length: Total sequence length
        baseline_value: Baseline value

    Returns:
        Dictionary containing test data organized by period
    """
    # Create evenly spaced periods
    test_periods = np.linspace(min_period, max_period, n_periods)

    test_data = {
        'periods': test_periods.tolist(),
        'sequences': []
    }

    for period in test_periods:
        period_data = {
            'period': float(period),
            'inputs': [],
            'targets': []
        }

        for _ in range(trials_per_period):
            input_seq, target_seq = generate_beat_sequence(
                period=period,
                n_pulses=n_pulses,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                baseline_value=baseline_value
            )

            period_data['inputs'].append(input_seq)
            period_data['targets'].append(target_seq)

        # Convert to numpy arrays
        period_data['inputs'] = np.array(period_data['inputs'])
        period_data['targets'] = np.array(period_data['targets'])

        test_data['sequences'].append(period_data)

    return test_data
