"""
Input generation utilities for creating beat sequences with noise.
"""

import numpy as np
from typing import List, Tuple


def generate_noisy_pulse_times(
    first_beat_time: float,
    tempo: float,
    n_pulses: int,
    phase_noise: float,
    jitter: float,
) -> List[float]:
    """
    Generate pulse times with phase noise and jitter.

    Args:
        first_beat_time (float): First beat time in seconds.
        tempo: Base period between beats in seconds
        n_pulses: Number of pulses to generate
        phase_noise: Standard deviation of phase noise
        jitter: Standard deviation of jitter

    Returns:
        List of pulse onset times in seconds
    """
    # Random start offset (between 0 and -tempo)

    # start_offset = np.random.uniform(-tempo, 0)

    # Generate phase noise (cumulative)
    phase_noises = np.random.normal(0, phase_noise, n_pulses)

    # Generate jitter (individual)
    jitters = np.random.normal(0, jitter, n_pulses)

    pulse_times = []
    phase_times = []

    for i in range(n_pulses):
        if i == 0:
            # First beat
            # phase_time = start_offset + tempo + phase_noises[i]
            # beat_time = phase_time + jitters[i]
            phase_time = first_beat_time
            beat_time = first_beat_time
        else:
            # Subsequent beats
            phase_time = phase_times[i - 1] + tempo + phase_noises[i]
            beat_time = phase_time + jitters[i]

        phase_times.append(phase_time)
        pulse_times.append(beat_time)

    # Ensure no negative times
    pulse_times = [max(0, t) for t in pulse_times]

    return pulse_times


def create_input_tensor_from_times(
        pulse_times: List[float],
        pulse_width: float,
        pulse_height: float,
        sequence_length: float,
        dt: float,
        baseline_value: float = 0.0
) -> np.ndarray:
    """
    Create input tensor from pulse times.

    Args:
        pulse_times: List of pulse onset times in seconds
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of pulses
        sequence_length: Total sequence length in seconds
        dt: Time step in seconds
        baseline_value: Baseline value when no pulse

    Returns:
        Input sequence array (timesteps,)
    """
    n_timesteps = int(sequence_length / dt)
    input_seq = np.ones(n_timesteps) * baseline_value

    # Calculate pulse width in samples
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


def parse_pulse_times_string(times_str: str) -> List[float]:
    """
    Parse comma-separated string of pulse times.

    Args:
        times_str: Comma-separated string of times

    Returns:
        List of pulse times as floats
    """
    try:
        times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
        return times
    except ValueError as e:
        raise ValueError(f"Invalid pulse times format: {e}")


def validate_pulse_times(
        pulse_times: List[float],
        sequence_length: float,
        pulse_width: float
) -> Tuple[bool, str]:
    """
    Validate pulse times for potential issues.

    Args:
        pulse_times: List of pulse onset times
        sequence_length: Total sequence length
        pulse_width: Width of each pulse

    Returns:
        (is_valid, message) tuple
    """
    if not pulse_times:
        return False, "No pulse times provided"

    # Check for negative times
    if any(t < 0 for t in pulse_times):
        return False, "Pulse times cannot be negative"

    # Check if any pulse extends beyond sequence
    max_time = max(pulse_times) + pulse_width
    if max_time > sequence_length:
        return True, f"Warning: Last pulse extends beyond sequence ({max_time:.3f} > {sequence_length})"

    # Check for overlapping pulses
    sorted_times = sorted(pulse_times)
    for i in range(len(sorted_times) - 1):
        if sorted_times[i + 1] - sorted_times[i] < pulse_width:
            return True, f"Warning: Overlapping pulses detected at {sorted_times[i]:.3f}s and {sorted_times[i + 1]:.3f}s"

    return True, "Pulse times valid"
