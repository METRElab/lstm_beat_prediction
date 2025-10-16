"""
Data generation functions for beat prediction task.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch


def generate_beat_sequence_rectangle(
    period: float,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    iti: float,
    baseline_value: float = 0.0,
    skip_first_n: int = 0,
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
        iti: Inter-trial interval (time before first and after last beat)
        baseline_value: Value when no pulse is present
        skip_first_n: Number of initial target pulses to skip (keep baseline)

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

    # Start time for first pulse (after ITI/2)
    start_delay = iti / 2

    # Generate input pulses
    for i in range(n_pulses):
        pulse_start_time = start_delay + i * period
        pulse_start_idx = int(pulse_start_time / dt)
        pulse_end_idx = pulse_start_idx + pulse_samples

        # Add pulse to input if within bounds
        if pulse_end_idx < n_timesteps:
            input_sequence[pulse_start_idx:pulse_end_idx] = pulse_height

    # Generate target (prediction) - same as input but shifted earlier
    # Start from pulse index skip_first_n instead of 0
    for i in range(skip_first_n, n_pulses):
        pulse_start_time = start_delay + i * period
        # Target appears one pulse_width earlier
        target_start_time = pulse_start_time - pulse_width
        target_start_idx = int(target_start_time / dt)
        target_end_idx = target_start_idx + pulse_samples

        # Add pulse to target if within bounds
        if target_start_idx >= 0 and target_end_idx < n_timesteps:
            target_sequence[target_start_idx:target_end_idx] = pulse_height

    # Add an extra predicted pulse after the last input pulse
    # (shifted by one pulse_width to the left from where the next pulse would be)
    extra_pulse_time = start_delay + n_pulses * period - pulse_width
    extra_pulse_start_idx = int(extra_pulse_time / dt)
    extra_pulse_end_idx = extra_pulse_start_idx + pulse_samples

    if extra_pulse_start_idx >= 0 and extra_pulse_end_idx < n_timesteps:
        target_sequence[extra_pulse_start_idx:extra_pulse_end_idx] = pulse_height

    return input_sequence, target_sequence


def calculate_iti(min_iti: float, mean_iti: float) -> float:
    """
    Calculate inter-trial interval using exponential distribution.

    Args:
        min_iti: Minimum ITI value
        mean_iti: Mean ITI value for exponential distribution

    Returns:
        ITI value sampled from shifted exponential distribution
    """
    # Calculate scale parameter for exponential distribution
    # Since we want mean_iti after shifting by min_iti
    scale = mean_iti - min_iti

    # Sample from exponential and add minimum
    iti = min_iti + np.random.exponential(scale)

    return iti


def create_batch_rectangle(
    batch_size: int,
    min_period: float,
    max_period: float,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    min_iti: float,
    mean_iti: float,
    baseline_value: float = 0.0,
    skip_first_n: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Create a batch of sequences with random periods and ITIs.

    Args:
        batch_size: Number of sequences in batch
        min_period: Minimum period in seconds
        max_period: Maximum period in seconds
        n_pulses: Number of pulses per sequence
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of pulses
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        min_iti: Minimum inter-trial interval
        mean_iti: Mean inter-trial interval
        baseline_value: Baseline value when no pulse
        skip_first_n: Number of initial target pulses to skip (keep baseline)

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
        step = 0.1  # your custom step size
        # Generate discrete values
        values = np.arange(min_period, max_period + step, step)

        # Randomly choose one
        period = np.random.choice(values)
        # period = np.random.uniform(min_period, max_period)
        periods.append(period)

        # Calculate ITI for this trial
        iti = calculate_iti(min_iti, mean_iti)

        # Generate sequence
        input_seq, target_seq = generate_beat_sequence_rectangle(
            period=period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            iti=iti,
            baseline_value=baseline_value,
            skip_first_n=skip_first_n,
        )

        # Add to batch (expand dims for LSTM input)
        inputs[i, :, 0] = input_seq
        targets[i, :, 0] = target_seq

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods


def generate_test_sequences_rectangle(
    n_periods: int,
    min_period: float,
    max_period: float,
    trials_per_period: int,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    min_iti: float,
    mean_iti: float,
    baseline_value: float = 0.0,
    skip_first_n: int = 0,
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
        min_iti: Minimum inter-trial interval
        mean_iti: Mean inter-trial interval
        baseline_value: Baseline value
        skip_first_n: Number of initial target pulses to skip (keep baseline)

    Returns:
        Dictionary containing test data organized by period
    """
    # Create evenly spaced periods
    test_periods = np.linspace(min_period, max_period, n_periods)

    test_data = {
        "periods": test_periods.tolist(),
        "sequences": [],
        "skip_first_n": skip_first_n,
    }

    for period in test_periods:
        period_data = {"period": float(period), "inputs": [], "targets": []}

        for _ in range(trials_per_period):
            # Calculate ITI for this trial
            iti = calculate_iti(min_iti, mean_iti)

            input_seq, target_seq = generate_beat_sequence_rectangle(
                period=period,
                n_pulses=n_pulses,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                iti=iti,
                baseline_value=baseline_value,
                skip_first_n=skip_first_n,
            )

            period_data["inputs"].append(input_seq)
            period_data["targets"].append(target_seq)

        # Convert to numpy arrays
        period_data["inputs"] = np.array(period_data["inputs"])
        period_data["targets"] = np.array(period_data["targets"])

        test_data["sequences"].append(period_data)

    return test_data


def gamma_like_curve_numpy(
    dt: float, length: float, max_height: float = 1.0, shape_param: float = 2.0
) -> np.ndarray:
    """
    Creates a gamma-like distribution that reaches zero at the specified length.

    Args:
        dt: Time step size
        length: Total duration after which the curve reaches zero
        max_height: Maximum height of the curve
        shape_param: Shape parameter controlling the skewness

    Returns:
        Curve values as numpy array
    """
    # Create time points
    t = np.arange(0, length + dt, dt, dtype=np.float32)

    # Normalize time to [0, 1] for easier calculation
    t_norm = t / length

    # Create gamma-like shape that goes to zero at t=length
    y = t_norm ** (shape_param - 1) * (1 - t_norm) ** 2

    # Normalize to max_height
    y_max = y.max()
    if y_max > 0:
        y = y * (max_height / y_max)

    return y


def generate_beat_sequence_gamma(
    period: float,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    iti: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gamma_length: Optional[float] = None,
    gamma_shape: float = 2.5,
    gamma_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single beat sequence with gamma-like target distributions.

    Args:
        period: Time between beats in seconds
        n_pulses: Number of pulses in the sequence
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of the pulse
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        iti: Inter-trial interval (time before first and after last beat)
        baseline_value: Value when no pulse is present for input
        output_offset: Offset for output baseline
        gamma_length: Duration of gamma distribution (None = use pulse_width)
        gamma_shape: Shape parameter for gamma-like curve
        gamma_max_height: Maximum height of gamma curve (None = use pulse_height - output_offset)
        skip_first_n: Number of initial pulses to skip for targets (keep baseline)

    Returns:
        input_sequence: 1D array of input values (rectangular pulses)
        target_sequence: 1D array of target values (gamma-like distributions)
    """
    n_timesteps = int(sequence_length / dt)

    # Set gamma defaults
    if gamma_length is None:
        gamma_length = pulse_width
    if gamma_max_height is None:
        gamma_max_height = pulse_height - output_offset

    # Initialize sequences
    input_sequence = np.ones(n_timesteps) * baseline_value
    target_sequence = np.ones(n_timesteps) * (-output_offset)  # Target baseline

    # Calculate number of samples for pulse width
    pulse_samples = int(pulse_width / dt)

    # Start time for first pulse (after ITI/2)
    start_delay = iti / 2

    # Generate input pulses (rectangular)
    for i in range(n_pulses):
        pulse_start_time = start_delay + i * period
        pulse_start_idx = int(pulse_start_time / dt)
        pulse_end_idx = pulse_start_idx + pulse_samples

        # Add pulse to input if within bounds
        if pulse_end_idx < n_timesteps:
            input_sequence[pulse_start_idx:pulse_end_idx] = pulse_height

    # Generate gamma-like curve template
    gamma_curve = gamma_like_curve_numpy(
        dt, gamma_length, gamma_max_height, gamma_shape
    )

    # Generate target gamma distributions
    # Skip first few pulses based on skip_first_n parameter
    for i in range(skip_first_n, n_pulses):
        # Each gamma distribution should end right before the input pulse starts
        pulse_start_time = start_delay + i * period
        gamma_start_time = pulse_start_time - gamma_length
        gamma_start_idx = int(gamma_start_time / dt)
        gamma_end_idx = gamma_start_idx + len(gamma_curve)

        # Add gamma curve to target if within bounds
        if gamma_start_idx >= 0 and gamma_end_idx <= n_timesteps:
            target_sequence[gamma_start_idx:gamma_end_idx] = gamma_curve + (
                -output_offset
            )

    # Add an extra gamma distribution after the last input pulse
    # This one is shifted by one pulse width to the left from where the next pulse would be
    extra_pulse_time = start_delay + n_pulses * period - pulse_width
    extra_gamma_start = extra_pulse_time - gamma_length
    extra_gamma_start_idx = int(extra_gamma_start / dt)
    extra_gamma_end_idx = extra_gamma_start_idx + len(gamma_curve)

    if extra_gamma_start_idx >= 0 and extra_gamma_end_idx <= n_timesteps:
        target_sequence[extra_gamma_start_idx:extra_gamma_end_idx] = gamma_curve + (
            -output_offset
        )

    return input_sequence, target_sequence


def create_batch_gamma(
    batch_size: int,
    min_period: float,
    max_period: float,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    min_iti: float,
    mean_iti: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gamma_length: Optional[float] = None,
    gamma_shape: float = 2.5,
    gamma_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Create a batch of sequences with gamma-like targets and random periods and ITIs.

    Args:
        batch_size: Number of sequences in batch
        min_period: Minimum period in seconds
        max_period: Maximum period in seconds
        n_pulses: Number of pulses per sequence
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of pulses
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        min_iti: Minimum inter-trial interval
        mean_iti: Mean inter-trial interval
        baseline_value: Baseline value when no pulse for input
        output_offset: Offset for output baseline
        gamma_length: Duration of gamma distribution (None = use pulse_width)
        gamma_shape: Shape parameter for gamma-like curve
        gamma_max_height: Maximum height of gamma curve (None = use pulse_height - output_offset)
        skip_first_n: Number of initial pulses to skip for targets

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences with gamma distributions (batch_size, sequence_length, 1)
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

        # Calculate ITI for this trial
        iti = calculate_iti(min_iti, mean_iti)

        # Generate sequence with gamma targets
        input_seq, target_seq = generate_beat_sequence_gamma(
            period=period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            iti=iti,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gamma_length=gamma_length,
            gamma_shape=gamma_shape,
            gamma_max_height=gamma_max_height,
            skip_first_n=skip_first_n,
        )

        # Add to batch (expand dims for LSTM input)
        inputs[i, :, 0] = input_seq
        targets[i, :, 0] = target_seq

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods


def generate_test_sequences_gamma(
    n_periods: int,
    min_period: float,
    max_period: float,
    trials_per_period: int,
    n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    min_iti: float,
    mean_iti: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gamma_length: Optional[float] = None,
    gamma_shape: float = 2.5,
    gamma_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Dict[str, Any]:
    """
    Generate test sequences with gamma-like targets and evenly spaced periods.

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
        min_iti: Minimum inter-trial interval
        mean_iti: Mean inter-trial interval
        baseline_value: Baseline value for input
        output_offset: Offset for output baseline
        gamma_length: Duration of gamma distribution (None = use pulse_width)
        gamma_shape: Shape parameter for gamma-like curve
        gamma_max_height: Maximum height of gamma curve (None = use pulse_height - output_offset)
        skip_first_n: Number of initial pulses to skip for targets

    Returns:
        Dictionary containing test data organized by period
    """
    # Create evenly spaced periods
    test_periods = np.linspace(min_period, max_period, n_periods)

    test_data = {
        "periods": test_periods.tolist(),
        "sequences": [],
        "target_type": "gamma",
        "gamma_params": {
            "gamma_shape": gamma_shape,
            "gamma_length": gamma_length or pulse_width,
            "gamma_max_height": gamma_max_height or (pulse_height - output_offset),
            "skip_first_n": skip_first_n,
        },
    }

    for period in test_periods:
        period_data = {"period": float(period), "inputs": [], "targets": []}

        for _ in range(trials_per_period):
            # Calculate ITI for this trial
            iti = calculate_iti(min_iti, mean_iti)

            input_seq, target_seq = generate_beat_sequence_gamma(
                period=period,
                n_pulses=n_pulses,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                iti=iti,
                baseline_value=baseline_value,
                output_offset=output_offset,
                gamma_length=gamma_length,
                gamma_shape=gamma_shape,
                gamma_max_height=gamma_max_height,
                skip_first_n=skip_first_n,
            )

            period_data["inputs"].append(input_seq)
            period_data["targets"].append(target_seq)

        # Convert to numpy arrays
        period_data["inputs"] = np.array(period_data["inputs"])
        period_data["targets"] = np.array(period_data["targets"])

        test_data["sequences"].append(period_data)

    return test_data


def create_batch_from_config(
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Create a batch of sequences using configuration dictionary.

    Args:
        config: Configuration dictionary containing all parameters

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences (batch_size, sequence_length, 1)
        periods: List of periods used for each sequence
    """
    # Extract common parameters from config
    batch_size = config["training"]["batch_size"]
    min_period = config["task"]["min_period"]
    max_period = config["task"]["max_period"]
    n_pulses = config["task"]["n_pulses"]
    pulse_width = config["task"]["pulse_width"]
    pulse_height = config["task"]["pulse_height"]
    dt = config["task"]["dt"]
    sequence_length = config["task"]["sequence_length"]
    min_iti = config["task"]["min_iti"]
    mean_iti = config["task"]["mean_iti"]
    baseline_value = config["task"]["baseline_value"]

    # Extract target type and optional parameters
    target_type = config["task"].get("target_type", "rectangle")
    output_offset = config["task"].get("output_offset", -0.5)
    skip_first_n = config["task"].get("skip_first_n", 2)

    # Extract gamma parameters if present
    gamma_params = config["task"].get("gamma_params", {})
    gamma_length = gamma_params.get("gamma_length", None)
    gamma_shape = gamma_params.get("gamma_shape", 2.5)
    gamma_max_height = gamma_params.get("gamma_max_height", None)

    if target_type == "gamma":
        return create_batch_gamma(
            batch_size=batch_size,
            min_period=min_period,
            max_period=max_period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            min_iti=min_iti,
            mean_iti=mean_iti,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gamma_length=gamma_length,
            gamma_shape=gamma_shape,
            gamma_max_height=gamma_max_height,
            skip_first_n=skip_first_n,
        )
    elif target_type == "rectangle":  # rectangle
        return create_batch_rectangle(
            batch_size=batch_size,
            min_period=min_period,
            max_period=max_period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            min_iti=min_iti,
            mean_iti=mean_iti,
            baseline_value=baseline_value,
            skip_first_n=skip_first_n,
        )
    else:
        raise ValueError("Unknown target type")


def generate_test_sequences_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate test sequences using configuration dictionary.

    Args:
        config: Configuration dictionary containing all parameters

    Returns:
        Dictionary containing test data organized by period
    """
    # Extract parameters from config
    n_periods = config["testing"]["n_test_periods"]
    trials_per_period = config["testing"]["test_trials_per_period"]

    min_period = config["task"]["min_period"]
    max_period = config["task"]["max_period"]
    n_pulses = config["task"]["n_pulses"]
    pulse_width = config["task"]["pulse_width"]
    pulse_height = config["task"]["pulse_height"]
    dt = config["task"]["dt"]
    sequence_length = config["task"]["sequence_length"]
    min_iti = config["task"]["min_iti"]
    mean_iti = config["task"]["mean_iti"]
    baseline_value = config["task"]["baseline_value"]

    # Extract target type and optional parameters
    target_type = config["task"].get("target_type", "rectangle")
    output_offset = config["task"].get("output_offset", -0.5)
    skip_first_n = config["task"].get("skip_first_n", 2)

    # Extract gamma parameters if present
    gamma_params = config["task"].get("gamma_params", {})
    gamma_length = gamma_params.get("gamma_length", None)
    gamma_shape = gamma_params.get("gamma_shape", 2.5)
    gamma_max_height = gamma_params.get("gamma_max_height", None)

    if target_type == "gamma":
        return generate_test_sequences_gamma(
            n_periods=n_periods,
            min_period=min_period,
            max_period=max_period,
            trials_per_period=trials_per_period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            min_iti=min_iti,
            mean_iti=mean_iti,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gamma_length=gamma_length,
            gamma_shape=gamma_shape,
            gamma_max_height=gamma_max_height,
            skip_first_n=skip_first_n,
        )
    elif target_type == "rectangle":  # rectangle
        return generate_test_sequences_rectangle(
            n_periods=n_periods,
            min_period=min_period,
            max_period=max_period,
            trials_per_period=trials_per_period,
            n_pulses=n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            min_iti=min_iti,
            mean_iti=mean_iti,
            baseline_value=baseline_value,
            skip_first_n=skip_first_n,
        )
    else:
        raise ValueError("Unknown target type")
