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
) -> Tuple[np.ndarray, np.ndarray, int]:
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
        last_pulse_idx: int last pulse index
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

    # Find last pulse end idx
    last_pulse_start_time = start_delay + (n_pulses - 1) * period
    last_pulse_start_idx = int(last_pulse_start_time / dt)
    last_pulse_end_idx = last_pulse_start_idx + pulse_samples

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

    return input_sequence, target_sequence, extra_pulse_end_idx


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


def sample_n_pulses(min_n: int, max_n: int) -> int:
    """
    Sample pulse count from truncated Poisson distribution.

    Args:
        min_n: Minimum number of pulses
        max_n: Maximum number of pulses

    Returns:
        Sampled pulse count in [min_n, max_n]
    """
    if min_n == max_n:
        return min_n

    # Use Poisson with lambda = midpoint, truncated to [min_n, max_n]
    lambda_param = (min_n + max_n) / 2
    while True:
        sample = np.random.poisson(lambda_param)
        if min_n <= sample <= max_n:
            return sample


def sample_period(task_config: Dict[str, Any]) -> float:
    """
    Sample period from config - either discrete list, discrete with steps, or continuous range.

    Args:
        task_config: Task configuration dictionary

    Returns:
        Sampled period value
    """
    # Check if discrete periods are specified as a list
    if "period_values" in task_config:
        return float(np.random.choice(task_config["period_values"]))

    # Check if step size is specified (discrete with steps)
    if "period_step" in task_config:
        min_p = task_config["min_period"]
        max_p = task_config["max_period"]
        step = task_config["period_step"]
        values = np.arange(min_p, max_p + step / 2, step)  # +step/2 to include max
        return float(np.random.choice(values))

    # Default: continuous uniform
    return float(np.random.uniform(task_config["min_period"], task_config["max_period"]))


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
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
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
        last_inputs_idx: Index of last input pulse in each sequence (batch_size)
    """
    n_timesteps = int(sequence_length / dt)

    # Initialize batch tensors
    inputs = np.zeros((batch_size, n_timesteps, 1))
    targets = np.zeros((batch_size, n_timesteps, 1))
    last_inputs_idx = []
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
        input_seq, target_seq, last_input_pulse_idx = generate_beat_sequence_rectangle(
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
        last_inputs_idx.append(last_input_pulse_idx)

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods, last_inputs_idx


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

            input_seq, target_seq, _ = generate_beat_sequence_rectangle(
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
) -> Tuple[np.ndarray, np.ndarray, int]:
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
        last_pulse_end_idx: int, index of the last pulse
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

    # Find last pulse end idx
    last_pulse_start_time = start_delay + (n_pulses - 1) * period
    last_pulse_start_idx = int(last_pulse_start_time / dt)
    last_pulse_end_idx = last_pulse_start_idx + pulse_samples

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

    return input_sequence, target_sequence, last_pulse_end_idx


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
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
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
        last_inputs_idx: Index of last input pulse in each sequence (batch_size)

    """
    n_timesteps = int(sequence_length / dt)

    # Initialize batch tensors
    inputs = np.zeros((batch_size, n_timesteps, 1))
    targets = np.zeros((batch_size, n_timesteps, 1))
    periods = []
    last_inputs_idx = []

    for i in range(batch_size):
        # Random period for this sequence
        period = np.random.uniform(min_period, max_period)
        periods.append(period)

        # Calculate ITI for this trial
        iti = calculate_iti(min_iti, mean_iti)

        # Generate sequence with gamma targets
        input_seq, target_seq, last_input_pulse_idx = generate_beat_sequence_gamma(
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
        last_inputs_idx.append(last_input_pulse_idx)

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods, last_inputs_idx


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

            input_seq, target_seq, _ = generate_beat_sequence_gamma(
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


def gaussian_curve_numpy(
    dt: float, length: float, max_height: float = 1.0, sigma: float = 0.05
) -> np.ndarray:
    """
    Creates a Gaussian distribution that spans the specified length.

    Args:
        dt: Time step size
        length: Total duration of the Gaussian curve
        max_height: Maximum height of the curve at the peak
        sigma: Standard deviation of the Gaussian (controls width)

    Returns:
        Curve values as numpy array
    """
    # Create time points
    t = np.arange(0, length + dt, dt, dtype=np.float32)

    # Center the Gaussian at the end of the length
    mean = length / 2

    # Create Gaussian shape
    y = max_height * np.exp(-0.5 * ((t - mean) / sigma) ** 2)

    return y


def calculate_beat_times_with_noise(
    period: float,
    n_pulses: int,
    phase_noise_std: float,
    jitter_std: float,
    random_start_offset: float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate beat times with phase noise and jitter.

    Args:
        period: Base period between beats
        n_pulses: Number of pulses
        phase_noise_std: Standard deviation for phase noise
        jitter_std: Standard deviation for jitter
        random_start_offset: Random offset before first beat (negative value)

    Returns:
        beat_times: Final beat times including all noise
        phase_times: Beat times with only phase noise (no jitter)
        jitters: Individual jitter values for each beat
    """
    # Generate phase noise for each beat (cumulative effect)
    phase_noises = np.random.normal(0, phase_noise_std, n_pulses)

    # Generate jitter for each beat (individual effect)
    jitters = np.random.normal(0, jitter_std, n_pulses)

    beat_times = []
    phase_times = []

    for i in range(n_pulses):
        if i == 0:
            # First beat: random_start_offset + period + phase_noise + jitter
            phase_time = random_start_offset + period + phase_noises[i]
            beat_time = phase_time + jitters[i]
        else:
            # Subsequent beats: previous_phase + period + phase_noise + jitter
            phase_time = phase_times[i - 1] + period + phase_noises[i]
            beat_time = phase_time + jitters[i]

        phase_times.append(phase_time)
        beat_times.append(beat_time)

    return beat_times, phase_times, jitters.tolist()


def generate_beat_sequence_gaussian_with_noise(
    period: float,
    phase_noise_std: float,
    jitter_std: float,
    min_n_pulses: int,
    max_n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gaussian_length: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    gaussian_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a single beat sequence with Gaussian targets and timing noise.

    Args:
        period: Base time between beats in seconds
        phase_noise_std: Standard deviation for phase noise
        jitter_std: Standard deviation for jitter noise
        min_n_pulses: Minimum number of pulses
        max_n_pulses: Maximum number of pulses
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of the pulse
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        baseline_value: Value when no pulse is present for input
        output_offset: Offset for output baseline
        gaussian_length: Duration of Gaussian distribution (None = use pulse_width)
        gaussian_sigma: Sigma for Gaussian curve (None = gaussian_length / 6)
        gaussian_max_height: Maximum height of Gaussian curve
        skip_first_n: Number of initial pulses to skip for targets

    Returns:
        input_sequence: 1D array of input values (rectangular pulses)
        target_sequence: 1D array of target values (Gaussian distributions)
        last_pulse_idx: int, index of the last pulse
    """
    n_timesteps = int(sequence_length / dt)

    # Randomly select number of pulses
    n_pulses = np.random.randint(min_n_pulses, max_n_pulses + 1)

    # Set Gaussian defaults
    if gaussian_length is None:
        gaussian_length = pulse_width
    if gaussian_sigma is None:
        gaussian_sigma = gaussian_length / 6  # 99.7% within length
    if gaussian_max_height is None:
        gaussian_max_height = pulse_height - output_offset

    # Initialize sequences
    input_sequence = np.ones(n_timesteps) * baseline_value
    target_sequence = np.ones(n_timesteps) * (-output_offset)  # Target baseline

    # Calculate number of samples for pulse width
    pulse_samples = int(pulse_width / dt)

    # Random start offset (between 0 and -period)
    random_start_offset = np.random.uniform(-period, 0)

    # Calculate beat times with noise
    beat_times, _, _ = calculate_beat_times_with_noise(
        period, n_pulses, phase_noise_std, jitter_std, random_start_offset
    )

    # Track last pulse index
    last_pulse_idx = 0

    # Generate input pulses (rectangular) at noisy beat times
    for beat_time in beat_times:
        pulse_start_idx = int(beat_time / dt)
        pulse_end_idx = pulse_start_idx + pulse_samples

        # Add pulse to input if within bounds
        if pulse_start_idx >= 0 and pulse_end_idx < n_timesteps:
            input_sequence[pulse_start_idx:pulse_end_idx] = pulse_height
            last_pulse_idx = pulse_end_idx
        elif pulse_start_idx < n_timesteps:
            # Partial pulse at the end
            input_sequence[pulse_start_idx : min(pulse_end_idx, n_timesteps)] = (
                pulse_height
            )
            last_pulse_idx = min(pulse_end_idx, n_timesteps)

    # Generate Gaussian curve template
    gaussian_curve = gaussian_curve_numpy(
        dt, gaussian_length, gaussian_max_height, gaussian_sigma
    )

    # Generate target Gaussian distributions
    # Skip first few pulses based on skip_first_n parameter
    for i, beat_time in enumerate(beat_times):
        if i < skip_first_n:
            continue

        # Each Gaussian distribution should end right before the input pulse starts
        gaussian_end_time = beat_time
        gaussian_end_idx = int(gaussian_end_time / dt)
        gaussian_start_idx = gaussian_end_idx - len(gaussian_curve)

        # Add Gaussian curve to target if within bounds
        if gaussian_start_idx >= 0 and gaussian_end_idx <= n_timesteps:
            target_sequence[gaussian_start_idx:gaussian_end_idx] = gaussian_curve + (
                -output_offset
            )
        elif gaussian_start_idx < n_timesteps and gaussian_end_idx > 0:
            # Handle partial Gaussian at boundaries
            valid_start = max(0, gaussian_start_idx)
            valid_end = min(n_timesteps, gaussian_end_idx)
            curve_start = max(0, -gaussian_start_idx)
            curve_end = curve_start + (valid_end - valid_start)
            if curve_end <= len(gaussian_curve):
                target_sequence[valid_start:valid_end] = gaussian_curve[
                    curve_start:curve_end
                ] + (-output_offset)

    # Add an extra predicted Gaussian after the last input pulse
    valid_end = n_timesteps
    if len(beat_times) > 0:
        # Predict next beat based on period (without noise for prediction)
        next_beat_time = beat_times[-1] + period
        gaussian_end_time = next_beat_time
        gaussian_end_idx = int(gaussian_end_time / dt)
        gaussian_start_idx = gaussian_end_idx - len(gaussian_curve)

        if gaussian_start_idx >= 0 and gaussian_end_idx <= n_timesteps:
            target_sequence[gaussian_start_idx:gaussian_end_idx] = gaussian_curve + (
                -output_offset
            )
            valid_end = gaussian_end_idx
        elif gaussian_start_idx < n_timesteps and gaussian_end_idx > 0:
            valid_start = max(0, gaussian_start_idx)
            valid_end = min(n_timesteps, gaussian_end_idx)
            curve_start = max(0, -gaussian_start_idx)
            curve_end = curve_start + (valid_end - valid_start)
            if curve_end <= len(gaussian_curve):
                target_sequence[valid_start:valid_end] = gaussian_curve[
                    curve_start:curve_end
                ] + (-output_offset)

    return input_sequence, target_sequence, valid_end


def create_batch_gaussian_with_noise(
    batch_size: int,
    min_period: float,
    max_period: float,
    phase_noise_std: float,
    jitter_std: float,
    min_n_pulses: int,
    max_n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gaussian_length: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    gaussian_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
    """
    Create a batch of sequences with Gaussian targets and timing noise.

    Args:
        batch_size: Number of sequences in batch
        min_period: Minimum period in seconds
        max_period: Maximum period in seconds
        phase_noise_std: Standard deviation for phase noise
        jitter_std: Standard deviation for jitter noise
        min_n_pulses: Minimum number of pulses
        max_n_pulses: Maximum number of pulses
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of pulses
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        baseline_value: Baseline value when no pulse for input
        output_offset: Offset for output baseline
        gaussian_length: Duration of Gaussian distribution
        gaussian_sigma: Sigma for Gaussian curve
        gaussian_max_height: Maximum height of Gaussian curve
        skip_first_n: Number of initial pulses to skip for targets

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences with Gaussian distributions
        periods: List of periods used for each sequence
        last_inputs_idx: Index of last input pulse in each sequence
    """
    n_timesteps = int(sequence_length / dt)

    # Initialize batch tensors
    inputs = np.zeros((batch_size, n_timesteps, 1))
    targets = np.zeros((batch_size, n_timesteps, 1))
    periods = []
    last_inputs_idx = []

    for i in range(batch_size):
        # Random period for this sequence
        period = np.random.uniform(min_period, max_period)
        periods.append(period)

        # Generate sequence with Gaussian targets and noise
        input_seq, target_seq, last_input_pulse_idx = (
            generate_beat_sequence_gaussian_with_noise(
                period=period,
                phase_noise_std=phase_noise_std,
                jitter_std=jitter_std,
                min_n_pulses=min_n_pulses,
                max_n_pulses=max_n_pulses,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                baseline_value=baseline_value,
                output_offset=output_offset,
                gaussian_length=gaussian_length,
                gaussian_sigma=gaussian_sigma,
                gaussian_max_height=gaussian_max_height,
                skip_first_n=skip_first_n,
            )
        )

        # Add to batch
        inputs[i, :, 0] = input_seq
        targets[i, :, 0] = target_seq
        last_inputs_idx.append(last_input_pulse_idx)

    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods, last_inputs_idx


def create_batch_sync_continuation(
        config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int], List[Dict[str, Any]]]:
    """
    Create a batch of sync-continuation sequences.

    Args:
        config: Configuration dictionary

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences (batch_size, sequence_length, 1)
        periods: List of periods used for each sequence
        continuation_end_indices: Index where continuation ends for each sequence
        phase_infos: List of phase info dicts for each sequence
    """
    batch_size = config["training"]["batch_size"]
    task_config = config["task"]

    # Extract phase parameters
    attention_phase = task_config.get("attention_phase", {})
    min_n_attention = attention_phase.get("min_n_pulses", 2)
    max_n_attention = attention_phase.get("max_n_pulses", 4)

    sync_phase = task_config.get("sync_phase", {})
    min_n_sync = sync_phase.get("min_n_pulses", 4)
    max_n_sync = sync_phase.get("max_n_pulses", 6)

    continuation_phase = task_config.get("continuation_phase", {})
    min_n_continuation = continuation_phase.get("min_n_pulses", 2)
    max_n_continuation = continuation_phase.get("max_n_pulses", 4)

    # Extract other parameters
    pulse_width = task_config["pulse_width"]
    pulse_height = task_config["pulse_height"]
    dt = task_config["dt"]
    sequence_length = task_config["sequence_length"]
    baseline_value = task_config.get("baseline_value", 0.0)
    output_offset = task_config.get("output_offset", 0.0)

    target_shape = task_config.get("target_shape", "gaussian")

    noise_params = task_config.get("noise_params", {})
    phase_noise_std = noise_params.get("phase_noise_std", 0.0)
    jitter_std = noise_params.get("jitter_std", 0.0)

    gamma_params = task_config.get("gamma_params", {})
    gamma_length = gamma_params.get("gamma_length", None)
    gamma_shape = gamma_params.get("gamma_shape", 2.5)
    gamma_max_height = gamma_params.get("gamma_max_height", None)

    gaussian_params = task_config.get("gaussian_params", {})
    gaussian_length = gaussian_params.get("gaussian_length", None)
    gaussian_sigma = gaussian_params.get("gaussian_sigma", None)
    gaussian_max_height = gaussian_params.get("gaussian_max_height", None)

    skip_first_n_sync = task_config.get("skip_first_n_sync", 0)

    n_timesteps = int(sequence_length / dt)

    # Initialize batch arrays
    inputs = np.zeros((batch_size, n_timesteps, 1))
    targets = np.zeros((batch_size, n_timesteps, 1))
    periods = []
    continuation_end_indices = []
    phase_infos = []

    for i in range(batch_size):
        # Sample period (discrete or continuous based on config)
        period = sample_period(task_config)
        periods.append(period)

        # Generate sequence
        input_seq, target_seq, cont_end_idx, phase_info = generate_beat_sequence_sync_continuation(
            period=period,
            min_n_attention=min_n_attention,
            max_n_attention=max_n_attention,
            min_n_sync=min_n_sync,
            max_n_sync=max_n_sync,
            min_n_continuation=min_n_continuation,
            max_n_continuation=max_n_continuation,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            phase_noise_std=phase_noise_std,
            jitter_std=jitter_std,
            target_shape=target_shape,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gamma_length=gamma_length,
            gamma_shape=gamma_shape,
            gamma_max_height=gamma_max_height,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            gaussian_max_height=gaussian_max_height,
            skip_first_n_sync=skip_first_n_sync,
        )

        inputs[i, :, 0] = input_seq
        targets[i, :, 0] = target_seq
        continuation_end_indices.append(cont_end_idx)
        phase_infos.append(phase_info)

    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)

    return inputs_tensor, targets_tensor, periods, continuation_end_indices, phase_infos


def generate_test_sequences_gaussian_with_noise(
    n_periods: int,
    min_period: float,
    max_period: float,
    trials_per_period: int,
    phase_noise_std: float,
    jitter_std: float,
    min_n_pulses: int,
    max_n_pulses: int,
    pulse_width: float,
    pulse_height: float,
    dt: float,
    sequence_length: float,
    baseline_value: float = 0.0,
    output_offset: float = -0.5,
    gaussian_length: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    gaussian_max_height: Optional[float] = None,
    skip_first_n: int = 2,
) -> Dict[str, Any]:
    """
    Generate test sequences with Gaussian targets and evenly spaced periods.

    Args:
        n_periods: Number of different periods to test
        min_period: Minimum period
        max_period: Maximum period
        trials_per_period: Number of trials for each period
        phase_noise_std: Standard deviation for phase noise
        jitter_std: Standard deviation for jitter noise
        min_n_pulses: Minimum number of pulses
        max_n_pulses: Maximum number of pulses
        pulse_width: Width of each pulse
        pulse_height: Height of pulses
        dt: Time step
        sequence_length: Total sequence length
        baseline_value: Baseline value for input
        output_offset: Offset for output baseline
        gaussian_length: Duration of Gaussian distribution
        gaussian_sigma: Sigma for Gaussian curve
        gaussian_max_height: Maximum height of Gaussian curve
        skip_first_n: Number of initial pulses to skip for targets

    Returns:
        Dictionary containing test data organized by period
    """
    # Create evenly spaced periods
    test_periods = np.linspace(min_period, max_period, n_periods)

    test_data = {
        "periods": test_periods.tolist(),
        "sequences": [],
        "target_type": "gaussian_noise",
        "gaussian_params": {
            "gaussian_sigma": gaussian_sigma,
            "gaussian_length": gaussian_length or pulse_width,
            "gaussian_max_height": gaussian_max_height
            or (pulse_height - output_offset),
            "skip_first_n": skip_first_n,
        },
        "noise_params": {
            "phase_noise_std": phase_noise_std,
            "jitter_std": jitter_std,
            "min_n_pulses": min_n_pulses,
            "max_n_pulses": max_n_pulses,
        },
    }

    for period in test_periods:
        period_data = {"period": float(period), "inputs": [], "targets": []}

        for _ in range(trials_per_period):
            input_seq, target_seq, _ = generate_beat_sequence_gaussian_with_noise(
                period=period,
                phase_noise_std=phase_noise_std,
                jitter_std=jitter_std,
                min_n_pulses=min_n_pulses,
                max_n_pulses=max_n_pulses,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                baseline_value=baseline_value,
                output_offset=output_offset,
                gaussian_length=gaussian_length,
                gaussian_sigma=gaussian_sigma,
                gaussian_max_height=gaussian_max_height,
                skip_first_n=skip_first_n,
            )

            period_data["inputs"].append(input_seq)
            period_data["targets"].append(target_seq)

        # Convert to numpy arrays
        period_data["inputs"] = np.array(period_data["inputs"])
        period_data["targets"] = np.array(period_data["targets"])

        test_data["sequences"].append(period_data)

    return test_data


def generate_beat_sequence_sync_continuation(
        period: float,
        min_n_attention: int,
        max_n_attention: int,
        min_n_sync: int,
        max_n_sync: int,
        min_n_continuation: int,
        max_n_continuation: int,
        pulse_width: float,
        pulse_height: float,
        dt: float,
        sequence_length: float,
        phase_noise_std: float,
        jitter_std: float,
        target_shape: str,
        baseline_value: float = 0.0,
        output_offset: float = -0.5,
        gamma_length: Optional[float] = None,
        gamma_shape: float = 2.5,
        gamma_max_height: Optional[float] = None,
        gaussian_length: Optional[float] = None,
        gaussian_sigma: Optional[float] = None,
        gaussian_max_height: Optional[float] = None,
        skip_first_n_sync: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """
    Generate three-phase beat sequence: Attention -> Synchronization -> Continuation.

    Args:
        period: Base time between beats in seconds
        min_n_attention: Minimum attention pulses
        max_n_attention: Maximum attention pulses
        min_n_sync: Minimum sync pulses
        max_n_sync: Maximum sync pulses
        min_n_continuation: Minimum continuation pulses
        max_n_continuation: Maximum continuation pulses
        pulse_width: Width of each pulse in seconds
        pulse_height: Height of the pulse
        dt: Time step in seconds
        sequence_length: Total sequence length in seconds
        phase_noise_std: Standard deviation for phase noise (attention + sync only)
        jitter_std: Standard deviation for jitter noise (attention + sync only)
        target_shape: "rectangle", "gamma", or "gaussian"
        baseline_value: Value when no pulse is present for input
        output_offset: Offset for output baseline
        gamma_length: Duration of gamma distribution
        gamma_shape: Shape parameter for gamma-like curve
        gamma_max_height: Maximum height of gamma curve
        gaussian_length: Duration of Gaussian distribution
        gaussian_sigma: Sigma for Gaussian curve
        gaussian_max_height: Maximum height of Gaussian curve
        skip_first_n_sync: Number of initial sync targets to skip

    Returns:
        input_sequence: 1D array of input values
        target_sequence: 1D array of target values
        continuation_end_idx: Index where continuation phase ends
        phase_info: Dict with phase boundaries and sampled counts
    """
    n_timesteps = int(sequence_length / dt)

    # Sample pulse counts for each phase
    n_attention = sample_n_pulses(min_n_attention, max_n_attention)
    n_sync = sample_n_pulses(min_n_sync, max_n_sync)
    n_continuation = sample_n_pulses(min_n_continuation, max_n_continuation)

    # Initialize sequences
    input_sequence = np.ones(n_timesteps) * baseline_value
    target_sequence = np.ones(n_timesteps) * (-output_offset)

    # Calculate pulse width in samples
    pulse_samples = int(pulse_width / dt)

    # Set shape-specific defaults
    if target_shape == "gamma":
        if gamma_length is None:
            gamma_length = pulse_width
        if gamma_max_height is None:
            gamma_max_height = pulse_height - output_offset
        target_curve = gamma_like_curve_numpy(dt, gamma_length, gamma_max_height, gamma_shape)
        target_length = gamma_length
    elif target_shape == "gaussian":
        if gaussian_length is None:
            gaussian_length = pulse_width
        if gaussian_sigma is None:
            gaussian_sigma = gaussian_length / 6
        if gaussian_max_height is None:
            gaussian_max_height = pulse_height - output_offset
        target_curve = gaussian_curve_numpy(dt, gaussian_length, gaussian_max_height, gaussian_sigma)
        target_length = gaussian_length
    else:  # rectangle
        target_curve = np.ones(pulse_samples) * pulse_height
        target_length = pulse_width

    # Random start offset (between 0 and period)
    random_start_offset = np.random.uniform(0, period)

    # Generate beat times for attention + sync phases (with noise)
    total_input_pulses = n_attention + n_sync

    # Generate phase noise and jitter for input phases
    phase_noises = np.random.normal(0, phase_noise_std, total_input_pulses)
    jitters = np.random.normal(0, jitter_std, total_input_pulses)

    beat_times = []
    phase_times = []

    for i in range(total_input_pulses):
        if i == 0:
            phase_time = random_start_offset
            beat_time = phase_time + jitters[i]
        else:
            phase_time = phase_times[i - 1] + period + phase_noises[i]
            beat_time = phase_time + jitters[i]

        phase_times.append(phase_time)
        beat_times.append(max(0, beat_time))  # Ensure non-negative

    # Split beat times into attention and sync
    attention_beat_times = beat_times[:n_attention]
    sync_beat_times = beat_times[n_attention:]

    # Generate continuation beat times (deterministic, based on last sync beat)
    continuation_beat_times = []
    if n_continuation > 0 and len(sync_beat_times) > 0:
        last_sync_time = sync_beat_times[-1] if sync_beat_times else (
            attention_beat_times[-1] if attention_beat_times else random_start_offset)
        for i in range(n_continuation):
            cont_time = last_sync_time + (i + 1) * period
            continuation_beat_times.append(cont_time)
    elif n_continuation > 0 and len(attention_beat_times) > 0:
        last_time = attention_beat_times[-1]
        for i in range(n_continuation):
            cont_time = last_time + (i + 1) * period
            continuation_beat_times.append(cont_time)

    # Track phase boundaries (in indices)
    phase_info = {
        "n_attention": n_attention,
        "n_sync": n_sync,
        "n_continuation": n_continuation,
        "period": period,
        "attention_beat_times": attention_beat_times,
        "sync_beat_times": sync_beat_times,
        "continuation_beat_times": continuation_beat_times,
    }

    # === Generate INPUT pulses (attention + sync phases only) ===
    all_input_times = attention_beat_times + sync_beat_times

    for beat_time in all_input_times:
        pulse_start_idx = int(beat_time / dt)
        pulse_end_idx = pulse_start_idx + pulse_samples

        if pulse_start_idx >= 0 and pulse_end_idx <= n_timesteps:
            input_sequence[pulse_start_idx:pulse_end_idx] = pulse_height
        elif pulse_start_idx < n_timesteps and pulse_start_idx >= 0:
            input_sequence[pulse_start_idx:min(pulse_end_idx, n_timesteps)] = pulse_height

    # === Generate TARGET pulses (sync + continuation phases only) ===

    # Sync phase targets (skip first N if specified)
    for i, beat_time in enumerate(sync_beat_times):
        if i < skip_first_n_sync:
            continue

        # Target ends at beat time (prediction before the beat)
        target_end_time = beat_time
        target_end_idx = int(target_end_time / dt)
        target_start_idx = target_end_idx - len(target_curve)

        if target_start_idx >= 0 and target_end_idx <= n_timesteps:
            target_sequence[target_start_idx:target_end_idx] = target_curve + (-output_offset)
        elif target_start_idx < n_timesteps and target_end_idx > 0:
            valid_start = max(0, target_start_idx)
            valid_end = min(n_timesteps, target_end_idx)
            curve_start = max(0, -target_start_idx)
            curve_end = curve_start + (valid_end - valid_start)
            if curve_end <= len(target_curve):
                target_sequence[valid_start:valid_end] = target_curve[curve_start:curve_end] + (-output_offset)

    # Continuation phase targets (deterministic)
    continuation_end_idx = 0
    for beat_time in continuation_beat_times:
        target_end_time = beat_time
        target_end_idx = int(target_end_time / dt)
        target_start_idx = target_end_idx - len(target_curve)

        if target_start_idx >= 0 and target_end_idx <= n_timesteps:
            target_sequence[target_start_idx:target_end_idx] = target_curve + (-output_offset)
            continuation_end_idx = target_end_idx
        elif target_start_idx < n_timesteps and target_end_idx > 0:
            valid_start = max(0, target_start_idx)
            valid_end = min(n_timesteps, target_end_idx)
            curve_start = max(0, -target_start_idx)
            curve_end = curve_start + (valid_end - valid_start)
            if curve_end <= len(target_curve):
                target_sequence[valid_start:valid_end] = target_curve[curve_start:curve_end] + (-output_offset)
                continuation_end_idx = valid_end

    # If no continuation, end after last sync target
    if n_continuation == 0 and len(sync_beat_times) > 0:
        last_sync_time = sync_beat_times[-1]
        continuation_end_idx = int(last_sync_time / dt) + pulse_samples

    # Calculate phase boundary indices for visualization
    if attention_beat_times:
        attention_end_idx = int(attention_beat_times[-1] / dt) + pulse_samples
    else:
        attention_end_idx = 0

    if sync_beat_times:
        sync_end_idx = int(sync_beat_times[-1] / dt) + pulse_samples
    else:
        sync_end_idx = attention_end_idx

    # Calculate where skipped sync pulses end
    if skip_first_n_sync > 0 and len(sync_beat_times) >= skip_first_n_sync:
        # End of the last skipped sync pulse
        last_skipped_sync_time = sync_beat_times[skip_first_n_sync - 1]
        skipped_sync_end_idx = int(last_skipped_sync_time / dt) + pulse_samples
    else:
        skipped_sync_end_idx = attention_end_idx  # No skipped sync pulses

    phase_info["attention_end_idx"] = attention_end_idx
    phase_info["sync_end_idx"] = sync_end_idx
    phase_info["continuation_end_idx"] = continuation_end_idx
    phase_info["skipped_sync_end_idx"] = skipped_sync_end_idx  # <-- ADD THIS

    return input_sequence, target_sequence, continuation_end_idx, phase_info


def create_batch_from_config(
        config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int], Optional[List[Dict[str, Any]]]]:
    """
    Create a batch of sequences using configuration dictionary.

    Args:
        config: Configuration dictionary containing all parameters

    Returns:
        inputs: Batch of input sequences (batch_size, sequence_length, 1)
        targets: Batch of target sequences (batch_size, sequence_length, 1)
        periods: List of periods used for each sequence
        last_inputs_idx: Index of last pulse in each input sequence (or continuation_end_idx for sync_continuation)
        phase_infos: List of phase info dicts (only for sync_continuation, None otherwise)
    """
    task_type = config["task"].get("task_type", "legacy")

    # New sync-continuation task
    if task_type == "sync_continuation":
        inputs, targets, periods, cont_end_indices, phase_infos = create_batch_sync_continuation(config)
        return inputs, targets, periods, cont_end_indices, phase_infos

    # Legacy behavior - existing code below
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

    # Extract gaussian parameters if present
    gaussian_params = config["task"].get("gaussian_params", {})
    gaussian_length = gaussian_params.get("gaussian_length", None)
    gaussian_sigma = gaussian_params.get("gaussian_sigma", None)
    gaussian_max_height = gaussian_params.get("gaussian_max_height", None)

    # Extract noise parameters if present
    noise_params = config["task"].get("noise_params", {})
    phase_noise_std = noise_params.get("phase_noise_std", 0.01)
    jitter_std = noise_params.get("jitter_std", 0.005)
    min_n_pulses = noise_params.get("min_n_pulses", 3)
    max_n_pulses = noise_params.get("max_n_pulses", 7)

    if target_type == "gaussian_noise":
        inputs, targets, periods, last_inputs_idx = create_batch_gaussian_with_noise(
            batch_size=batch_size,
            min_period=min_period,
            max_period=max_period,
            phase_noise_std=phase_noise_std,
            jitter_std=jitter_std,
            min_n_pulses=min_n_pulses,
            max_n_pulses=max_n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            gaussian_max_height=gaussian_max_height,
            skip_first_n=skip_first_n,
        )
    elif target_type == "gamma":
        inputs, targets, periods, last_inputs_idx = create_batch_gamma(
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
    elif target_type == "rectangle":
        inputs, targets, periods, last_inputs_idx = create_batch_rectangle(
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
        raise ValueError(f"Unknown target type: {target_type}")

    return inputs, targets, periods, last_inputs_idx, None


def generate_test_sequences_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate test sequences using configuration dictionary.

    Args:
        config: Configuration dictionary containing all parameters

    Returns:
        Dictionary containing test data organized by period
    """
    # Check for sync-continuation task type
    task_type = config["task"].get("task_type", "legacy")
    if task_type == "sync_continuation":
        return generate_test_sequences_sync_continuation(config)

    # Legacy behavior - existing code below
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

    # Extract gaussian parameters if present
    gaussian_params = config["task"].get("gaussian_params", {})
    gaussian_length = gaussian_params.get("gaussian_length", None)
    gaussian_sigma = gaussian_params.get("gaussian_sigma", None)
    gaussian_max_height = gaussian_params.get("gaussian_max_height", None)

    # Extract noise parameters if present
    noise_params = config["task"].get("noise_params", {})
    phase_noise_std = noise_params.get("phase_noise_std", 0.01)
    jitter_std = noise_params.get("jitter_std", 0.005)
    min_n_pulses = noise_params.get("min_n_pulses", 3)
    max_n_pulses = noise_params.get("max_n_pulses", 7)

    if target_type == "gaussian_noise":
        return generate_test_sequences_gaussian_with_noise(
            n_periods=n_periods,
            min_period=min_period,
            max_period=max_period,
            trials_per_period=trials_per_period,
            phase_noise_std=phase_noise_std,
            jitter_std=jitter_std,
            min_n_pulses=min_n_pulses,
            max_n_pulses=max_n_pulses,
            pulse_width=pulse_width,
            pulse_height=pulse_height,
            dt=dt,
            sequence_length=sequence_length,
            baseline_value=baseline_value,
            output_offset=output_offset,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            gaussian_max_height=gaussian_max_height,
            skip_first_n=skip_first_n,
        )
    elif target_type == "gamma":
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
    elif target_type == "rectangle":
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
        raise ValueError(f"Unknown target type: {target_type}")


def generate_test_sequences_sync_continuation(
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate test sequences for sync-continuation task.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary containing test data organized by period
    """
    trials_per_period = config["testing"]["test_trials_per_period"]
    task_config = config["task"]

    # Create test periods based on config
    if "period_step" in task_config:
        # Use half the training step for testing (to test interpolation)
        min_p = task_config["min_period"]
        max_p = task_config["max_period"]
        test_step = task_config.get("test_period_step", task_config["period_step"] / 2)
        test_periods = np.arange(min_p, max_p + test_step / 2, test_step)
    elif "period_values" in task_config:
        # If discrete values, test on those plus midpoints
        train_periods = np.array(sorted(task_config["period_values"]))
        midpoints = (train_periods[:-1] + train_periods[1:]) / 2
        test_periods = np.sort(np.concatenate([train_periods, midpoints]))
    else:
        # Default: evenly spaced
        n_periods = config["testing"].get("n_test_periods", 9)
        min_p = task_config["min_period"]
        max_p = task_config["max_period"]
        test_periods = np.linspace(min_p, max_p, n_periods)

    test_data = {
        "periods": test_periods.tolist(),
        "sequences": [],
        "task_type": "sync_continuation",
    }

    # Extract all parameters
    attention_phase = task_config.get("attention_phase", {})
    min_n_attention = attention_phase.get("min_n_pulses", 2)
    max_n_attention = attention_phase.get("max_n_pulses", 4)

    sync_phase = task_config.get("sync_phase", {})
    min_n_sync = sync_phase.get("min_n_pulses", 4)
    max_n_sync = sync_phase.get("max_n_pulses", 6)

    continuation_phase = task_config.get("continuation_phase", {})
    min_n_continuation = continuation_phase.get("min_n_pulses", 2)
    max_n_continuation = continuation_phase.get("max_n_pulses", 4)

    pulse_width = task_config["pulse_width"]
    pulse_height = task_config["pulse_height"]
    dt = task_config["dt"]
    sequence_length = task_config["sequence_length"]
    baseline_value = task_config.get("baseline_value", 0.0)
    output_offset = task_config.get("output_offset", 0.0)
    target_shape = task_config.get("target_shape", "gaussian")

    noise_params = task_config.get("noise_params", {})
    phase_noise_std = noise_params.get("phase_noise_std", 0.0)
    jitter_std = noise_params.get("jitter_std", 0.0)

    gamma_params = task_config.get("gamma_params", {})
    gamma_length = gamma_params.get("gamma_length", None)
    gamma_shape = gamma_params.get("gamma_shape", 2.5)
    gamma_max_height = gamma_params.get("gamma_max_height", None)

    gaussian_params = task_config.get("gaussian_params", {})
    gaussian_length = gaussian_params.get("gaussian_length", None)
    gaussian_sigma = gaussian_params.get("gaussian_sigma", None)
    gaussian_max_height = gaussian_params.get("gaussian_max_height", None)

    skip_first_n_sync = task_config.get("skip_first_n_sync", 0)

    for period in test_periods:
        period_data = {
            "period": float(period),
            "inputs": [],
            "targets": [],
            "continuation_end_indices": [],
            "phase_infos": [],
        }

        for _ in range(trials_per_period):
            input_seq, target_seq, cont_end_idx, phase_info = generate_beat_sequence_sync_continuation(
                period=float(period),
                min_n_attention=min_n_attention,
                max_n_attention=max_n_attention,
                min_n_sync=min_n_sync,
                max_n_sync=max_n_sync,
                min_n_continuation=min_n_continuation,
                max_n_continuation=max_n_continuation,
                pulse_width=pulse_width,
                pulse_height=pulse_height,
                dt=dt,
                sequence_length=sequence_length,
                phase_noise_std=phase_noise_std,
                jitter_std=jitter_std,
                target_shape=target_shape,
                baseline_value=baseline_value,
                output_offset=output_offset,
                gamma_length=gamma_length,
                gamma_shape=gamma_shape,
                gamma_max_height=gamma_max_height,
                gaussian_length=gaussian_length,
                gaussian_sigma=gaussian_sigma,
                gaussian_max_height=gaussian_max_height,
                skip_first_n_sync=skip_first_n_sync,
            )

            period_data["inputs"].append(input_seq)
            period_data["targets"].append(target_seq)
            period_data["continuation_end_indices"].append(cont_end_idx)
            period_data["phase_infos"].append(phase_info)

        period_data["inputs"] = np.array(period_data["inputs"])
        period_data["targets"] = np.array(period_data["targets"])

        test_data["sequences"].append(period_data)

    return test_data

