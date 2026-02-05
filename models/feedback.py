"""
Feedback loop components for LSTM beat prediction.

Provides FeedbackBuffer for delayed pulse injection and FeedbackPulseGenerator
for threshold-based feedback pulse generation.
"""

from typing import Dict, Any, Optional
from collections import deque
import torch
import numpy as np


class FeedbackBuffer:
    """
    Ring buffer managing delayed feedback pulse injection.

    When a feedback pulse is triggered, it is stored in the buffer and
    released after the configured delay period.
    """

    def __init__(
        self,
        delay: float,
        dt: float,
        batch_size: int,
        device: torch.device
    ):
        """
        Initialize feedback buffer.

        Args:
            delay: Delay in seconds between trigger and injection
            dt: Time step size in seconds
            batch_size: Number of sequences in batch
            device: Torch device for tensors
        """
        self.delay_steps = max(1, int(delay / dt))
        self.batch_size = batch_size
        self.device = device
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset buffer to initial state with zeros."""
        self.buffer = deque(maxlen=self.delay_steps + 1)
        for _ in range(self.delay_steps + 1):
            self.buffer.append(torch.zeros(self.batch_size, 1, device=self.device))

    def push(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        Push new feedback pulse and return delayed pulse.

        Args:
            pulse: New pulse tensor of shape (batch_size, 1)

        Returns:
            Delayed pulse from delay_steps ago, shape (batch_size, 1)
        """
        self.buffer.append(pulse.clone())
        return self.buffer[0]


class FeedbackPulseGenerator:
    """
    Generates feedback pulses based on output threshold crossings.

    When the model output exceeds the threshold, a feedback pulse is generated.
    A refractory period prevents multiple triggers in quick succession.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dt: float,
        device: torch.device
    ):
        """
        Initialize pulse generator.

        Args:
            config: Feedback configuration dictionary with keys:
                - threshold: Output value triggering feedback
                - pulse_shape: 'rectangular', 'gaussian', or 'gamma'
                - pulse_width: Duration of pulse in seconds
                - pulse_height: Amplitude of pulse
                - refractory_period: Minimum time between triggers in seconds
            dt: Time step size in seconds
            device: Torch device for tensors
        """
        self.threshold = config.get('threshold', 0.5)
        self.pulse_shape = config.get('pulse_shape', 'rectangular')
        self.pulse_width = config.get('pulse_width', 0.05)
        self.pulse_width_steps = max(1, int(self.pulse_width / dt))
        self.pulse_height = config.get('pulse_height', 1.0)
        self.refractory_period = config.get('refractory_period', 0.05)
        self.refractory_steps = max(1, int(self.refractory_period / dt))
        self.device = device
        self.dt = dt

        # Pre-compute pulse template
        self.pulse_template = self._create_pulse_template()

        # State tracking (initialized in reset())
        self.active_pulses = None
        self.refractory_counters = None
        self.batch_size = None

    def _create_pulse_template(self) -> torch.Tensor:
        """
        Create pulse shape template.

        Returns:
            Pulse template tensor of shape (pulse_width_steps,)
        """
        if self.pulse_shape == 'rectangular':
            return torch.ones(self.pulse_width_steps, device=self.device) * self.pulse_height

        elif self.pulse_shape == 'gaussian':
            # Gaussian centered at middle of pulse
            t = torch.linspace(0, 1, self.pulse_width_steps, device=self.device)
            sigma = 1.0 / 6.0  # So pulse spans ~3 sigma on each side
            template = self.pulse_height * torch.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
            return template

        elif self.pulse_shape == 'gamma':
            # Gamma-like shape (asymmetric, fast rise, slow decay)
            t = torch.linspace(0, 1, self.pulse_width_steps, device=self.device)
            # Shape parameter for gamma-like curve
            k = 2.0
            template = self.pulse_height * (t ** k) * torch.exp(-k * t / 0.3)
            # Normalize to max height
            if template.max() > 0:
                template = template / template.max() * self.pulse_height
            return template

        else:
            # Default to rectangular
            return torch.ones(self.pulse_width_steps, device=self.device) * self.pulse_height

    def reset(self, batch_size: int):
        """
        Reset state for new sequence.

        Args:
            batch_size: Number of sequences in batch
        """
        self.batch_size = batch_size
        self.active_pulses = torch.zeros(
            batch_size, self.pulse_width_steps, device=self.device
        )
        self.refractory_counters = torch.zeros(batch_size, device=self.device)

    def step(self, output: torch.Tensor) -> torch.Tensor:
        """
        Check if output crosses threshold and generate feedback pulse.

        This method should be called at each timestep. It:
        1. Checks if output exceeds threshold (and not in refractory period)
        2. Starts new pulse if triggered (immediately)
        3. Returns current pulse value for this timestep
        4. Advances pulse and refractory counters for next step

        Args:
            output: Model output tensor of shape (batch_size, 1)

        Returns:
            Current feedback pulse value, shape (batch_size, 1)
        """
        if self.active_pulses is None:
            raise RuntimeError("Must call reset() before step()")

        batch_size = output.shape[0]

        # Check threshold crossing (not in refractory)
        above_threshold = (output.squeeze(-1) >= self.threshold).float()
        not_refractory = (self.refractory_counters <= 0).float()
        should_trigger = above_threshold * not_refractory

        # Add new pulse templates where triggered (before reading current value)
        for i in range(batch_size):
            if should_trigger[i] > 0:
                self.active_pulses[i] = self.pulse_template.clone()
                self.refractory_counters[i] = float(self.refractory_steps)

        # Get current pulse value (now includes newly triggered pulses)
        current_pulse = self.active_pulses[:, 0:1].clone()

        # Shift active pulses left (time progression for next step)
        self.active_pulses = torch.roll(self.active_pulses, -1, dims=1)
        self.active_pulses[:, -1] = 0  # Clear last position

        # Decrement refractory counters
        self.refractory_counters = torch.clamp(self.refractory_counters - 1, min=0)

        return current_pulse


def create_feedback_components(
    config: Dict[str, Any],
    dt: float,
    batch_size: int,
    device: torch.device
) -> tuple:
    """
    Factory function to create feedback buffer and pulse generator.

    Args:
        config: Feedback configuration dictionary
        dt: Time step size in seconds
        batch_size: Number of sequences in batch
        device: Torch device for tensors

    Returns:
        Tuple of (FeedbackBuffer, FeedbackPulseGenerator)
    """
    buffer = FeedbackBuffer(
        delay=config.get('delay', 0.1),
        dt=dt,
        batch_size=batch_size,
        device=device
    )

    pulse_gen = FeedbackPulseGenerator(
        config=config,
        dt=dt,
        device=device
    )
    pulse_gen.reset(batch_size)

    return buffer, pulse_gen
