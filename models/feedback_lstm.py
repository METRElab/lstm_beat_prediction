"""
LSTM model with feedback loop support for beat prediction.

Provides FeedbackLSTM which wraps the base BeatPredictionLSTM with
additive feedback injection capability.
"""

from typing import Tuple, Optional, Dict, Any, List, Union
import torch
import torch.nn as nn

from .lstm_model import BeatPredictionLSTM
from .feedback import FeedbackBuffer, FeedbackPulseGenerator, create_feedback_components


class FeedbackLSTM(nn.Module):
    """
    LSTM with integrated feedback loop for beat prediction.

    This model wraps BeatPredictionLSTM and adds a feedback mechanism where:
    - When output exceeds a threshold, a feedback pulse is triggered
    - The pulse is injected back into the input after a configurable delay
    - Feedback is added to the external input (additive mode)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout: float = 0.0,
        feedback_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FeedbackLSTM.

        Args:
            input_size: Size of input features (typically 1)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output (typically 1)
            dropout: Dropout rate between LSTM layers
            feedback_config: Feedback configuration dictionary with keys:
                - enabled: Whether feedback is active
                - threshold: Output value triggering feedback
                - delay: Delay in seconds before feedback injection
                - pulse_shape: 'rectangular', 'gaussian', or 'gamma'
                - pulse_width: Duration of pulse in seconds
                - pulse_height: Amplitude of pulse
                - refractory_period: Minimum time between triggers in seconds
        """
        super().__init__()

        self.feedback_config = feedback_config or {}
        self.feedback_enabled = self.feedback_config.get('enabled', False)

        # Use the same input_size as base model (feedback is additive)
        self.lstm = BeatPredictionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard forward pass without feedback (for compatibility).

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state tuple (h0, c0)

        Returns:
            output: Predictions of shape (batch_size, seq_len, output_size)
            hidden: Final hidden state tuple
        """
        return self.lstm(x, hidden)

    def forward_with_feedback(
        self,
        x: torch.Tensor,
        dt: float,
        continuation_start_idx: Optional[Union[int, List[int], torch.Tensor]] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass with feedback loop (step-by-step).

        Processes the sequence one timestep at a time, checking for threshold
        crossings and injecting delayed feedback pulses.

        Args:
            x: Input tensor (batch_size, seq_len, input_size) - external input
            dt: Time step size in seconds
            continuation_start_idx: Timestep index where continuation phase begins.
                                   Can be a single int (same for all), a list of ints
                                   (one per batch item), or a tensor of shape (batch_size,).
                                   If provided, enables feedback decay in continuation phase.
            hidden: Optional initial hidden state tuple

        Returns:
            outputs: Predictions (batch_size, seq_len, output_size)
            hidden: Final hidden state tuple
            feedback_signal: Generated feedback (batch_size, seq_len, 1)
        """
        if not self.feedback_enabled:
            # No feedback - just run standard forward
            output, hidden = self.lstm(x, hidden)
            feedback_signal = torch.zeros_like(output)
            return output, hidden, feedback_signal

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize feedback components
        buffer, pulse_gen = create_feedback_components(
            config=self.feedback_config,
            dt=dt,
            batch_size=batch_size,
            device=device
        )

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.lstm.init_hidden(batch_size, device)

        # Convert continuation_start_idx to tensor if provided
        cont_start_tensor = None
        if continuation_start_idx is not None:
            if isinstance(continuation_start_idx, int):
                cont_start_tensor = torch.tensor(
                    [continuation_start_idx] * batch_size,
                    dtype=torch.long,
                    device=device
                )
            elif isinstance(continuation_start_idx, list):
                cont_start_tensor = torch.tensor(
                    continuation_start_idx,
                    dtype=torch.long,
                    device=device
                )
            else:
                cont_start_tensor = continuation_start_idx.to(device)

        outputs = []
        feedback_signals = []

        for t in range(seq_len):
            # Update phase information for pulse generator (for decay)
            if cont_start_tensor is not None:
                in_continuation = (t >= cont_start_tensor)
                pulse_gen.set_continuation_phase(in_continuation)

            # Get external input at timestep t
            x_t = x[:, t:t+1, :]  # (batch, 1, input_size)

            # Get delayed feedback from buffer (initially zeros)
            # We push a dummy zero first to get the delayed output
            if t == 0:
                delayed_feedback = torch.zeros(batch_size, 1, device=device)
            else:
                # The feedback from previous timestep's output check
                delayed_feedback = buffer.buffer[0]

            # Combine input with feedback (additive)
            combined_input = x_t + delayed_feedback.unsqueeze(1)

            # Forward through LSTM for single timestep
            output_t, hidden = self.lstm(combined_input, hidden)
            outputs.append(output_t)

            # Check threshold and generate feedback pulse
            feedback_pulse = pulse_gen.step(output_t.squeeze(1))

            # Push new feedback pulse to buffer
            buffer.push(feedback_pulse)
            feedback_signals.append(feedback_pulse)

        # Stack outputs and feedback signals
        outputs = torch.cat(outputs, dim=1)
        feedback_signals = torch.stack(feedback_signals, dim=1)

        return outputs, hidden, feedback_signals

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h0, c0) hidden states
        """
        return self.lstm.init_hidden(batch_size, device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions without returning hidden states.

        Args:
            x: Input tensor

        Returns:
            Predictions
        """
        with torch.no_grad():
            output, _ = self.forward(x)
        return output

    def predict_with_feedback(
        self,
        x: torch.Tensor,
        dt: float,
        continuation_start_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with feedback, without returning hidden states.

        Args:
            x: Input tensor
            dt: Time step size in seconds
            continuation_start_idx: Timestep index where continuation phase begins

        Returns:
            Tuple of (predictions, feedback_signal)
        """
        with torch.no_grad():
            output, _, feedback = self.forward_with_feedback(
                x, dt, continuation_start_idx=continuation_start_idx
            )
        return output, feedback


def create_feedback_model(config: Dict[str, Any]) -> FeedbackLSTM:
    """
    Factory function to create FeedbackLSTM from config.

    Args:
        config: Full configuration dictionary with 'model' and 'feedback' sections

    Returns:
        Configured FeedbackLSTM instance
    """
    model_config = config.get('model', {})
    feedback_config = config.get('feedback', {})

    return FeedbackLSTM(
        input_size=model_config.get('input_size', 1),
        hidden_size=model_config.get('hidden_size', 16),
        num_layers=model_config.get('num_layers', 1),
        output_size=model_config.get('output_size', 1),
        dropout=model_config.get('dropout', 0.0),
        feedback_config=feedback_config
    )
