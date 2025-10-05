"""
LSTM model for beat prediction task.
"""

from typing import Tuple
import torch
import torch.nn as nn


class BeatPredictionLSTM(nn.Module):
    """
    Simple LSTM model for predicting beats.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.0
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Size of input features (1 for our task)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output (1 for our task)
            dropout: Dropout rate between LSTM layers
        """
        super(BeatPredictionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            x: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Initial hidden state tuple (h0, c0)

        Returns:
            output: Predictions of shape (batch_size, sequence_length, output_size)
            hidden: Final hidden state tuple
        """
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply output layer to each time step
        output = self.output_layer(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h0, c0) hidden states
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0

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
