"""
Models package for LSTM beat prediction.

Contains neural network models for predicting beats in time series data.
"""

from .lstm_model import BeatPredictionLSTM

__all__ = [
    'BeatPredictionLSTM'
]
