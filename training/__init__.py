"""
Training package for LSTM beat prediction.

Contains training utilities, optimizers, and training loops for LSTM models.
"""

from .trainer import Trainer
from .lr_scheduler import BidirectionalLRScheduler

__all__ = [
    'Trainer',
    'BidirectionalLRScheduler',
]
