from typing import Optional
import torch.optim as optim
import logging
from torch.optim.optimizer import Optimizer

class BidirectionalLRScheduler:
    """
    A learning rate scheduler that adjusts LR bidirectionally based on loss changes.

    Increases LR when loss increases (to escape local minima) and decreases it
    when loss improves (for fine-tuning).
    """

    def __init__(
            self,
            optimizer: Optimizer,
            increase_factor: float = 1.5,
            decrease_factor: float = 0.7,
            max_lr: float = 1.0,
            min_lr: float = 1e-7,
            patience_up: int = 3,
            patience_down: int = 5,
            threshold: float = 0.001,
            verbose: bool = False,
            logger: logging.Logger = None
    ) -> None:
        """
        Initialize the bidirectional learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer instance
            increase_factor: Multiplicative factor for increasing LR
            decrease_factor: Multiplicative factor for decreasing LR
            max_lr: Maximum allowed learning rate
            min_lr: Minimum allowed learning rate
            patience_up: Number of epochs to wait before increasing LR
            patience_down: Number of epochs to wait before decreasing LR
            threshold: Minimum relative change in loss to trigger adjustment
            verbose: Whether to print LR changes
            logger: Logger instance
        """
        self.optimizer = optimizer
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience_up = patience_up
        self.patience_down = patience_down
        self.threshold = threshold
        self.verbose = verbose
        self.logger = logger

        self.best_loss = float('inf')
        self.last_loss = float('inf')
        self.wait_count_up = 0
        self.wait_count_down = 0

    def step(self, current_loss: float) -> float:
        """
        Adjust learning rate based on current loss.

        Args:
            current_loss: Current epoch's loss value

        Returns:
            float: Updated learning rate
        """
        current_lr = self.optimizer.param_groups[0]['lr']

        # Check if loss increased significantly
        if current_loss > self.last_loss * (1 + self.threshold):
            self.wait_count_up += 1
            self.wait_count_down = 0

            if self.wait_count_up >= self.patience_up:
                # Increase learning rate
                new_lr = min(current_lr * self.increase_factor, self.max_lr)
                self._set_lr(new_lr)
                self.wait_count_up = 0

                if self.verbose:
                    self.logger.info(f"Loss increased. LR: {current_lr:.6f} -> {new_lr:.6f}")

        # Check if loss decreased significantly
        elif current_loss < self.best_loss * (1 - self.threshold):
            self.wait_count_down += 1
            self.wait_count_up = 0

            if self.wait_count_down >= self.patience_down:
                # Decrease learning rate
                new_lr = max(current_lr * self.decrease_factor, self.min_lr)
                self._set_lr(new_lr)
                self.wait_count_down = 0

                if self.verbose:
                    self.logger.info(f"Loss decreased. LR: {current_lr:.6f} -> {new_lr:.6f}")

        # Update best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss

        self.last_loss = current_loss

        return self.optimizer.param_groups[0]['lr']

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
