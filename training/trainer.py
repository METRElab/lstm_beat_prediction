"""
Training utilities for LSTM beat prediction model.
"""

from typing import Dict, Any
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import BeatPredictionLSTM
from data import create_batch


class Trainer:
    """
    Handles training, validation, and checkpointing for the LSTM model.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            model: BeatPredictionLSTM,
            experiment_path: str,
            logger: logging.Logger
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            model: LSTM model to train
            experiment_path: Path to save experiment results
            logger: Logger instance
        """
        self.config = config
        self.model = model
        self.experiment_path = Path(experiment_path)
        self.logger = logger

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        # Setup loss function
        self.criterion = nn.MSELoss()

        # Setup paths
        self.checkpoint_dir = self.experiment_path / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(str(self.experiment_path / 'tensorboard'))
        else:
            self.writer = None

        # Training state
        self.best_loss = float('inf')
        self.current_epoch = 0

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Calculate number of batches
        trials_per_epoch = self.config['training']['trials_per_epoch']
        batch_size = self.config['training']['batch_size']
        n_batches_total = trials_per_epoch // batch_size

        for _ in range(n_batches_total):
            # Generate batch
            inputs, targets, _ = create_batch(
                batch_size=batch_size,
                min_period=self.config['task']['min_period'],
                max_period=self.config['task']['max_period'],
                n_pulses=self.config['task']['n_pulses'],
                pulse_width=self.config['task']['pulse_width'],
                pulse_height=self.config['task']['pulse_height'],
                dt=self.config['task']['dt'],
                sequence_length=self.config['task']['sequence_length'],
                baseline_value=self.config['task']['baseline_value']
            )

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs, _ = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        validation_trials = self.config['training']['validation_trials']
        batch_size = self.config['training']['batch_size']
        n_batches_total = validation_trials // batch_size

        with torch.no_grad():
            for _ in range(n_batches_total):
                # Generate validation batch
                inputs, targets, _ = create_batch(
                    batch_size=batch_size,
                    min_period=self.config['task']['min_period'],
                    max_period=self.config['task']['max_period'],
                    n_pulses=self.config['task']['n_pulses'],
                    pulse_width=self.config['task']['pulse_width'],
                    pulse_height=self.config['task']['pulse_height'],
                    dt=self.config['task']['dt'],
                    sequence_length=self.config['task']['sequence_length'],
                    baseline_value=self.config['task']['baseline_value']
                )

                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs, _ = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Track loss
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }

        # Save regular checkpoint
        if not self.config['training']['checkpoint_best_only']:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch} with loss {self.best_loss:.6f}")

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with training history
        """
        n_epochs = self.config['training']['epochs']
        log_interval = self.config['logging']['log_interval']

        history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': []
        }

        self.logger.info(f"Starting training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Store history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['epochs'].append(epoch)

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)

            # Check if best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            # Save checkpoint
            if is_best or (epoch % self.config['logging']['save_interval'] == 0):
                self.save_checkpoint(epoch, val_loss, is_best)

            # Log progress
            if epoch % log_interval == 0 or is_best:
                self.logger.info(
                    f"Epoch {epoch}/{n_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best: {self.best_loss:.6f}"
                )

        # Close tensorboard writer
        if self.writer:
            self.writer.close()

        self.logger.info(f"Training complete. Best validation loss: {self.best_loss:.6f}")

        return history
