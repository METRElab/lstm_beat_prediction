"""
Training utilities with feedback loop support for LSTM beat prediction.

Provides FeedbackTrainer which extends the base Trainer with step-by-step
training to support feedback loop during training.
"""

from typing import Dict, Any
import logging
from pathlib import Path
import torch

from .trainer import Trainer
from models import FeedbackLSTM
from data import create_batch_from_config


class FeedbackTrainer(Trainer):
    """
    Trainer with feedback loop support.

    Extends the base Trainer to use step-by-step forward pass when feedback
    is enabled, allowing the model to receive its own delayed output as input.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: FeedbackLSTM,
        experiment_path: str,
        logger: logging.Logger
    ):
        """
        Initialize feedback trainer.

        Args:
            config: Configuration dictionary
            model: FeedbackLSTM model to train
            experiment_path: Path to save experiment results
            logger: Logger instance
        """
        super().__init__(config, model, experiment_path, logger)

        self.feedback_config = config.get('feedback', {})
        self.feedback_enabled = self.feedback_config.get('enabled', False)
        self.dt = config['task']['dt']

        if self.feedback_enabled:
            self.logger.info(
                f"Feedback enabled: threshold={self.feedback_config.get('threshold')}, "
                f"delay={self.feedback_config.get('delay')}s"
            )

    def train_epoch(self) -> float:
        """
        Train for one epoch with feedback support.

        When feedback is disabled, uses the standard batch training.
        When feedback is enabled, uses step-by-step training with feedback injection.

        Returns:
            Average training loss for the epoch
        """
        if not self.feedback_enabled:
            return super().train_epoch()

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Calculate number of batches
        trials_per_epoch = self.config['training']['trials_per_epoch']
        batch_size = self.config['training']['batch_size']
        n_batches_total = trials_per_epoch // batch_size

        for _ in range(n_batches_total):
            # Generate batch
            batch_result = create_batch_from_config(self.config)
            inputs, targets, periods, end_indices, phase_infos = batch_result

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Get continuation start indices for decay (if applicable)
            continuation_start_indices = None
            task_type = self.config.get("task", {}).get("task_type", "legacy")
            if task_type == "sync_continuation" and phase_infos is not None:
                continuation_start_indices = [
                    phase_info.get("sync_end_idx", 0) for phase_info in phase_infos
                ]

            # Forward pass with feedback (step-by-step)
            outputs, _, feedback_signal = self.model.forward_with_feedback(
                inputs, self.dt, continuation_start_idx=continuation_start_indices
            )

            # Get masking options from config
            task_config = self.config.get("task", {})
            ignore_attention = task_config.get("ignore_attention_error", False)
            ignore_skipped_sync = task_config.get("ignore_skipped_sync_error", False)
            ignore_tail = task_config.get("ignore_tail_error", False)
            task_type = task_config.get("task_type", "legacy")

            # Apply masking for sync_continuation task
            if task_type == "sync_continuation" and phase_infos is not None:
                for i, phase_info in enumerate(phase_infos):
                    # Mask attention phase
                    if ignore_attention:
                        attention_end_idx = phase_info.get("attention_end_idx", 0)
                        if attention_end_idx > 0:
                            outputs[i, :attention_end_idx, :] = targets[i, :attention_end_idx, :].detach()

                    # Mask skipped sync pulses
                    if ignore_skipped_sync:
                        attention_end_idx = phase_info.get("attention_end_idx", 0)
                        skipped_sync_end_idx = phase_info.get("skipped_sync_end_idx", attention_end_idx)
                        if skipped_sync_end_idx > attention_end_idx:
                            outputs[i, attention_end_idx:skipped_sync_end_idx, :] = targets[
                                i, attention_end_idx:skipped_sync_end_idx, :].detach()

                    # Mask tail after continuation
                    if ignore_tail:
                        continuation_end_idx = phase_info.get("continuation_end_idx", outputs.shape[1])
                        outputs[i, continuation_end_idx:, :] = targets[i, continuation_end_idx:, :].detach()

            # Legacy behavior for non-sync_continuation tasks
            elif ignore_tail and end_indices is not None:
                for i, end_idx in enumerate(end_indices):
                    outputs[i, end_idx:, :] = targets[i, end_idx:, :].detach()

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
        Validate the model with feedback support.

        Returns:
            Average validation loss
        """
        if not self.feedback_enabled:
            return super().validate()

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        validation_trials = self.config['training']['validation_trials']
        batch_size = self.config['training']['batch_size']
        n_batches_total = validation_trials // batch_size

        with torch.no_grad():
            for _ in range(n_batches_total):
                # Generate validation batch
                batch_result = create_batch_from_config(self.config)
                inputs, targets, periods, end_indices, phase_infos = batch_result

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Get continuation start indices for decay (if applicable)
                continuation_start_indices = None
                task_type = self.config.get("task", {}).get("task_type", "legacy")
                if task_type == "sync_continuation" and phase_infos is not None:
                    continuation_start_indices = [
                        phase_info.get("sync_end_idx", 0) for phase_info in phase_infos
                    ]

                # Forward pass with feedback
                outputs, _, feedback_signal = self.model.forward_with_feedback(
                    inputs, self.dt, continuation_start_idx=continuation_start_indices
                )

                # Get masking options from config
                task_config = self.config.get("task", {})
                ignore_attention = task_config.get("ignore_attention_error", False)
                ignore_skipped_sync = task_config.get("ignore_skipped_sync_error", False)
                ignore_tail = task_config.get("ignore_tail_error", False)
                task_type = task_config.get("task_type", "legacy")

                # Apply masking for sync_continuation task
                if task_type == "sync_continuation" and phase_infos is not None:
                    for i, phase_info in enumerate(phase_infos):
                        # Mask attention phase
                        if ignore_attention:
                            attention_end_idx = phase_info.get("attention_end_idx", 0)
                            if attention_end_idx > 0:
                                outputs[i, :attention_end_idx, :] = targets[i, :attention_end_idx, :].detach()

                        # Mask skipped sync pulses
                        if ignore_skipped_sync:
                            attention_end_idx = phase_info.get("attention_end_idx", 0)
                            skipped_sync_end_idx = phase_info.get("skipped_sync_end_idx", attention_end_idx)
                            if skipped_sync_end_idx > attention_end_idx:
                                outputs[i, attention_end_idx:skipped_sync_end_idx, :] = targets[
                                    i, attention_end_idx:skipped_sync_end_idx, :].detach()

                        # Mask tail after continuation
                        if ignore_tail:
                            continuation_end_idx = phase_info.get("continuation_end_idx", outputs.shape[1])
                            outputs[i, continuation_end_idx:, :] = targets[i, continuation_end_idx:, :].detach()

                # Legacy behavior for non-sync_continuation tasks
                elif ignore_tail and end_indices is not None:
                    for i, end_idx in enumerate(end_indices):
                        outputs[i, end_idx:, :] = targets[i, end_idx:, :].detach()

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Track loss
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss
