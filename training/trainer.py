"""
CSI Trainer for PyTorch training loop.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import sys
import random
import numpy as np


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Mean Square Error in dB.

    Args:
        pred: Predicted CSI [batch, seq_len, 2] (real, imag)
        target: Target CSI [batch, seq_len, 2] (real, imag)

    Returns:
        NMSE in dB (lower is better)
    """
    mse = torch.sum((pred - target) ** 2)
    norm = torch.sum(target ** 2)
    nmse_linear = mse / (norm + 1e-10)
    return 10 * torch.log10(nmse_linear + 1e-10)


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Setup logger that outputs to both file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)

    # Console handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    max_steps: Optional[int] = None
    seed: int = 42


class CSITrainer:
    """
    Trainer for CSI feedback model.

    Handles the training loop, evaluation, checkpointing, and logging.
    Supports gradient accumulation and warmup scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: str = "./outputs",
    ):
        """
        Args:
            model: The CSI model to train
            train_loader: Training data loader
            test_loader: Test data loader
            config: Full configuration dictionary
            device: Device to train on
            output_dir: Directory for checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger (both file and console)
        log_file = self.output_dir / "training.log"
        self.logger = setup_logger("csi_trainer", str(log_file))

        # Parse training config
        train_cfg = config.get('training', {})
        self.train_config = TrainingConfig(
            epochs=train_cfg.get('epochs', 10),
            batch_size=train_cfg.get('batch_size', 32),
            gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
            learning_rate=train_cfg.get('learning_rate', 1e-4),
            weight_decay=train_cfg.get('weight_decay', 0.01),
            warmup_ratio=train_cfg.get('warmup_ratio', 0.0),
            grad_clip=train_cfg.get('grad_clip', 1.0),
            eval_every=train_cfg.get('eval_every', 100),
            save_every=train_cfg.get('save_every', 500),
            log_every=train_cfg.get('log_every', 10),
            max_steps=train_cfg.get('max_steps'),
            seed=train_cfg.get('seed', 42),
        )

        # Set random seed for reproducibility
        set_seed(self.train_config.seed)
        self.logger.info(f"Random seed set to {self.train_config.seed}")

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler (with warmup)
        self.scheduler = self._create_scheduler()

        # Setup loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        # Use different learning rates for LoRA and non-LoRA params
        lora_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "lora_" in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": float(self.train_config.learning_rate) * 0.1},
            {"params": lora_params, "lr": float(self.train_config.learning_rate)},
        ]

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.train_config.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        try:
            from transformers import get_cosine_schedule_with_warmup
        except ImportError:
            self.logger.warning("transformers not available, falling back to CosineAnnealingLR without warmup")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.epochs * len(self.train_loader),
            )

        accumulation_steps = max(1, self.train_config.gradient_accumulation_steps)
        total_steps = self.train_config.epochs * len(self.train_loader) // accumulation_steps
        warmup_steps = int(total_steps * self.train_config.warmup_ratio)

        self.logger.info(
            f"Scheduler: cosine with warmup | total_steps={total_steps} | "
            f"warmup_steps={warmup_steps} | grad_accum={accumulation_steps}"
        )

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def fit(self):
        """Run the full training loop."""
        self.logger.info(f"Starting training for {self.train_config.epochs} epochs")
        accumulation_steps = max(1, self.train_config.gradient_accumulation_steps)
        total_opt_steps = self.train_config.epochs * len(self.train_loader) // accumulation_steps
        self.logger.info(f"Total optimization steps: {total_opt_steps}")

        for epoch in range(self.train_config.epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            # Evaluate at end of epoch
            test_mse, test_nmse_db = self.evaluate()

            self.logger.info(
                f"Epoch {epoch+1}/{self.train_config.epochs} | "
                f"Train Loss: {train_loss:.6f} | Test MSE: {test_mse:.6f} | Test NMSE: {test_nmse_db:.2f} dB"
            )

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                self.logger.info(f"Reached max steps ({self.train_config.max_steps})")
                break

        self.logger.info("Training complete!")

    def train_epoch(self) -> float:
        """Run one training epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = max(1, self.train_config.gradient_accumulation_steps)
        accumulated_loss = 0.0

        # Zero gradients at the start
        self.optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle both old format (dl_csi, ul_csi) and new format (dl_csi, ul_csi, env_info)
            if len(batch_data) == 3:
                dl_csi, ul_csi, env_info = batch_data
            else:
                dl_csi, ul_csi = batch_data
                env_info = None

            # Move to device
            dl_csi = dl_csi.to(self.device)
            ul_csi = ul_csi.to(self.device)
            if env_info is not None:
                env_info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in env_info.items()}

            # Forward pass
            pred_ul_csi = self.model(dl_csi, env_info=env_info)

            # Compute loss (scale for gradient accumulation)
            loss = self.criterion(pred_ul_csi, ul_csi)
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Accumulate loss for logging (unscaled)
            accumulated_loss += loss.item()
            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            # Update weights only after accumulation_steps mini-batches (or at epoch end)
            is_accumulation_end = (batch_idx + 1) % accumulation_steps == 0
            is_epoch_end = (batch_idx + 1) == len(self.train_loader)

            if is_accumulation_end or is_epoch_end:
                # Gradient clipping
                if self.train_config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.grad_clip
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging (based on optimization steps)
                if self.global_step % self.train_config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    avg_loss = accumulated_loss / min(accumulation_steps, batch_idx + 1)
                    self.logger.info(
                        f"  Step {self.global_step} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"LR: {lr:.2e}"
                    )
                    accumulated_loss = 0.0

                # Evaluation
                if self.global_step % self.train_config.eval_every == 0:
                    test_mse, test_nmse_db = self.evaluate()
                    self.model.train()
                    self.logger.info(f"  [Eval] Test MSE: {test_mse:.6f} | Test NMSE: {test_nmse_db:.2f} dB")

                # Save checkpoint
                if self.global_step % self.train_config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                # Max steps limit
                if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                    break

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on test set.

        Returns:
            Tuple of (mse_loss, nmse_db)
        """
        self.model.eval()
        total_mse = 0.0
        total_nmse_db = 0.0
        num_batches = 0

        for batch_data in self.test_loader:
            # Handle both old format (dl_csi, ul_csi) and new format (dl_csi, ul_csi, env_info)
            if len(batch_data) == 3:
                dl_csi, ul_csi, env_info = batch_data
            else:
                dl_csi, ul_csi = batch_data
                env_info = None

            dl_csi = dl_csi.to(self.device)
            ul_csi = ul_csi.to(self.device)
            if env_info is not None:
                env_info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in env_info.items()}

            pred_ul_csi = self.model(dl_csi, env_info=env_info)
            loss = self.criterion(pred_ul_csi, ul_csi)
            nmse = nmse_db(pred_ul_csi, ul_csi)

            total_mse += loss.item()
            total_nmse_db += nmse.item()
            num_batches += 1

        return total_mse / num_batches, total_nmse_db / num_batches

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]

        self.logger.info(f"Loaded checkpoint from {path}")
