"""
CSI Trainer for PyTorch training loop.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Optional
import logging
import sys


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


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    max_steps: Optional[int] = None


class CSITrainer:
    """
    Trainer for CSI feedback model.

    Handles the training loop, evaluation, checkpointing, and logging.
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
            learning_rate=train_cfg.get('learning_rate', 1e-4),
            weight_decay=train_cfg.get('weight_decay', 0.01),
            grad_clip=train_cfg.get('grad_clip', 1.0),
            eval_every=train_cfg.get('eval_every', 100),
            save_every=train_cfg.get('save_every', 500),
            log_every=train_cfg.get('log_every', 10),
            max_steps=train_cfg.get('max_steps'),
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
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
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.epochs * len(self.train_loader),
        )

    def fit(self):
        """Run the full training loop."""
        self.logger.info(f"Starting training for {self.train_config.epochs} epochs")
        self.logger.info(f"Total training steps: {len(self.train_loader) * self.train_config.epochs}")

        for epoch in range(self.train_config.epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            # Evaluate at end of epoch
            test_loss = self.evaluate()

            self.logger.info(
                f"Epoch {epoch+1}/{self.train_config.epochs} | "
                f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
            )

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                self.logger.info(f"Reached max steps ({self.train_config.max_steps})")
                break

        self.logger.info("Training complete!")

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

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

            # Debug: check input data
            # if batch_idx % self.train_config.log_every == 0:
            #     print(f"  Input dl_csi: max={dl_csi.max():.4f}, min={dl_csi.min():.4f}, dtype={dl_csi.dtype}")
            #     print(f"  Input ul_csi: max={ul_csi.max():.4f}, min={ul_csi.min():.4f}, dtype={ul_csi.dtype}")
            #     if env_info:
            #         print(f"  Env phases: max={env_info['phases'].max():.4f}, min={env_info['phases'].min():.4f}")

            # Forward pass
            pred_ul_csi = self.model(dl_csi, env_info=env_info)

            # Check for NaN in prediction
            # if torch.isnan(pred_ul_csi).any() or torch.isinf(pred_ul_csi).any():
            #     print(f"  WARNING: pred has NaN/Inf! Shape: {pred_ul_csi.shape}, max: {pred_ul_csi.max():.4f}, min: {pred_ul_csi.min():.4f}")
            #     print(f"  dl_csi stats: max={dl_csi.max():.4f}, min={dl_csi.min():.4f}")
            #     continue # Skip this batch

            # Compute loss
            loss = self.criterion(pred_ul_csi, ul_csi)
            # print(f"Loss: {loss.item()}, pred has NaN: {torch.isnan(pred_ul_csi).any()}, target has NaN: {torch.isnan(ul_csi).any()}")
            # # Check for NaN in loss
            # if torch.isnan(loss) or torch.isinf(loss):
            #     print(f"  WARNING: Loss is NaN/Inf! pred max={pred_ul_csi.max():.4f}, min={pred_ul_csi.min():.4f}")
            #     print(f"  ul_csi stats: max={ul_csi.max():.4f}, min={ul_csi.min():.4f}")
            #     continue  # Skip this batch

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.train_config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"  Step {self.global_step} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {lr:.2e}"
                )

            # Evaluation
            if self.global_step % self.train_config.eval_every == 0:
                test_loss = self.evaluate()
                self.model.train()
                self.logger.info(f"  [Eval] Test Loss: {test_loss:.6f}")

            # Save checkpoint
            if self.global_step % self.train_config.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Max steps limit
            if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                break

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on test set."""
        self.model.eval()
        total_loss = 0.0
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

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

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
