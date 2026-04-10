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
            {"params": other_params, "lr": self.train_config.learning_rate * 0.1},
            {"params": lora_params, "lr": self.train_config.learning_rate},
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
        print(f"Starting training for {self.train_config.epochs} epochs")
        print(f"Total training steps: {len(self.train_loader) * self.train_config.epochs}")

        for epoch in range(self.train_config.epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            # Evaluate at end of epoch
            test_loss = self.evaluate()

            print(
                f"Epoch {epoch+1}/{self.train_config.epochs} | "
                f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
            )

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                print(f"Reached max steps ({self.train_config.max_steps})")
                break

        print("\nTraining complete!")

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (dl_csi, ul_csi) in enumerate(self.train_loader):
            # Move to device
            dl_csi = dl_csi.to(self.device)
            ul_csi = ul_csi.to(self.device)

            # Forward pass
            pred_ul_csi = self.model(dl_csi)

            # Compute loss
            loss = self.criterion(pred_ul_csi, ul_csi)

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
                print(
                    f"  Step {self.global_step} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {lr:.2e}"
                )

            # Evaluation
            if self.global_step % self.train_config.eval_every == 0:
                test_loss = self.evaluate()
                self.model.train()
                print(f"  [Eval] Test Loss: {test_loss:.6f}")

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

        for dl_csi, ul_csi in self.test_loader:
            dl_csi = dl_csi.to(self.device)
            ul_csi = ul_csi.to(self.device)

            pred_ul_csi = self.model(dl_csi)
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
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]

        print(f"Loaded checkpoint from {path}")
