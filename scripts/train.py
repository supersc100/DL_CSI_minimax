"""
Training script for CSI feedback model.

Example usage:
    python scripts/train.py --config config/csi_config.yaml
    python scripts/train.py --config config/csi_config.yaml --max_steps 100 --eval_every 50
"""
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepseek_csi_model import DeepSeekCSIModel
from models.lora_utils import LoRAConfig, setup_lora
from training.trainer import CSITrainer
from data.csi_dataset import create_csi_dataloaders


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict, device: torch.device) -> nn.Module:
    """Initialize and set up the CSI model."""
    print("=" * 60)
    print("Setting up CSI Feedback Model")
    print("=" * 60)

    # Get model configuration
    model_cfg = config.get('model', {})
    model_path = model_cfg.get('deepseek_model_path', './models/deepseek-7b')

    print(f"Loading DeepSeek model from: {model_path}")

    # Create CSI model
    model = DeepSeekCSIModel(
        model_path=model_path,
        hidden_dim=model_cfg.get('hidden_dim', 4096),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        output_dim=2,
        use_flash_attention=model_cfg.get('use_flash_attention', False),
    )

    # Move to device
    model = model.to(device)

    # Setup LoRA
    lora_cfg = config.get('lora', {})
    if lora_cfg.get('enabled', True):
        print("\nSetting up LoRA fine-tuning...")
        lora_config = LoRAConfig(
            r=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.05),
        )
        model = setup_lora(model, lora_config)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train CSI Feedback Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/csi_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing CSI data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for model outputs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Evaluation frequency in steps (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.eval_every:
        config['training']['eval_every'] = args.eval_every

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Setup model
    model = setup_model(config, device)

    # Create data loaders
    print("\n" + "=" * 60)
    print("Loading CSI Data")
    print("=" * 60)

    train_loader, test_loader = create_csi_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 2),
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Create trainer
    trainer = CSITrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer.fit()


if __name__ == "__main__":
    main()
