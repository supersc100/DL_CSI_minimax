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
import logging
import random
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepseek_csi_model import DeepSeekCSIModel
from models.lora_utils import LoRAConfig, setup_lora
from training.trainer import CSITrainer
from data.csi_dataset import CSIDataLoader


def setup_logger(log_file: str) -> logging.Logger:
    """Setup logger that outputs to both file and stdout."""
    logger = logging.getLogger("train_script")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Console handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict, device: torch.device, logger: logging.Logger) -> nn.Module:
    """Initialize and set up the CSI model."""
    logger.info("=" * 60)
    logger.info("Setting up CSI Feedback Model")
    logger.info("=" * 60)

    # Get model configuration
    model_cfg = config.get('model', {})
    model_path = model_cfg.get('model_path', './models/deepseek-1_5b')

    # Get environment configuration
    env_cfg = config.get('environment', {})
    use_env_info = env_cfg.get('enabled', False)
    env_phase_dim = env_cfg.get('phase_dim', 20)
    env_angles_dim = env_cfg.get('angles_delays_dim', 20)
    env_cov_dim = env_cfg.get('cov_dim', 1024)

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Environment info enabled: {use_env_info}")

    # Create CSI model
    model = DeepSeekCSIModel(
        model_path=model_path,
        hidden_dim=model_cfg.get('hidden_dim', 4096),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        output_dim=2,
        use_flash_attention=model_cfg.get('use_flash_attention', False),
        use_environment_info=use_env_info,
        env_phase_dim=env_phase_dim,
        env_angles_dim=env_angles_dim,
        env_cov_dim=env_cov_dim,
    )

    # Move to device
    model = model.to(device)

    # Setup LoRA
    lora_cfg = config.get('lora', {})
    if lora_cfg.get('enabled', True):
        logger.info("Setting up LoRA fine-tuning...")
        target_modules = lora_cfg.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_config = LoRAConfig(
            r=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.05),
            target_modules=target_modules,
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

    # Create output directory first for logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(str(output_dir / "training.log"))

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.eval_every:
        config['training']['eval_every'] = args.eval_every

    # Set random seed for reproducibility (before model creation)
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    logger.info(f"\nUsing device: {device}")

    # Setup model
    model = setup_model(config, device, logger)

    # Create data loaders
    logger.info("\n" + "=" * 60)
    logger.info("Loading CSI Data")
    logger.info("=" * 60)

    # Always try to load env_info if present in data file;
    # environment.enabled controls whether the model uses it.
    from data.csi_dataset import CSIDataLoader
    data_loader = CSIDataLoader(
        train_file=f"{args.data_dir}/csi_data_train.h5",
        test_file=f"{args.data_dir}/csi_data_test.h5",
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 2),
        load_env_info=True,  # Auto-detect from HDF5 file
    )
    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

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
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    trainer.fit()


if __name__ == "__main__":
    main()
