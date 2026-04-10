"""
Download DeepSeek model for offline use.

This script downloads the DeepSeek-LLM-7B model from HuggingFace
and saves it locally for offline inference.
"""
import argparse
import os
from pathlib import Path


def download_deepseek_model(
    model_name: str = "deepseek-ai/deepseek-llm-7b-base",
    output_dir: str = "./models/deepseek-7b",
    token: str = None,
):
    """
    Download DeepSeek model for offline use.

    Args:
        model_name: HuggingFace model name
        output_dir: Local directory to save the model
        token: HuggingFace token for gated models
    """
    from huggingface_hub import snapshot_download
    import os

    print(f"Downloading {model_name} to {output_dir}...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download model files
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"\nModel downloaded successfully to {output_dir}")
        print(f"Files: {list(output_path.iterdir())}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative: Manually download from HuggingFace:")
        print(f"  huggingface-cli download {model_name} --local {output_dir}")


def verify_model(model_dir: str) -> bool:
    """
    Verify that the model files are present and valid.

    Args:
        model_dir: Path to the model directory

    Returns:
        True if model is valid, False otherwise
    """
    from transformers import AutoConfig
    from pathlib import Path

    model_path = Path(model_dir)

    # Check for essential files
    essential_files = ["config.json"]
    optional_files = ["model.safetensors", "pytorch_model.bin", "model.bin"]

    has_config = any(f in [f.name for f in model_path.iterdir()] for f in essential_files)
    has_weights = any(f in [f.name for f in model_path.iterdir()] for f in optional_files)

    if not has_config:
        print(f"ERROR: config.json not found in {model_dir}")
        return False

    if not has_weights:
        print(f"WARNING: No model weights found in {model_dir}")

    # Try to load config
    try:
        config = AutoConfig.from_pretrained(model_dir)
        print(f"Model verified: {config.model_type}")
        return True
    except Exception as e:
        print(f"ERROR: Could not load model config: {e}")
        return False


def download_with_accelerate(
    model_name: str = "deepseek-ai/deepseek-llm-7b-base",
    output_dir: str = "./models/deepseek-7b",
):
    """
    Alternative download using accelerate's disk offloading setup.

    This is useful for models that don't fit in RAM.
    """
    from accelerate import init_empty_weights, download_checkpoint
    import torch

    print("Setting up model with disk offloading...")

    with init_empty_weights():
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    # Download checkpoint
    download_checkpoint(model_name, output_dir, force=True)
    print(f"Model downloaded to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DeepSeek model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/deepseek-7b",
        help="Output directory for model",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (for gated models)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing model",
    )

    args = parser.parse_args()

    if args.verify:
        verify_model(args.output_dir)
    else:
        download_deepseek_model(args.model_name, args.output_dir, args.token)
