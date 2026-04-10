"""
LoRA utilities for DeepSeek CSI model fine-tuning.
"""
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from peft.util import find_embedding_exportable_modules
import torch
import torch.nn as nn


class LoRAConfig:
    """LoRA configuration for CSI model fine-tuning."""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        bias: str = "none",
        task_type: str = "FEATURE_EXTRACTION",
    ):
        if target_modules is None:
            # Default target modules for DeepSeek decoder
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias
        self.task_type = task_type

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )


def setup_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Set up LoRA for the given model.

    Args:
        model: The base model (DeepSeek CSI model)
        config: LoRA configuration

    Returns:
        Model with LoRA adapters injected
    """
    peft_config = config.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only the trainable LoRA parameters from the model."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model.
    After merging, LoRA weights are incorporated into the base weights,
    and the model can be saved as a regular model.
    """
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    return model


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights."""
    state_dict = get_lora_state_dict(model)
    torch.save(state_dict, path)


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA weights into the model."""
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model
