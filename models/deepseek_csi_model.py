"""
DeepSeek CSI Model for downlink to uplink CSI prediction.
"""
import torch
import torch.nn as nn
from typing import Optional
import math


class CSIEmbedding(nn.Module):
    """Custom embedding layer for CSI data."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4096, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.position_encoder = PositionalEncoding(hidden_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.position_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class CSIRegressionHead(nn.Module):
    """Regression head for CSI prediction."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepSeekCSIModel(nn.Module):
    """Main CSI Feedback Model using frozen DeepSeek as feature extractor."""

    def __init__(self, model_path: str, hidden_dim: int = 4096, max_seq_len: int = 2048,
                 output_dim: int = 2, use_flash_attention: bool = False):
        super().__init__()
        self.model_path = model_path
        self.hidden_dim = hidden_dim

        self.deepseek_model = self._load_deepseek_model(model_path, hidden_dim, use_flash_attention)
        actual_hidden_dim = self.deepseek_model.config.hidden_size

        self.csi_embedding = CSIEmbedding(input_dim=2, hidden_dim=actual_hidden_dim, max_seq_len=max_seq_len)
        self.csi_regression = CSIRegressionHead(actual_hidden_dim)
        self._freeze_deepseek()

    def _load_deepseek_model(self, model_path: str, hidden_dim: int, use_flash_attention: bool):
        from transformers import AutoModelForCausalLM, PretrainedConfig
        print(f"Loading DeepSeek model from: {model_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16,
                attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            )
            print(f"  Model: {model.config.model_type}, Hidden: {model.config.hidden_size}")
            return model
        except FileNotFoundError:
            print(f"  WARNING: Model not found at {model_path}. Using mock config for testing.")
            config = PretrainedConfig(model_type="llama", hidden_size=hidden_dim,
                                      intermediate_size=hidden_dim*4, num_hidden_layers=1,
                                      num_attention_heads=8, vocab_size=32000, max_position_embeddings=2048)
            return AutoModelForCausalLM.from_config(config)

    def _freeze_deepseek(self):
        for param in self.deepseek_model.parameters():
            param.requires_grad = False

    def forward(self, csi_down: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = csi_down.shape
        hidden_states = self.csi_embedding(csi_down)
        position_ids = torch.arange(seq_len, device=csi_down.device).unsqueeze(0).expand(batch_size, -1)

        for layer in self.deepseek_model.model.layers:
            layer_output = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_output[0]

        if hasattr(self.deepseek_model.model, 'norm'):
            hidden_states = self.deepseek_model.model.norm(hidden_states)

        return self.csi_regression(hidden_states)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"\nTrainable: {trainable:,} ({trainable/total*100:.2f}%), Frozen: {frozen:,}")


def create_csi_model(model_path: str, config_path: Optional[str] = None) -> DeepSeekCSIModel:
    import yaml
    hidden_dim, max_seq_len = 4096, 2048
    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            hidden_dim = cfg.get('model', {}).get('hidden_dim', 4096)
            max_seq_len = cfg.get('model', {}).get('max_seq_len', 2048)
    return DeepSeekCSIModel(model_path=model_path, hidden_dim=hidden_dim, max_seq_len=max_seq_len)
