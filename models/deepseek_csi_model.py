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

    def forward(self, x: torch.Tensor, env_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(x)
        if env_features is not None:
            seq_len = x.shape[1]
            # Broadcast env_features to match sequence length
            env_emb = env_features.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, env_emb], dim=-1)
            x = self.position_encoder(x)
        else:
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
        self.register_buffer('pe', pe.unsqueeze(0).to(torch.bfloat16))

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


class EnvironmentEncoder(nn.Module):
    """
    Encode environmental information (phase, angles/delays, covariance) into feature vectors.
    """

    def __init__(
        self,
        phase_dim: int = 20,
        angles_dim: int = 20,
        cov_dim: int = 1024,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        # Phase encoder: num_paths -> hidden_dim // 4
        self.phase_encoder = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        # Angles/delay encoder: num_dominant * 4 -> hidden_dim // 4
        self.angles_encoder = nn.Sequential(
            nn.Linear(angles_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        # Covariance encoder: ant_size^2 -> hidden_dim // 4
        self.cov_encoder = nn.Sequential(
            nn.Linear(cov_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        # Combined projection: 3 * (hidden_dim // 4) -> hidden_dim
        self.fusion = nn.Linear(hidden_dim // 4 * 3, hidden_dim)

    def forward(self, env_dict: dict) -> torch.Tensor:
        """
        Args:
            env_dict: dict with keys 'phases', 'angles_delays', 'covariance'
                - phases: [batch, num_paths]
                - angles_delays: [batch, num_dominant * 4]
                - covariance: [batch, ant_size, ant_size]

        Returns:
            env_features: [batch, hidden_dim]
        """
        phase_feat = self.phase_encoder(env_dict['phases'])
        angle_feat = self.angles_encoder(env_dict['angles_delays'])
        # Flatten covariance matrix
        cov = env_dict['covariance']
        if cov.dim() == 3:
            batch = cov.shape[0]
            cov = cov.reshape(batch, -1)  # [batch, ant_size * ant_size]
        cov_feat = self.cov_encoder(cov)

        combined = torch.cat([phase_feat, angle_feat, cov_feat], dim=-1)
        return self.fusion(combined)  # [batch, hidden_dim]


class DeepSeekCSIModel(nn.Module):
    """Main CSI Feedback Model using frozen DeepSeek as feature extractor."""

    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 4096,
        max_seq_len: int = 2048,
        output_dim: int = 2,
        use_flash_attention: bool = False,
        use_environment_info: bool = False,
        env_phase_dim: int = 20,
        env_angles_dim: int = 20,
        env_cov_dim: int = 1024,
    ):
        super().__init__()
        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.use_environment_info = use_environment_info

        self.deepseek_model = self._load_deepseek_model(model_path, hidden_dim, use_flash_attention)
        actual_hidden_dim = self.deepseek_model.config.hidden_size

        # Environment encoder (only created when enabled)
        self.env_encoder = None
        if use_environment_info:
            self.env_encoder = EnvironmentEncoder(
                phase_dim=env_phase_dim,
                angles_dim=env_angles_dim,
                cov_dim=env_cov_dim,
                hidden_dim=actual_hidden_dim,
            )

        self.csi_embedding = CSIEmbedding(input_dim=2, hidden_dim=actual_hidden_dim, max_seq_len=max_seq_len)
        self.csi_regression = CSIRegressionHead(actual_hidden_dim)

        # Convert custom layers to bfloat16 to match DeepSeek model dtype
        self.csi_embedding = self.csi_embedding.to(torch.bfloat16)
        self.csi_regression = self.csi_regression.to(torch.bfloat16)
        if self.env_encoder is not None:
            self.env_encoder = self.env_encoder.to(torch.bfloat16)

        self._freeze_deepseek()

    def _load_deepseek_model(self, model_path: str, hidden_dim: int, use_flash_attention: bool):
        from transformers import AutoModelForCausalLM, PretrainedConfig, LlamaConfig
        import os
        print(f"Loading DeepSeek model from: {model_path}")

        # Check if path exists first
        if not os.path.exists(model_path):
            print(f"  WARNING: Model path not found at {model_path}. Using mock config for testing.")
            config = LlamaConfig(
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 4,
                num_hidden_layers=1,
                num_attention_heads=8,
                vocab_size=32000,
                max_position_embeddings=2048,
            )
            model = AutoModelForCausalLM.from_config(config)
            # Ensure model.model.layers exists
            if not hasattr(model.model, 'layers') or model.model.layers is None:
                class MockLayer(nn.Module):
                    def __init__(self, hidden_dim):
                        super().__init__()
                        self.hidden_dim = hidden_dim
                    def forward(self, hidden_states, position_ids=None, **kwargs):
                        return hidden_states
                model.model.layers = nn.ModuleList([MockLayer(hidden_dim)])
            if not hasattr(model.model, 'norm') or model.model.norm is None:
                model.model.norm = nn.Identity()
            return model

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            print(f"  Model: {model.config.model_type}, Hidden: {model.config.hidden_size}")
            return model
        except Exception:
            print(f"  WARNING: Failed to load model. Using mock config for testing.")
            config = LlamaConfig(
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 4,
                num_hidden_layers=1,
                num_attention_heads=8,
                vocab_size=32000,
                max_position_embeddings=2048,
            )
            model = AutoModelForCausalLM.from_config(config)
            if not hasattr(model.model, 'layers') or model.model.layers is None:
                class MockLayer(nn.Module):
                    def __init__(self, hidden_dim):
                        super().__init__()
                        self.hidden_dim = hidden_dim
                    def forward(self, hidden_states, position_ids=None, **kwargs):
                        return hidden_states
                model.model.layers = nn.ModuleList([MockLayer(hidden_dim)])
            if not hasattr(model.model, 'norm') or model.model.norm is None:
                model.model.norm = nn.Identity()
            return model

    def _freeze_deepseek(self):
        for param in self.deepseek_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, env_info: Optional[dict] = None, **kwargs) -> torch.Tensor:
        csi_down = input_ids
        batch_size, seq_len, _ = csi_down.shape

        # Convert input to bfloat16 to match model dtype
        csi_down = csi_down.to(torch.bfloat16)
        # assert csi_down.dtype == torch.bfloat16, f"csi_down dtype: {csi_down.dtype}"

        # Encode environmental info if available and enabled
        env_features = None
        if self.use_environment_info and env_info is not None and self.env_encoder is not None:
            env_features = self.env_encoder(env_info)  # [batch, hidden_dim]
            env_features = env_features.to(csi_down.device, torch.bfloat16)
            # assert env_features.dtype == torch.bfloat16, f"env_features dtype: {env_features.dtype}"

        hidden_states = self.csi_embedding(csi_down, env_features)
        # assert hidden_states.dtype == torch.bfloat16, f"hidden_states after embedding dtype: {hidden_states.dtype}"
        # print(f"After embedding: max={hidden_states.max():.4f}, min={hidden_states.min():.4f}, has_nan={torch.isnan(hidden_states).any()}")

        # All custom layers and inputs use bfloat16 to match DeepSeek model dtype
        position_ids = torch.arange(seq_len, device=csi_down.device).unsqueeze(0).expand(batch_size, -1)

        model_output = self.deepseek_model.model(
            inputs_embeds=hidden_states,
            position_ids=position_ids,
        )
        hidden_states = model_output[0]
        # Model output may be float32 - convert to bfloat16 for regression head
        hidden_states = hidden_states.to(torch.bfloat16)
        # assert hidden_states.dtype == torch.bfloat16, f"hidden_states after model dtype: {hidden_states.dtype}"
        # print(f"After Qwen2: max={hidden_states.max():.4f}, min={hidden_states.min():.4f}, has_nan={torch.isnan(hidden_states).any()}")

        output = self.csi_regression(hidden_states)
        # assert output.dtype == torch.bfloat16, f"output dtype: {output.dtype}"
        # print(f"After regression: max={output.max():.4f}, min={output.min():.4f}, has_nan={torch.isnan(output).any()}")
        # Convert output to float32 for loss computation with target
        return output.to(torch.float32)
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"\nTrainable: {trainable:,} ({trainable/total*100:.2f}%), Frozen: {frozen:,}")


def create_csi_model(
    model_path: str,
    config_path: Optional[str] = None,
    config_dict: Optional[dict] = None,
) -> DeepSeekCSIModel:
    import yaml
    hidden_dim, max_seq_len = 4096, 2048
    use_env_info = False
    env_phase_dim = 20
    env_angles_dim = 20
    env_cov_dim = 1024

    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            hidden_dim = cfg.get('model', {}).get('hidden_dim', 4096)
            max_seq_len = cfg.get('model', {}).get('max_seq_len', 2048)
            env_cfg = cfg.get('environment', {})
            use_env_info = env_cfg.get('enabled', False)
            env_phase_dim = env_cfg.get('phase_dim', 20)
            env_angles_dim = env_cfg.get('angles_delays_dim', 20)
            env_cov_dim = env_cfg.get('cov_dim', 1024)
    elif config_dict:
        hidden_dim = config_dict.get('model', {}).get('hidden_dim', 4096)
        max_seq_len = config_dict.get('model', {}).get('max_seq_len', 2048)
        env_cfg = config_dict.get('environment', {})
        use_env_info = env_cfg.get('enabled', False)
        env_phase_dim = env_cfg.get('phase_dim', 20)
        env_angles_dim = env_cfg.get('angles_delays_dim', 20)
        env_cov_dim = env_cfg.get('cov_dim', 1024)

    return DeepSeekCSIModel(
        model_path=model_path,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        use_environment_info=use_env_info,
        env_phase_dim=env_phase_dim,
        env_angles_dim=env_angles_dim,
        env_cov_dim=env_cov_dim,
    )
