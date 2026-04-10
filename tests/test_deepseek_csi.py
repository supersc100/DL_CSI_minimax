"""
Tests for DeepSeek CSI Model.

Run with:
    python -m pytest tests/test_deepseek_csi.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepseek_csi_model import CSIEmbedding, CSIRegressionHead


class TestCSIEmbedding:
    """Tests for CSI Embedding layer."""

    def test_embedding_output_shape(self):
        """Test that embedding produces correct output shape."""
        batch_size = 4
        seq_len = 128
        input_dim = 2
        hidden_dim = 512

        embedding = CSIEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_seq_len=256
        )

        csi_input = torch.randn(batch_size, seq_len, input_dim)
        output = embedding(csi_input)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_embedding_preserves_seq_len(self):
        """Test that sequence length is preserved through embedding."""
        seq_len = 64
        embedding = CSIEmbedding(input_dim=2, hidden_dim=256, max_seq_len=128)

        csi_input = torch.randn(2, seq_len, 2)
        output = embedding(csi_input)

        assert output.shape[1] == seq_len

    def test_embedding_trainable(self):
        """Test that embedding layer is trainable."""
        embedding = CSIEmbedding(input_dim=2, hidden_dim=256, max_seq_len=128)

        # Check that projection layer has gradients
        csi_input = torch.randn(2, 32, 2)
        output = embedding(csi_input)
        output.sum().backward()

        assert embedding.proj.weight.grad is not None


class TestCSIRegressionHead:
    """Tests for CSI Regression Head."""

    def test_regression_output_shape(self):
        """Test that regression head produces correct output shape."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 512

        regression_head = CSIRegressionHead(hidden_dim=hidden_dim)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        output = regression_head(hidden_states)

        assert output.shape == (batch_size, seq_len, 2)

    def test_regression_single_position(self):
        """Test regression on single position."""
        regression_head = CSIRegressionHead(hidden_dim=256)
        hidden = torch.randn(1, 1, 256)
        output = regression_head(hidden)

        assert output.shape == (1, 1, 2)

    def test_regression_trainable(self):
        """Test that regression head is trainable."""
        regression_head = CSIRegressionHead(hidden_dim=256)
        hidden = torch.randn(2, 32, 256)
        output = regression_head(hidden)
        output.sum().backward()

        assert regression_head.fc.weight.grad is not None


class TestCSIModelMock:
    """Mock tests for full CSI model (without actual DeepSeek)."""

    def test_mock_forward_pass(self):
        """Test forward pass with mocked DeepSeek components."""
        from unittest.mock import MagicMock

        # Create mock DeepSeek model
        mock_deepseek = MagicMock()
        mock_deepseek.config.hidden_size = 512
        mock_deepseek.config.max_position_embeddings = 256
        mock_deepseek.model = MagicMock()

        # Mock embeddings
        mock_embed_tokens = torch.nn.Linear(2, 512)
        mock_lm_head = torch.nn.Linear(512, 2)

        # Create CSI model components
        embedding = CSIEmbedding(input_dim=2, hidden_dim=512, max_seq_len=256)
        regression_head = CSIRegressionHead(hidden_dim=512)

        # Test forward
        csi_input = torch.randn(2, 64, 2)
        embedded = embedding(csi_input)

        # Apply mock transformation (just pass through)
        transformed = embedded  # In real model, this would go through DeepSeek

        # Apply regression head
        output = regression_head(transformed)

        assert output.shape == (2, 64, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
