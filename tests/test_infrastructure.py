"""
Infrastructure validation tests to verify the testing setup works correctly.
"""

import pytest
import torch
from pathlib import Path


class TestInfrastructure:
    """Test suite to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_device, sample_prompt):
        """Test that all basic fixtures are working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert mock_device == "cpu"
        assert isinstance(sample_prompt, str)
        assert len(sample_prompt) > 0
    
    @pytest.mark.unit
    def test_torch_integration(self, mock_torch_tensor):
        """Test that PyTorch integration works in the test environment."""
        assert isinstance(mock_torch_tensor, torch.Tensor)
        assert mock_torch_tensor.shape == (1, 5)
        assert mock_torch_tensor.dtype == torch.long
    
    @pytest.mark.unit
    def test_mock_components(self, mock_tokenizer, mock_model):
        """Test that mock components are properly configured."""
        # Test mock tokenizer
        assert mock_tokenizer.encode("test") == [101, 2003, 4248, 2829, 4419]
        assert "brown fox" in mock_tokenizer.decode([1, 2, 3])
        
        # Test mock model
        mock_output = mock_model()
        assert hasattr(mock_output, 'logits')
        assert isinstance(mock_output.logits, torch.Tensor)
    
    @pytest.mark.unit
    def test_sampling_config(self, sampling_config):
        """Test that sampling configuration fixture works."""
        assert 'target_len' in sampling_config
        assert 'temperature' in sampling_config
        assert 'max_new_tokens' in sampling_config
        assert sampling_config['target_len'] == 10
        assert sampling_config['temperature'] == 1.0
        assert sampling_config['max_new_tokens'] == 5
    
    @pytest.mark.integration
    def test_full_mock_pipeline(self, mock_transformers, sampling_config):
        """Test that the full mocked pipeline works."""
        tokenizer = mock_transformers['tokenizer']
        model = mock_transformers['model']
        
        # Test tokenization
        tokens = tokenizer.encode("test input")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Test model inference
        model_output = model()
        assert hasattr(model_output, 'logits')
        assert isinstance(model_output.logits, torch.Tensor)
        
        # Test decoding
        decoded = tokenizer.decode([1, 2, 3])
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestMarkers:
    """Test that pytest markers work correctly."""
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test with unit marker."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test with integration marker."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test with slow marker."""
        assert True


def test_basic_functionality():
    """Basic test to verify pytest is working."""
    assert 1 + 1 == 2


def test_torch_basic():
    """Test basic PyTorch functionality."""
    tensor = torch.tensor([1, 2, 3])
    assert tensor.shape == (3,)
    assert tensor.sum().item() == 6