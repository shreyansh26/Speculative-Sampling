"""
Shared pytest fixtures for the speculative sampling test suite.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_device():
    """Provides a mock device for testing without requiring GPU/CPU specifics."""
    return "cpu"


@pytest.fixture
def mock_torch_tensor():
    """Provides a mock torch tensor for testing."""
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


@pytest.fixture
def sample_prompt():
    """Provides a sample text prompt for testing."""
    return "The quick brown fox"


@pytest.fixture
def sample_tokens():
    """Provides sample tokenized input for testing."""
    return torch.tensor([[101, 2003, 4248, 2829, 4419]], dtype=torch.long)


@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2003, 4248, 2829, 4419]
    tokenizer.decode.return_value = "The quick brown fox jumps over"
    tokenizer.return_tensors = "pt"
    return tokenizer


@pytest.fixture
def mock_model():
    """Provides a mock language model for testing."""
    model = MagicMock()
    
    # Mock the logits output
    mock_logits = torch.randn(1, 5, 50257)  # batch_size=1, seq_len=5, vocab_size=50257
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    model.return_value = mock_output
    
    return model


@pytest.fixture
def mock_transformers():
    """Provides mocked transformers components to avoid loading real models."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class:
        
        # Configure the mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 2003, 4248, 2829, 4419]
        mock_tokenizer.decode.return_value = "Mocked decoded text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Configure the mock model
        mock_model = MagicMock()
        mock_logits = torch.randn(1, 5, 50257)
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        yield {
            'tokenizer': mock_tokenizer,
            'model': mock_model,
            'tokenizer_class': mock_tokenizer_class,
            'model_class': mock_model_class
        }


@pytest.fixture
def sampling_config():
    """Provides common sampling configuration for tests."""
    return {
        'target_len': 10,
        'temperature': 1.0,
        'max_new_tokens': 5
    }


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability for testing."""
    with patch('torch.cuda.is_available') as mock_cuda_available:
        mock_cuda_available.return_value = False  # Default to CPU for tests
        yield mock_cuda_available


@pytest.fixture
def suppress_output(capsys):
    """Fixture to capture and suppress stdout/stderr during tests."""
    return capsys