import pytest
import torch
from src.models.llm import LLM

@pytest.fixture
def llm():
    """Fixture to create a basic LLM instance"""
    model = LLM(model_name="Qwen/Qwen2.5-1.5B")
    return model

@pytest.fixture
def loaded_llm(llm):
    """Fixture to create and load an LLM instance"""
    llm.load()
    return llm

def test_llm_initialization(llm):
    """Test if LLM is initialized correctly"""
    assert llm.model_name == "Qwen/Qwen2.5-1.5B"
    assert llm.model is None
    assert llm.tokenizer is None

def test_llm_loading(loaded_llm):
    """Test if model and tokenizer are loaded correctly"""
    assert loaded_llm.model is not None
    assert loaded_llm.tokenizer is not None

def test_model_device(loaded_llm):
    """Test if model is on the correct device"""
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_device = next(loaded_llm.model.parameters()).device.type
    assert actual_device == expected_device

def test_generate_without_loading():
    """Test generate method fails properly when model isn't loaded"""
    llm = LLM()
    with pytest.raises(ValueError, match="Model and tokenizer must be loaded first"):
        llm.generate("test prompt")

def test_generate_with_loading(loaded_llm):
    """Test generate method produces output"""
    prompt = "Once upon a time"
    generated_text = loaded_llm.generate(prompt)

    assert isinstance(generated_text, str)
    assert len(generated_text) > 0
    assert prompt in generated_text
