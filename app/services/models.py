"""Model client management for AI playlist ranking"""

from typing import Any, Dict, Optional


class ModelResponse:
    """Standardized response from AI models"""

    def __init__(self, success: bool, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.success = success
        self.data = data or {}
        self.error = error


class BaseModelClient:
    """Base class for model clients"""

    def generate_json(self, prompt: str, temperature: float = 0.7, max_length: int = 256) -> ModelResponse:
        raise NotImplementedError


class HuggingFaceClient(BaseModelClient):
    """Client for Hugging Face models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # In production, you’d configure API access here.

    def generate_json(self, prompt: str, temperature: float = 0.7, max_length: int = 256) -> ModelResponse:
        # TODO: Replace with actual Hugging Face API call
        # Mock implementation for demo
        return ModelResponse(
            success=True,
            data={
                "score": 7.5,
                "reason": "Good match for your mood",
                "factors": ["tempo", "energy"]
            }
        )


class OllamaClient(BaseModelClient):
    """Client for Ollama (local models)"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        # In production, you’d connect to Ollama here.

    def generate_json(self, prompt: str, temperature: float = 0.7, max_length: int = 256) -> ModelResponse:
        # TODO: Replace with actual Ollama API call
        # Mock implementation for demo
        return ModelResponse(
            success=True,
            data={
                "score": 8.0,
                "reason": "Matches well with your described vibe",
                "factors": ["genre", "mood"]
            }
        )


class ModelManager:
    """Manages multiple model clients with fallback"""

    def __init__(self, primary: BaseModelClient, fallback: Optional[BaseModelClient] = None):
        self.primary = primary
        self.fallback = fallback

    def generate_json(self, prompt: str, temperature: float = 0.7, max_length: int = 256) -> ModelResponse:
        try:
            return self.primary.generate_json(prompt, temperature, max_length)
        except Exception as e:
            if self.fallback:
                return self.fallback.generate_json(prompt, temperature, max_length)
            return ModelResponse(success=False, error=str(e))


class ModelClientFactory:
    """Factory to create model clients by type"""

    @staticmethod
    def create_client(client_type: str, model_name: str) -> BaseModelClient:
        if client_type == "huggingface":
            return HuggingFaceClient(model_name)
        elif client_type == "ollama":
            return OllamaClient(model_name)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")