"""AI model clients for mood analysis and ranking"""

import json
import os
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModelResponse:
    """Standardized model response"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class BaseModelClient(ABC):
    """Base class for AI model clients"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from model"""
        pass


class HuggingFaceClient(BaseModelClient):
    """Hugging Face Inference API client"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium",
                 api_token: Optional[str] = None):
        """Initialize HF client"""
        self.model_name = model_name
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"

        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

    def generate(self, prompt: str, max_length: int = 150,
                 temperature: float = 0.7, **kwargs) -> ModelResponse:
        """Generate text using HF Inference API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    return ModelResponse(success=True, data={"text": generated_text})
                else:
                    return ModelResponse(success=False, error="Invalid response format")
            else:
                return ModelResponse(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}"
                )

        except requests.RequestException as e:
            return ModelResponse(success=False, error=f"Request failed: {str(e)}")
        except Exception as e:
            return ModelResponse(success=False, error=f"Unexpected error: {str(e)}")


class OllamaClient(BaseModelClient):
    """Ollama local model client"""

    def __init__(self, model_name: str = "llama2",
                 base_url: str = "http://localhost:11434"):
        """Initialize Ollama client"""
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.7,
                 **kwargs) -> ModelResponse:
        """Generate text using Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                return ModelResponse(success=True, data={"text": generated_text})
            else:
                return ModelResponse(
                    success=False,
                    error=f"Ollama error: {response.status_code} - {response.text}"
                )

        except requests.RequestException as e:
            return ModelResponse(success=False, error=f"Request failed: {str(e)}")
        except Exception as e:
            return ModelResponse(success=False, error=f"Unexpected error: {str(e)}")


def ensure_json(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from AI model response"""
    if not response_text or not response_text.strip():
        return {}

    text = response_text.strip()

    # Try to parse directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON blocks in markdown format
    json_markers = ["```json", "```JSON", "```"]
    for marker in json_markers:
        if marker in text:
            try:
                # Extract content between markers
                parts = text.split(marker)
                if len(parts) >= 3:
                    json_content = parts[1].strip()
                    return json.loads(json_content)
            except (json.JSONDecodeError, IndexError):
                continue

    # Look for JSON-like content between braces
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            json_content = text[start_idx:end_idx + 1]
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract key-value pairs manually
    return _extract_key_value_pairs(text)


def _extract_key_value_pairs(text: str) -> Dict[str, Any]:
    """Extract key-value pairs from unstructured text"""
    result = {}

    # Common patterns for extracting data
    patterns = [
        r'score[:\s]+(\d+\.?\d*)',
        r'rating[:\s]+(\d+\.?\d*)',
        r'reason[:\s]+"([^"]+)"',
        r'explanation[:\s]+"([^"]+)"',
        r'match[:\s]+(\d+\.?\d*)'
    ]

    import re
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            key = pattern.split('[')[0]  # Extract key name
            value = matches[0]
            # Try to convert to number if possible
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value

    return result


class ModelClientFactory:
    """Factory for creating model clients"""

    @staticmethod
    def create_client(client_type: str, **kwargs) -> BaseModelClient:
        """Create model client of specified type"""
        if client_type.lower() == "huggingface":
            return HuggingFaceClient(**kwargs)
        elif client_type.lower() == "ollama":
            return OllamaClient(**kwargs)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

    @staticmethod
    def get_available_clients() -> List[str]:
        """Get list of available client types"""
        return ["huggingface", "ollama"]


class ModelManager:
    """Manage multiple model clients with fallback"""

    def __init__(self, primary_client: BaseModelClient,
                 fallback_client: Optional[BaseModelClient] = None):
        """Initialize with primary and optional fallback client"""
        self.primary_client = primary_client
        self.fallback_client = fallback_client

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate with primary client, fallback on failure"""
        response = self.primary_client.generate(prompt, **kwargs)

        if not response.success and self.fallback_client:
            return self.fallback_client.generate(prompt, **kwargs)

        return response

    def generate_json(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate and ensure JSON response"""
        response = self.generate(prompt, **kwargs)

        if response.success:
            try:
                json_data = ensure_json(response.data.get("text", ""))
                return ModelResponse(success=True, data=json_data)
            except Exception as e:
                return ModelResponse(
                    success=False,
                    error=f"Failed to parse JSON: {str(e)}"
                )

        return response