from typing import Dict, Any, Optional
import requests
import json
from pathlib import Path

# API Endpoints
API_ENDPOINTS = {
    "small": "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small",
    "large": "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large",
    "embeddings": "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding",
}

# Model names
MODEL_NAMES = {
    "small": "vnptai_hackathon_small",
    "large": "vnptai_hackathon_large",
    "embeddings": "vnptai_hackathon_embedding",
}


class VNPTAIClient:
    """VNPT AI API Client for LLM and Embeddings"""

    def __init__(self, model_type: str = "large"):
        """
        Initialize VNPT AI Client

        Args:
            model_type: 'small', 'large', or 'embeddings'
        """
        self.model_type = model_type.lower()

        if self.model_type not in ["small", "large", "embeddings"]:
            raise ValueError(
                "Invalid model_type. Must be 'small', 'large', or 'embeddings'"
            )

        self._load_credentials()

    def _load_credentials(self):
        """Load API credentials from api-keys.json"""
        self._load_from_json()

    def _load_from_json(self):
        """Load credentials from api-keys.json"""
        json_path = Path(__file__).parent.parent / "configs" / "api-keys.json"

        if not json_path.exists():
            raise FileNotFoundError(f"api-keys.json not found at {json_path}")

        with open(json_path, "r") as f:
            api_keys = json.load(f)

        # Map model type to index
        model_index = {
            "large": 0,
            "small": 1,
            "embeddings": 2,
        }

        credentials = api_keys[model_index[self.model_type]]
        self.authorization = credentials["authorization"]
        self.token_id = credentials["tokenId"]
        self.token_key = credentials["tokenKey"]
        self.api_name = credentials["llmApiName"]

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": self.authorization,
            "Token-id": self.token_id,
            "Token-key": self.token_key,
            "Content-Type": "application/json",
        }

    def call(
        self,
        content: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 20,
        max_completion_tokens: int = 100,
        verbose: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the VNPT AI API

        Args:
            content: User message content
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_completion_tokens: Maximum tokens in response
            verbose: If True, print request details

        Returns:
            API response as dictionary or None on error
        """
        endpoint = API_ENDPOINTS[self.model_type]
        headers = self._get_headers()

        # Different payload structure for embeddings vs chat models
        if self.model_type == "embeddings":
            json_data = {
                "model": MODEL_NAMES[self.model_type],
                "input": content,
                "encoding_format": "float",
            }
        else:
            json_data = {
                "model": MODEL_NAMES[self.model_type],
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": 1,
                "max_completion_tokens": max_completion_tokens,
            }

        try:
            if verbose:
                print("🚀 Sending request to VNPT AI API...")
                print(
                    f"📍 Model: {self.model_type.upper()} - {self.api_name if hasattr(self, 'api_name') else 'N/A'}"
                )
                print(f"🔗 Endpoint: {endpoint}")
                print(f"💬 Content: {content}\n")

            response = requests.post(
                endpoint, headers=headers, json=json_data, timeout=30
            )

            if verbose:
                print(f"✅ Status Code: {response.status_code}")
                print(f"📦 Response: {json.dumps(response.json(), indent=2)}\n")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            raise TimeoutError("Request timeout") from e
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            raise Exception(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}") from e

    def get_completion(self, content: str, **kwargs) -> Optional[str]:

        try:
            response = self.call(content, **kwargs)

            if response and "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0].get("message", {}).get("content")
                if text:
                    try:
                        if isinstance(text, str):
                            text = text.encode().decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        pass
                return text

            raise Exception("Invalid response: No choices in API response")
        except Exception as e:
            raise Exception(f"{str(e)}") from e

    def get_embedding(self, content: str, **kwargs) -> Optional[list]:
        """
        Get embedding vector from the API (for embedding model only)

        Args:
            content: Text to embed
            **kwargs: Additional parameters for call()

        Returns:
            Embedding vector as list or None on error
        """
        response = self.call(content, **kwargs)

        if response and "data" in response and len(response["data"]) > 0:
            return response["data"][0].get("embedding", None)

        return None
