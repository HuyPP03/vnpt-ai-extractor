"""
OpenAI Client Module

A reusable module that encapsulates OpenAI client initialization
with credentials stored securely in configuration files.

Usage:
    from openai_client import OpenAIClient

    client = OpenAIClient()
    response = client.chat_completion("Hello!")
    print(response)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import openai


class OpenAIClient:
    """Reusable OpenAI client with hidden API keys"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize OpenAI client with credentials from config file.

        Args:
            config_file: Path to config JSON file. Defaults to 'openai_config.json'
                        in the same directory as this module.

        Raises:
            FileNotFoundError: If config file is not found.
            KeyError: If required config keys are missing.
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
        self.client = self._initialize_client()

    def _get_default_config_path(self) -> str:
        """Get default config file path in same directory as this module."""
        return str(Path(__file__).parent / "openai_config.json")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Returns:
            Dictionary containing OpenAI configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file is invalid JSON.
        """
        config_path = Path(self.config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("openai", {})
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {self.config_file}: {e.msg}", e.doc, e.pos
            )

    def _initialize_client(self) -> openai.OpenAI:
        """
        Initialize OpenAI client with loaded credentials.

        Returns:
            OpenAI client instance.

        Raises:
            KeyError: If required config keys are missing.
        """
        required_keys = ["base_url", "api_key"]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise KeyError(f"Missing required config keys: {missing_keys}")

        return openai.OpenAI(
            base_url=self.config["base_url"], api_key=self.config["api_key"]
        )

    def chat_completion(
        self, message: str, model: Optional[str] = None, **kwargs
    ) -> str:
        """
        Send a message and get a chat completion response.

        Args:
            message: User message to send.
            model: Model name. Defaults to config value.
            **kwargs: Additional parameters to pass to completions.create()

        Returns:
            Response content string.
        """
        model = model or self.config.get("model", "gpt-4o-mini")

        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": message}], **kwargs
        )

        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content if content is not None else ""

        return ""
    
    def get_completion(
        self, content: str, temperature: float = 0.1, 
        max_completion_tokens: int = 100, model: Optional[str] = None, **kwargs
    ) -> Optional[str]:
        """
        Get completion - compatible interface with VNPT AI client.
        
        Args:
            content: Input prompt/message
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens in response
            model: Model name (optional)
            **kwargs: Additional parameters
            
        Returns:
            Response text or None on error
        """
        try:
            model = model or self.config.get("model", "gpt-4o-mini")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_completion_tokens,
                **kwargs
            )
            
            if hasattr(response, "choices") and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content if content is not None else ""
            
            return None
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None

    def chat_completions_advanced(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Send multiple messages and get full response object.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model name. Defaults to config value.
            **kwargs: Additional parameters to pass to completions.create()

        Returns:
            Full response object from OpenAI API.
        """
        model = model or self.config.get("model", "GPT-5-mini")

        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    def get_model(self) -> str:
        """Get the configured model name."""
        return self.config.get("model", "gpt-4o-mini")

    def get_base_url(self) -> str:
        """Get the configured base URL."""
        return self.config.get("base_url", "")
    
    def set_model(self, model: str):
        """Set the model to use."""
        self.config["model"] = model


# Convenience function for simple use cases
def create_client(config_file: Optional[str] = None) -> OpenAIClient:
    """
    Factory function to create an OpenAI client.

    Args:
        config_file: Path to config JSON file.

    Returns:
        OpenAIClient instance.
    """
    return OpenAIClient(config_file)


if __name__ == "__main__":
    # Example usage

    prompt = """
Giải pháp nào sau đây được đề xuất để xử lý lỗi xác thực khi nộp hồ sơ thẩm duyệt thiết kế về phòng cháy và chữa cháy?
A. Kiểm tra lại hồ sơ để đảm bảo đầy đủ giấy tờ
B. Liên hệ với cơ quan chức năng để được hỗ trợ trực tiếp
C. Gửi hồ sơ qua dịch vụ bưu chính công ích
D. Gọi tổng đài kỹ thuật của hệ thống để khắc phục lỗi
    """

    client = OpenAIClient()
    response = client.chat_completion(prompt)
    print(response)
