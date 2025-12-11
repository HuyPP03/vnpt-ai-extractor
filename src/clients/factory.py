from src.clients.openai import OpenAIClient
from src.clients.vnpt import VNPTAIClient
from typing import Optional

class ClientFactory:
    @staticmethod
    def create(provider: str, config: Optional[dict] = None):
        if provider == "vnpt":
            return VNPTAIClient(config.get("model_type", "large"))
        elif provider == "openai":
            return OpenAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")