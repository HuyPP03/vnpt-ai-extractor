from openai_client import OpenAIClient
from vnpt_ai_client import VNPTAIClient
from typing import Optional


class ModelWrapper:

    OPENAI_MODELS = [
        "openai",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-5-mini",
        "gpt-5",
    ]

    def __init__(self, model_type: str = "large"):

        self.model_type = model_type.lower()
        self.is_openai = self._is_openai_model(model_type)

        if self.is_openai:
            if OpenAIClient is None:
                raise ImportError("OpenAI client not available")
            self.client = OpenAIClient()

            if model_type != "openai":
                self.client.set_model(model_type)
        else:
            if VNPTAIClient is None:
                raise ImportError("VNPT AI client not available")
            self.client = VNPTAIClient(model_type=model_type)

    def _is_openai_model(self, model_type: str) -> bool:
        return model_type.lower() in self.OPENAI_MODELS

    def get_completion(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 100
    ) -> Optional[str]:

        try:
            if self.is_openai:
                response = self.client.get_completion(
                    content=prompt,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
            else:
                response = self.client.get_completion(
                    content=prompt,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    verbose=False,
                )

            return response

        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            return None

    def get_embedding(self, content: str, **kwargs) -> Optional[list]:

        try:
            if self.is_openai:
                # TODO: Implement OpenAI embedding
                raise NotImplementedError("OpenAI embedding not implemented yet")
            else:
                response = self.client.get_embedding(content, **kwargs)
            return response
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            return None
