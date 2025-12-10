from openai_client import OpenAIClient
from vnpt_ai_client import VNPTAIClient
from typing import Optional


class ModelWrapper:
    """Wrapper để gọi các model khác nhau với interface thống nhất"""
    
    # OpenAI model names
    OPENAI_MODELS = [
        "openai", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]
    
    def __init__(self, model_type: str = "large"):
        """
        Args:
            model_type: Model name
                - VNPT: 'small', 'large', 'embeddings'
                - OpenAI: 'openai', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', etc.
        """
        self.model_type = model_type.lower()
        self.is_openai = self._is_openai_model(model_type)
        
        if self.is_openai:
            if OpenAIClient is None:
                raise ImportError("OpenAI client not available")
            self.client = OpenAIClient()
            
            # Set specific OpenAI model if provided
            if model_type != "openai":
                self.client.set_model(model_type)
        else:
            if VNPTAIClient is None:
                raise ImportError("VNPT AI client not available")
            self.client = VNPTAIClient(model_type=model_type)
    
    def _is_openai_model(self, model_type: str) -> bool:
        """Check if model is OpenAI"""
        return model_type.lower() in self.OPENAI_MODELS
    
    def get_completion(self, prompt: str, temperature: float = 0.1, 
                      max_tokens: int = 100) -> Optional[str]:
        """
        Gọi model để lấy completion
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max completion tokens
            
        Returns:
            Response text or None on error
        """
        try:
            if self.is_openai:
                # OpenAI client
                response = self.client.get_completion(
                    content=prompt,
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
            else:
                # VNPT AI client
                response = self.client.get_completion(
                    content=prompt,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    verbose=False
                )
            
            return response
        
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            return None
    
    def get_embedding(self, content: str, **kwargs) -> Optional[list]:
        """
        Gọi model để lấy embedding
        
        Args:
            content: Input content
        """
        try:
            if self.is_openai:
                # OpenAI doesn't have embedding in this client yet
                # TODO: Implement OpenAI embedding
                raise NotImplementedError("OpenAI embedding not implemented yet")
            else:
                # VNPT AI client
                response = self.client.get_embedding(content, **kwargs)
            return response
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            return None