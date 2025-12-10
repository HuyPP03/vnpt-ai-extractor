from openai_client import OpenAIClient
from vnpt_ai_client import VNPTAIClient
from typing import Optional


class ModelWrapper:
    """Wrapper để gọi các model khác nhau với interface thống nhất"""
    
    def __init__(self, model_type: str = "large"):
        """
        Args:
            model_type: 'small', 'large', or 'openai'
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "openai":
            if OpenAIClient is None:
                raise ImportError("OpenAI client not available")
            self.client = OpenAIClient()
        else:
            if VNPTAIClient is None:
                raise ImportError("VNPT AI client not available")
            self.client = VNPTAIClient(model_type=model_type)
    
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
            if self.model_type == "openai":
                response = self.client.chat_completion(
                    message=prompt,
                    max_completion_tokens=max_tokens
                )
            else:
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
            if self.model_type == "openai":
                response = self.client.get_embedding(content, **kwargs)
            else:
                response = self.client.get_embedding(content, **kwargs)
            return response
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            return None