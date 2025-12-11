from typing import List, Dict
from functools import lru_cache


class VectorDBInterface:

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.cache = {}

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text_hash: str, text: str):
        """Cache embeddings để tránh tính toán lại"""
        if self.embedding_model:
            return self.embedding_model.encode(text)
        return None

    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Tìm kiếm documents liên quan

        Args:
            query: Câu hỏi cần tìm
            top_k: Số lượng documents trả về
            threshold: Ngưỡng similarity tối thiểu

        Returns:
            List of {text: str, score: float, metadata: dict}
        """
        # TODO: Implement actual vector search
        # Đây là placeholder
        return []

    def add_documents(self, documents: List[Dict], category: str):
        """Thêm documents vào database"""
        # TODO: Implement
        pass
