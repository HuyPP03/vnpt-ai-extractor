import numpy as np
import re
from typing import List, Dict, Any, Tuple
from utils.model import ModelWrapper


class SemanticContextFilter:

    def __init__(self):
        self.embedding_model = ModelWrapper(model_type="embeddings")
        self._embedding_cache = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Generate embedding
        if self.embedding_model is None:
            # Fallback: Simple TF-IDF style embedding
            embedding = self._simple_embedding(text)
        else:
            # Use actual embedding model
            try:
                if hasattr(self.embedding_model, "encode"):
                    # Sentence-Transformers style
                    embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                elif hasattr(self.embedding_model, "get_embedding"):
                    # Custom embedding model
                    embedding = self.embedding_model.get_embedding(text)
                else:
                    embedding = self._simple_embedding(text)
            except:
                embedding = self._simple_embedding(text)

        # Cache
        self._embedding_cache[text] = embedding
        return embedding

    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:

        # Normalize text
        text = text.lower()
        words = re.findall(r"\b\w+\b", text)

        # Create simple hash-based embedding
        embedding = np.zeros(dim)
        for i, word in enumerate(words):
            # Simple hash function
            hash_val = hash(word) % dim
            embedding[hash_val] += 1.0 / (i + 1)  # Position weight

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Tính cosine similarity giữa 2 vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def split_into_chunks(
        self, text: str, chunk_size: int = 200, overlap: int = 50
    ) -> List[Dict[str, Any]]:

        chunks = []

        sections = self._split_by_sections(text)

        if sections:
            # If we have natural sections, use them
            for section in sections:
                section_text = section["title"] + "\n" + section["content"]

                # If section is too long, split it
                if len(section_text) > chunk_size:
                    sub_chunks = self._split_long_text(
                        section_text, chunk_size, overlap
                    )
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append(
                            {
                                "text": sub_chunk,
                                "section_title": section["title"],
                                "is_partial": True,
                                "part": i + 1,
                            }
                        )
                else:
                    chunks.append(
                        {
                            "text": section_text,
                            "section_title": section["title"],
                            "is_partial": False,
                        }
                    )
        else:
            # Fallback: simple sliding window
            chunks = self._split_long_text(text, chunk_size, overlap)
            chunks = [
                {"text": c, "section_title": "", "is_partial": True} for c in chunks
            ]

        return chunks

    def _split_by_sections(self, text: str) -> List[Dict[str, str]]:
        """Chia text theo sections/paragraphs"""
        sections = []
        lines = text.split("\n")

        current_section = {"title": "", "content": ""}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section header (short line, often capitalized)
            is_header = (
                len(line) < 100
                and (line[0].isupper() if line else False)
                and not line.endswith((".", ",", ";", "!", "?", ":"))
            )

            if is_header and current_section["content"]:
                sections.append(current_section.copy())
                current_section = {"title": line, "content": ""}
            elif is_header:
                current_section["title"] = line
            else:
                current_section["content"] += line + " "

        if current_section["content"]:
            sections.append(current_section)

        return sections

    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chia text dài thành chunks với overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.7:
                    end = start + break_point + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def retrieve_relevant_chunks(
        self, context: str, question: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[Tuple[str, float]]:

        # Get question embedding
        question_embedding = self._get_embedding(question)

        # Split context into chunks
        chunks = self.split_into_chunks(context, chunk_size=300, overlap=50)

        # Calculate similarities
        chunk_scores = []
        for chunk in chunks:
            chunk_embedding = self._get_embedding(chunk["text"])
            similarity = self.cosine_similarity(question_embedding, chunk_embedding)

            # Boost score if section title is relevant
            if chunk["section_title"]:
                title_embedding = self._get_embedding(chunk["section_title"])
                title_similarity = self.cosine_similarity(
                    question_embedding, title_embedding
                )
                similarity = similarity * 0.7 + title_similarity * 0.3

            chunk_scores.append((chunk["text"], similarity))

        # Sort by similarity
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum similarity and take top-k
        relevant_chunks = [
            (text, score)
            for text, score in chunk_scores[:top_k]
            if score >= min_similarity
        ]

        return relevant_chunks

    def filter_context(
        self, context: str, question: str, max_chunks: int = 3, max_chars: int = 1000
    ) -> Tuple[str, Dict[str, Any]]:

        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(
            context=context, question=question, top_k=max_chunks, min_similarity=0.3
        )

        if not relevant_chunks:
            # Fallback: truncate original
            return context[:max_chars], {"method": "truncation", "chunks_found": 0}

        # Combine chunks
        combined_text = []
        total_chars = 0
        chunks_used = 0

        for chunk_text, score in relevant_chunks:
            if total_chars + len(chunk_text) > max_chars:
                # Add partial if space remains
                remaining = max_chars - total_chars
                if remaining > 200:
                    combined_text.append(chunk_text[:remaining] + "...")
                    chunks_used += 1
                break

            combined_text.append(chunk_text)
            total_chars += len(chunk_text)
            chunks_used += 1

        filtered = "\n\n".join(combined_text)

        metadata = {
            "method": "semantic_retrieval",
            "original_length": len(context),
            "filtered_length": len(filtered),
            "chunks_found": len(relevant_chunks),
            "chunks_used": chunks_used,
            "avg_similarity": (
                sum(s for _, s in relevant_chunks[:chunks_used]) / chunks_used
                if chunks_used > 0
                else 0
            ),
            "compression_ratio": f"{(1 - len(filtered)/len(context))*100:.1f}%",
        }

        return filtered, metadata
