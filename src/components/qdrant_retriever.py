import time
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantRetriever:
    """
    Component RAG retrieval từ Qdrant cho các loại câu hỏi:
    - COMPULSORY: An toàn, Pháp lý, Bắt buộc
    - MULTI_DOMAIN: Đa lĩnh vực (Lịch sử, Địa lý, Văn học, Văn hóa, v.v.)
    """

    # Cấu hình collection
    COLLECTION_NAME = "VN_Dataset"

    # Cấu hình các Qdrant servers
    QDRANT_SERVERS = [
        {
            "name": "VN_Dataset_1",
            "url": "https://73b43e51-6868-47f9-af08-fc527f33dcc5.europe-west3-0.gcp.cloud.qdrant.io",
            "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xcfHMGiy-aSEMFmBHLWzzBqoMXGDhG0SUW9uCIFSKnM",
            "domains": ["People_and_Society", "Law_and_Government"],
        },
        {
            "name": "VN_Dataset_2",
            "url": "https://b312715d-6775-45f4-823a-72b4d7a3c0cc.europe-west3-0.gcp.cloud.qdrant.io",
            "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.VDn0KFK9RCGY8g2tHweRX374QKTTbrwCCirVj1A3OYQ",
            "domains": ["Health", "Finance"],
        },
        {
            "name": "VN_Dataset_3",
            "url": "https://e0c71761-f2a4-4c19-8b47-3b6ff506c0ea.us-east4-0.gcp.cloud.qdrant.io",
            "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zIHdjfLkO-GhaDXFHFkoTljY1fmBJiKOJPNqaMVOzgs",
            "domains": ["News", "Books_and_Literature"],
        },
        {
            "name": "VN_Dataset_4",
            "url": "https://44b253c8-1c32-4eff-a2f8-7a62aff7baec.us-east4-0.gcp.cloud.qdrant.io",
            "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.D6zQLSUN9sq4P5RPOJ7SdRv4RmYzVLewklKq84kKjNU",
            "domains": ["Arts_and_Entertainment", "Science"],
        },
        {
            "name": "VN_Dataset_5",
            "url": "https://0a88d3e4-ed37-40c0-8aa5-da7b67933012.us-east4-0.gcp.cloud.qdrant.io",
            "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.svPpqJt2mboIK1bZMxIxA7IkvO7lXtt0EIj6E6gRPVg",
            "domains": ["Sensitive_Subjects", "Jobs_and_Education"],
        },
    ]

    # Mapping từ question subtype sang domain trong Qdrant
    SUBTYPE_TO_DOMAIN_MAP = {
        # COMPULSORY subtypes -> Law_and_Government hoặc Sensitive_Subjects
        "pháp luật": "Law_and_Government",
        "safety": "Sensitive_Subjects",
        "legal": "Law_and_Government",
        # STEM subtypes -> Science
        "vật lý": "Science",
        "hóa học": "Science",
        "sinh học": "Science",
        "toán học": "Science",
        "công nghệ": "Science",
        # PRECISION_CRITICAL subtypes -> Finance hoặc Health
        "tài chính": "Finance",
        "kế toán": "Finance",
        "y tế": "Health",
        "logic": "Science",
        # MULTI_DOMAIN subtypes -> các domain tương ứng
        "lịch sử": "People_and_Society",
        "địa lý": "People_and_Society",
        "chính trị": "People_and_Society",
        "triết học": "People_and_Society",
        "văn hóa": "People_and_Society",
        "văn học": "Books_and_Literature",
        "nghệ thuật": "Arts_and_Entertainment",
        "giáo dục": "Jobs_and_Education",
        "nghề nghiệp": "Jobs_and_Education",
        "tin tức": "News",
        # Fallback
        "general": "People_and_Society",
    }

    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Load model only when needed (lazy loading)"""
        if self.model is not None:
            return
        
        try:
            print(f"📦 Loading embedding model '{self.model_name}' (transformers)...")
            print(f"🔧 Device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling - lấy trung bình của token embeddings
        (giống cách sentence-transformers làm)
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode(self, text: str) -> List[float]:
        """
        Encode text thành embedding vector
        """
        self._load_model()
        
        # Tokenize
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # Mean pooling
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings (giống sentence-transformers)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Convert to list
        return sentence_embeddings[0].cpu().numpy().tolist()

    def _get_server_config(self, domain: str) -> Optional[Dict]:
        for server in self.QDRANT_SERVERS:
            if domain in server["domains"]:
                return server
        return None

    def _map_subtype_to_domain(self, question_type: str, subtype: str) -> str:
        if subtype in self.SUBTYPE_TO_DOMAIN_MAP:
            return self.SUBTYPE_TO_DOMAIN_MAP[subtype]

        # Fallback theo question_type
        if question_type == "COMPULSORY":
            return "Law_and_Government"
        elif question_type == "MULTI_DOMAIN":
            return "People_and_Society"

        return "People_and_Society"

    def retrieve(
        self,
        question: str,
        question_type: str,
        subtype: str = "general",
        top_k: int = 10,
        timeout: int = 60,
    ) -> List[Dict[str, Any]]:
        domain = self._map_subtype_to_domain(question_type, subtype)

        server_config = self._get_server_config(domain)
        if not server_config:
            print(
                f"❌ Error: Unknown domain '{domain}' for question type '{question_type}' and subtype '{subtype}'"
            )
            return []

        print(f"\n{'='*80}")
        print(f"RAG RETRIEVAL INFO:")
        print(f"Question Type: {question_type}")
        print(f"Subtype: {subtype}")
        print(f"Mapped Domain: {domain}")
        print(f"Server: {server_config['name']}")
        print(
            f"Question: {question[:100]}..."
            if len(question) > 100
            else f"Question: {question}"
        )
        print(f"{'='*80}\n")

        # Encode query thành vector
        try:
            print("🔄 Encoding query...")
            vector = self._encode(question)
            print(f"✅ Query encoded successfully (dimension: {len(vector)})")
        except Exception as e:
            print(f"❌ Error encoding query: {e}")
            return []

        results = self._search_qdrant(
            vector=vector,
            server_config=server_config,
            top_k=top_k,
            timeout=timeout,
        )

        return results

    def _search_qdrant(
        self,
        vector: List[float],
        server_config: Dict,
        top_k: int,
        timeout: int,
    ) -> List[Dict[str, Any]]:
        try:
            client = QdrantClient(
                url=server_config["url"],
                api_key=server_config["api_key"],
                timeout=timeout,
            )
            search_results = client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points

            results = []
            print(f"\n✅ FOUND {len(search_results)} RESULTS:")
            print("=" * 80)

            for i, hit in enumerate(search_results, 1):
                payload = hit.payload or {}
                full_text = payload.get("text", "N/A")

                if len(full_text) > 500:
                    preview_text = full_text[:500].replace("\n", " ") + "..."
                else:
                    preview_text = full_text.replace("\n", " ")

                print(f"\nResult #{i}")
                print(f"Score: {hit.score:.4f}")
                print(f"ID: {hit.id}")
                print(f"Domain: {payload.get('domain', 'N/A')}")
                print(f"Text Preview: {preview_text}")
                print("-" * 80)

                results.append(
                    {
                        "text": full_text,
                        "score": hit.score,
                        "id": hit.id,
                        "domain": payload.get("domain", "N/A"),
                        "metadata": payload,
                    }
                )

            return results

        except Exception as e:
            print(f"❌ Error during search: {e}")

            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                print(f"🔄 Retrying with longer timeout ({timeout * 2}s)...")
                return self._search_qdrant_retry(
                    vector=vector,
                    server_config=server_config,
                    top_k=top_k,
                    timeout=timeout * 2,
                )

            return []

    def _search_qdrant_retry(
        self,
        vector: List[float],
        server_config: Dict,
        top_k: int,
        timeout: int,
    ) -> List[Dict[str, Any]]:
        try:
            client = QdrantClient(
                url=server_config["url"],
                api_key=server_config["api_key"],
                timeout=timeout,
            )

            search_results = client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points

            results = []
            print(f"\n✅ RETRY SUCCESSFUL - FOUND {len(search_results)} RESULTS:")
            print("=" * 80)

            for i, hit in enumerate(search_results, 1):
                payload = hit.payload or {}
                full_text = payload.get("text", "N/A")

                if len(full_text) > 500:
                    preview_text = full_text[:500].replace("\n", " ") + "..."
                else:
                    preview_text = full_text.replace("\n", " ")

                print(f"\nResult #{i}")
                print(f"Score: {hit.score:.4f}")
                print(f"ID: {hit.id}")
                print(f"Domain: {payload.get('domain', 'N/A')}")
                print(f"Text Preview: {preview_text}")
                print("-" * 80)

                results.append(
                    {
                        "text": full_text,
                        "score": hit.score,
                        "id": hit.id,
                        "domain": payload.get("domain", "N/A"),
                        "metadata": payload,
                    }
                )

            return results

        except Exception as retry_error:
            print(f"❌ Retry failed: {retry_error}")
            return []

    def format_context_from_results(
        self,
        results: List[Dict[str, Any]],
        max_chars: int = 2000,
        include_scores: bool = False,
    ) -> str:
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            text = result["text"]
            score = result["score"]
            domain = result.get("domain", "N/A")
            if include_scores:
                header = f"[Document {i} - Domain: {domain} - Score: {score:.3f}]\n"
            else:
                header = f"[Document {i} - Domain: {domain}]\n"

            part = header + text + "\n\n"
            if total_chars + len(part) > max_chars:
                remaining = max_chars - total_chars - len(header) - 20
                if remaining > 100:
                    truncated_text = text[:remaining] + "..."
                    part = header + truncated_text + "\n\n"
                    context_parts.append(part)
                break

            context_parts.append(part)
            total_chars += len(part)

        return "".join(context_parts).strip()

    def retrieve_and_format(
        self,
        question: str,
        question_type: str,
        subtype: str = "general",
        top_k: int = 5,
        max_chars: int = 2000,
        include_scores: bool = False,
    ) -> Dict[str, Any]:

        results = self.retrieve(
            question=question,
            question_type=question_type,
            subtype=subtype,
            top_k=top_k,
        )

        context = self.format_context_from_results(
            results=results,
            max_chars=max_chars,
            include_scores=include_scores,
        )

        return {
            "context": context,
            "num_documents": len(results),
            "avg_score": (
                sum(r["score"] for r in results) / len(results) if results else 0.0
            ),
            "domains": list(set(r.get("domain", "N/A") for r in results)),
            "raw_results": results,
        }