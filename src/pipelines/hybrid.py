from typing import Dict, Any, List, Optional

from src.components import (
    AnswerExtractor,
    ModelWrapper,
    QuestionClassifier,
    SafetyClassifier,
    SemanticContextFilter,
    PromptSelector,
    QdrantRetriever,
)
from src.utils import DynamicChoicesFormatter, QuestionDifficulty


class ConfidenceScorer:

    @staticmethod
    def calculate_confidence(
        model_response: str, extracted_answer: str, valid_labels: List[str]
    ) -> float:

        if not model_response or not extracted_answer:
            return 0.0

        confidence = 0.5  # Base confidence

        # 1. Đáp án xuất hiện nhiều lần (max +0.3)
        answer_count = model_response.upper().count(extracted_answer.upper())
        confidence += min(answer_count * 0.1, 0.3)

        # 2. Có giải thích rõ ràng (length > 50 chars)
        if len(model_response) > 50:
            confidence += 0.1

        # 3. Có từ khóa xác định
        positive_keywords = ["Đáp án là", "Chắc chắn", "Rõ ràng", "Kết luận"]
        if any(kw in model_response for kw in positive_keywords):
            confidence += 0.1

        # 4. Không có từ khóa không chắc chắn
        negative_keywords = ["có thể", "không chắc", "khó nói", "không rõ"]
        if any(kw in model_response.lower() for kw in negative_keywords):
            confidence -= 0.2

        # 5. Đáp án ở cuối response (thường là kết luận)
        if model_response.strip().endswith(extracted_answer):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))


class HybridModelSelector:
    """
    Lựa chọn model phù hợp cho 5 loại câu hỏi mới:
    RAG, COMPULSORY, STEM, PRECISION_CRITICAL, MULTI_DOMAIN
    """

    @staticmethod
    def select_model(
        question_type: str,
        difficulty: str,
        context_length: int = 0,
        strategy: str = "hybrid",
        subtype: str = "general",
    ) -> str:

        if strategy == "cost-optimized":
            return HybridModelSelector._cost_optimized(
                question_type, difficulty, context_length, subtype
            )
        elif strategy == "quality-optimized":
            return HybridModelSelector._quality_optimized(
                question_type, difficulty, context_length, subtype
            )
        else:  # hybrid (default)
            return HybridModelSelector._hybrid_strategy(
                question_type, difficulty, context_length, subtype
            )

    @staticmethod
    def _cost_optimized(
        question_type: str, difficulty: str, context_length: int, subtype: str
    ) -> str:
        """Chiến lược tối ưu chi phí"""
        if question_type in ["STEM", "PRECISION_CRITICAL"]:
            return "large"  # Cần độ chính xác cao
        elif question_type == "COMPULSORY":
            return "large"  # An toàn quan trọng
        elif question_type == "RAG":
            return "small"  # Context đã filter, small đủ
        elif question_type == "MULTI_DOMAIN":
            return difficulty  # Dựa vào độ khó
        return "small"

    @staticmethod
    def _quality_optimized(
        question_type: str, difficulty: str, context_length: int, subtype: str
    ) -> str:
        """Chiến lược tối ưu chất lượng"""
        return "large"  # Tất cả dùng large

    @staticmethod
    def _hybrid_strategy(
        question_type: str, difficulty: str, context_length: int, subtype: str
    ) -> str:
        """Chiến lược cân bằng (mặc định)"""

        # STEM: Luôn large (độ chính xác quan trọng)
        if question_type == "STEM":
            return "large"

        # PRECISION_CRITICAL: Luôn large (độ chính xác tuyệt đối)
        elif question_type == "PRECISION_CRITICAL":
            return "large"

        # COMPULSORY: Luôn large (an toàn quan trọng)
        elif question_type == "COMPULSORY":
            return "large"

        # RAG: Dựa vào độ dài context
        elif question_type == "RAG":
            if context_length < 1000:
                return "small"
            else:
                return "large"

        # MULTI_DOMAIN: Dựa vào độ khó và subtype
        elif question_type == "MULTI_DOMAIN":
            # Triết học, lịch sử phức tạp → large
            if subtype in ["triết học", "lịch sử"] or difficulty == "large":
                return "large"
            else:
                return "small_with_fallback"

        return "large"  # Default


class HybridPipeline:
    """
    Pipeline tối ưu với hybrid model selection
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        large_model_name: str = "large",
        small_model_name: str = "small",
        compulsory_safety_mode: str = "keyword",
        use_qdrant_rag: bool = True,
        qdrant_top_k: int = 5,
        qdrant_max_chars: int = 2000,
    ):
        """
        Args:
            strategy: Chiến lược lựa chọn model
                     - "cost-optimized": Tối ưu chi phí
                     - "quality-optimized": Tối ưu chất lượng
                     - "hybrid": Cân bằng (mặc định)
            large_model_name: Tên large model (default: "large" cho VNPT, "gpt-4o-mini" cho OpenAI)
            small_model_name: Tên small model (default: "small" cho VNPT, "gpt-3.5-turbo" cho OpenAI)
            compulsory_safety_mode: Chế độ safety check cho câu hỏi COMPULSORY
                        - "keyword": Dùng keyword matching (nhanh, mặc định)
                        - "model": Dùng model verification (chính xác hơn)
            use_qdrant_rag: Có sử dụng Qdrant RAG cho COMPULSORY và MULTI_DOMAIN không
            qdrant_top_k: Số documents lấy từ Qdrant
            qdrant_max_chars: Độ dài tối đa của context từ Qdrant
        """
        self.strategy = strategy
        self.large_model_name = large_model_name
        self.small_model_name = small_model_name
        self.small_model = None  # Lazy loading
        self.large_model = None  # Lazy loading
        self.compulsory_safety_mode = compulsory_safety_mode
        self.use_qdrant_rag = use_qdrant_rag
        self.qdrant_top_k = qdrant_top_k
        self.qdrant_max_chars = qdrant_max_chars

        self.classifier = QuestionClassifier()
        self.context_filter = SemanticContextFilter()
        self.safety_classifier = SafetyClassifier()
        self.prompt_selector = PromptSelector()

        # Khởi tạo QdrantRetriever nếu cần
        self.qdrant_retriever = None
        if self.use_qdrant_rag:
            try:
                print("Initializing QdrantRetriever...")
                self.qdrant_retriever = QdrantRetriever()
                print("✓ QdrantRetriever initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize QdrantRetriever: {e}")
                print("Continuing without Qdrant RAG support...")
                self.use_qdrant_rag = False

        self.answer_extractor = AnswerExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.formatter = DynamicChoicesFormatter()

        # Statistics
        self.stats = {
            "small_used": 0,
            "large_used": 0,
            "fallback_triggered": 0,
            "rate_limit_fallback": 0,
            "compulsory_detected": 0,
            "qdrant_rag_used": 0,
            "total_processed": 0,
        }

    def _get_model(self, model_type: str) -> ModelWrapper:
        """Lazy loading models"""
        if model_type == "small":
            if self.small_model is None:
                self.small_model = ModelWrapper(model_type=self.small_model_name)
            return self.small_model
        else:
            if self.large_model is None:
                self.large_model = ModelWrapper(model_type=self.large_model_name)
            return self.large_model

    def _process_compulsory_question(
        self,
        qid: str,
        question: str,
        choices: List[str],
        ground_truth: str = None,
        subtype: str = "safety",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Xử lý câu hỏi COMPULSORY (Safety/Refusal/Law)
        Sử dụng safety_classifier và Qdrant RAG
        """
        if verbose:
            print(f"🔒 Processing COMPULSORY question (subtype={subtype})")

        # Xác định có dùng model verification không
        use_model_verification = self.compulsory_safety_mode == "model"

        # Nếu dùng model verification, cần model_wrapper
        model_wrapper = None
        if use_model_verification:
            model_wrapper = self._get_model("small")

        # Gọi safety classifier
        safety_result = self.safety_classifier.classify_safety(
            question=question,
            choices=choices,
            model_wrapper=model_wrapper,
            verbose=verbose,
            use_model_verification=use_model_verification,
        )

        if not safety_result["is_safe"]:
            # Có đáp án unsafe/refusal trong choices → chọn luôn đáp án đó
            if verbose:
                print("⚠️ Safety/Refusal answer detected in choices!")
                print(f"Auto-selecting answer: {safety_result['unsafe_answer']}")

            self.stats["compulsory_detected"] += 1
            self.stats["total_processed"] += 1

            result = {
                "qid": qid,
                "predicted": safety_result["unsafe_answer"],
                "raw_response": f"COMPULSORY: {safety_result.get('raw_response', 'keyword_detected')}",
                "model_used": "safety_classifier",
                "confidence": safety_result["confidence"],
                "ground_truth": ground_truth,
                "type": "COMPULSORY",
                "subtype": subtype,
                "difficulty": "compulsory",
                "extraction_failed": False,
                "safety_method": safety_result["method"],
                "qdrant_used": False,
            }

            if ground_truth:
                result["correct"] = result["predicted"] == ground_truth
            else:
                result["correct"] = None

            return result

        # Nếu không phát hiện được đáp án refusal rõ ràng, dùng model với RAG
        if verbose:
            print("No clear refusal answer detected, using model with RAG...")

        # Retrieve context từ Qdrant
        qdrant_context = self._retrieve_qdrant_context(
            question=question,
            question_type="COMPULSORY",
            subtype=subtype,
            verbose=verbose,
        )

        # Build prompt cho COMPULSORY với context (nếu có)
        prompt = self.prompt_selector.select_prompt(
            question_type="COMPULSORY",
            question=question,
            choices=choices,
            context=qdrant_context,
            subtype=subtype,
            model_type="large",
        )

        # Gọi large model (COMPULSORY cần độ chính xác cao)
        result = self._get_model_response(
            model_type="large",
            prompt=prompt,
            question_type="COMPULSORY",
            choices=choices,
            verbose=verbose,
        )

        # Add metadata
        result["qid"] = qid
        result["ground_truth"] = ground_truth
        result["type"] = "COMPULSORY"
        result["subtype"] = subtype
        result["difficulty"] = "compulsory"
        result["qdrant_used"] = qdrant_context is not None

        if ground_truth and result["predicted"]:
            result["correct"] = result["predicted"] == ground_truth
        else:
            result["correct"] = None

        self.stats["compulsory_detected"] += 1
        self.stats["total_processed"] += 1

        return result

    def _retrieve_qdrant_context(
        self,
        question: str,
        question_type: str,
        subtype: str,
        verbose: bool = False,
    ) -> Optional[str]:
        """
        Retrieve context từ Qdrant cho các câu hỏi COMPULSORY và MULTI_DOMAIN

        Returns:
            Context string hoặc None nếu không retrieve được
        """
        if not self.use_qdrant_rag or self.qdrant_retriever is None:
            return None

        # Chỉ retrieve cho COMPULSORY và MULTI_DOMAIN
        if question_type not in ["COMPULSORY", "MULTI_DOMAIN"]:
            return None

        try:
            if verbose:
                print(
                    f"📚 Retrieving context from Qdrant (type={question_type}, subtype={subtype})..."
                )

            rag_result = self.qdrant_retriever.retrieve_and_format(
                question=question,
                question_type=question_type,
                subtype=subtype,
                top_k=self.qdrant_top_k,
                max_chars=self.qdrant_max_chars,
                include_scores=False,
            )

            context = rag_result.get("context", "")

            if context and context.strip():
                self.stats["qdrant_rag_used"] += 1

                if verbose:
                    print(f"✓ Retrieved {rag_result['num_documents']} documents")
                    print(f"  Avg score: {rag_result['avg_score']:.4f}")
                    print(f"  Context length: {len(context)} chars")

                return context
            else:
                if verbose:
                    print("⚠️ No relevant context found in Qdrant")
                return None

        except Exception as e:
            if verbose:
                print(f"⚠️ Error retrieving from Qdrant: {e}")
            return None

    def process_single(
        self, item: Dict[str, Any], verbose: bool = False
    ) -> Dict[str, Any]:

        qid = item.get("qid", "unknown")
        question = item.get("question", "").strip()
        choices = item.get("choices", [])
        ground_truth = (
            item.get("answer", "").strip().upper() if "answer" in item else None
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"QID: {qid}")
            print(f"Question: {question[:100]}...")

        # 1. Phân loại câu hỏi (với choices để detect COMPULSORY)
        classification = self.classifier.classify(question, choices)
        question_type = classification["type"]
        subtype = classification.get("subtype", "general")

        # 2. Xử lý đặc biệt cho COMPULSORY (Safety/Refusal)
        if question_type == "COMPULSORY":
            return self._process_compulsory_question(
                qid=qid,
                question=question,
                choices=choices,
                ground_truth=ground_truth,
                subtype=subtype,
                verbose=verbose,
            )

        # 3. Phân loại độ khó
        difficulty = QuestionDifficulty.classify_difficulty(item)

        # 4. Xử lý context
        context_length = 0
        qdrant_context = None

        # 4a. Nếu là RAG (có context sẵn)
        if question_type == "RAG":
            context = classification.get("context", "")
            context_length = len(context)

            # Apply semantic filtering cho context dài
            if context_length > 10000:
                filtered_context, metadata = self.context_filter.filter_context(
                    context=context,
                    question=classification.get("question", question),
                    max_chunks=4,
                    max_chars=1000,
                )
                classification["context"] = filtered_context
                context_length = len(filtered_context)

                if verbose:
                    print(
                        f"Context filtered: {metadata['original_length']} → {metadata['filtered_length']} chars"
                    )

        # 4b. Nếu là MULTI_DOMAIN, retrieve context từ Qdrant
        elif question_type == "MULTI_DOMAIN":
            qdrant_context = self._retrieve_qdrant_context(
                question=question,
                question_type=question_type,
                subtype=subtype,
                verbose=verbose,
            )
            if qdrant_context:
                context_length = len(qdrant_context)

        # 5. Chọn model
        selected_model = HybridModelSelector.select_model(
            question_type=question_type,
            difficulty=difficulty,
            context_length=context_length,
            strategy=self.strategy,
            subtype=subtype,
        )

        if verbose:
            print(
                f"Type: {question_type}, Subtype: {subtype}, Difficulty: {difficulty}"
            )
            print(f"Selected model: {selected_model}")

        # 6. Build prompt
        # Chọn context phù hợp
        if question_type == "RAG":
            context = classification.get("context")
        elif question_type == "MULTI_DOMAIN":
            context = qdrant_context
        else:
            context = None

        prompt = self.prompt_selector.select_prompt(
            question_type=question_type,
            question=classification.get("question", question),
            choices=choices,
            context=context,
            subtype=subtype,
            model_type=(
                selected_model if selected_model != "small_with_fallback" else "small"
            ),
        )

        # 7. Get response
        if selected_model == "small_with_fallback":
            result = self._process_with_fallback(
                prompt=prompt,
                question_type=question_type,
                choices=choices,
                verbose=verbose,
            )
        else:
            result = self._get_model_response(
                model_type=selected_model,
                prompt=prompt,
                question_type=question_type,
                choices=choices,
                verbose=verbose,
            )

        # 8. Add metadata
        result["qid"] = qid
        result["ground_truth"] = ground_truth
        result["type"] = question_type
        result["subtype"] = subtype
        result["difficulty"] = difficulty
        result["qdrant_used"] = qdrant_context is not None

        # Check correctness
        if ground_truth and result["predicted"]:
            result["correct"] = result["predicted"] == ground_truth
        else:
            result["correct"] = None

        self.stats["total_processed"] += 1

        if verbose:
            print(f"Predicted: {result['predicted']}")
            if ground_truth:
                print(f"Ground truth: {ground_truth}")
                print(f"Correct: {result['correct']}")

        return result

    def _get_model_response(
        self,
        model_type: str,
        prompt: str,
        question_type: str,
        choices: List[str],
        verbose: bool = False,
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:

        model = self._get_model(model_type)

        # Adjust parameters based on question type
        if question_type in ["STEM", "PRECISION_CRITICAL"]:
            # Cần nhiều token hơn cho tính toán và giải thích
            max_tokens = 1024
            temperature = 0.05  # Nhiệt độ thấp cho độ chính xác cao
        elif question_type == "COMPULSORY":
            # Cần ổn định và an toàn
            max_tokens = 512
            temperature = 0.0  # Không có ngẫu nhiên
        else:
            # RAG, MULTI_DOMAIN
            max_tokens = 256
            temperature = 0.1

        # Call model
        try:
            response = model.get_completion(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens
            )

            if verbose:
                print(f"Model response ({model_type}): {response}")

            # Extract answer
            valid_labels = self.formatter.get_valid_labels(choices)
            predicted = self.answer_extractor.extract(response, valid_labels)

            # Validate
            is_valid = (
                self.formatter.validate_answer(predicted, choices)
                if predicted
                else False
            )

            if not is_valid:
                model = self._get_model("small")
                response = model.get_completion(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
                if verbose:
                    print(f"Model response (small): {response}")
                predicted = self.answer_extractor.extract(response, valid_labels)

            # Calculate confidence
            confidence = self.confidence_scorer.calculate_confidence(
                model_response=response,
                extracted_answer=predicted or "",
                valid_labels=valid_labels,
            )

            # Update stats
            if model_type == "small":
                self.stats["small_used"] += 1
            else:
                self.stats["large_used"] += 1

            return {
                "predicted": predicted,
                "raw_response": response,
                "model_used": model_type,
                "confidence": confidence,
                "extraction_failed": not is_valid,
            }

        except Exception as e:
            error_str = str(e)
            print(f"Error calling {model_type} model: {error_str}")
            is_rate_limit = any(
                keyword in error_str.lower()
                for keyword in ["rate limit", "quota", "401", "429", "unauthorized"]
            )

            if is_rate_limit and model_type == "large" and allow_fallback:
                print("⚠️ Large model hết quota! Tự động chuyển sang small model...")
                if verbose:
                    print("Fallback reason: Rate limit exceeded on large model")

                # Retry với small model
                small_result = self._get_model_response(
                    model_type="small",
                    prompt=prompt,
                    question_type=question_type,
                    choices=choices,
                    verbose=verbose,
                    allow_fallback=False,  # Không fallback nữa
                )

                # Thêm metadata về việc fallback
                small_result["rate_limit_fallback"] = True
                small_result["original_model"] = "large"
                self.stats["fallback_triggered"] += 1
                self.stats["rate_limit_fallback"] += 1

                return small_result

            # Nếu không thể fallback hoặc không phải rate limit
            return {
                "predicted": None,
                "raw_response": None,
                "model_used": model_type,
                "confidence": 0.0,
                "extraction_failed": True,
                "error": error_str,
                "rate_limit_fallback": False,
            }

    def _process_with_fallback(
        self, prompt: str, question_type: str, choices: List[str], verbose: bool = False
    ) -> Dict[str, Any]:

        if verbose:
            print("Trying small model first...")

        # Try small first
        small_result = self._get_model_response(
            model_type="small",
            prompt=prompt,
            question_type=question_type,
            choices=choices,
            verbose=verbose,
        )

        # Check if fallback needed
        need_fallback = False
        fallback_reason = None

        if small_result["predicted"] is None:
            need_fallback = True
            fallback_reason = "extraction_failed"
        elif small_result["confidence"] < 0.6:
            need_fallback = True
            fallback_reason = "low_confidence"

        if need_fallback:
            if verbose:
                print(f"Fallback to large model (reason: {fallback_reason})")

            # Retry with large
            large_result = self._get_model_response(
                model_type="large",
                prompt=prompt,
                question_type=question_type,
                choices=choices,
                verbose=verbose,
            )

            large_result["fallback_used"] = True
            large_result["fallback_reason"] = fallback_reason
            large_result["small_confidence"] = small_result["confidence"]

            self.stats["fallback_triggered"] += 1

            return large_result

        small_result["fallback_used"] = False
        return small_result

    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê sử dụng model"""
        total = self.stats["total_processed"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "small_percentage": f"{self.stats['small_used']/total*100:.1f}%",
            "large_percentage": f"{self.stats['large_used']/total*100:.1f}%",
            "fallback_rate": f"{self.stats['fallback_triggered']/total*100:.1f}%",
            "rate_limit_fallback_rate": f"{self.stats['rate_limit_fallback']/total*100:.1f}%",
            "compulsory_detection_rate": f"{self.stats['compulsory_detected']/total*100:.1f}%",
        }
