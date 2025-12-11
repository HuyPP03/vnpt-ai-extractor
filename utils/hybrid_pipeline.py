from typing import Dict, Any, List
from utils.model import ModelWrapper
from utils.classifier import QuestionClassifier
from utils.context_filter import SemanticContextFilter
from utils.extractor import AnswerExtractor
from utils.format_choices import DynamicChoicesFormatter

try:
    from utils.improved_prompt_builder import PromptSelector

    USE_IMPROVED_PROMPTS = True
except ImportError:
    from utils.prompt_builder import PromptBuilder

    USE_IMPROVED_PROMPTS = False


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


class QuestionDifficulty:
    """Phân loại độ khó của câu hỏi"""

    PRECISION_CRITICAL = "precision-critical"  # Không được trả lời
    COMPULSORY = "compulsory"  # Bắt buộc đúng
    NORMAL = "normal"  # Câu thường

    @classmethod
    def classify_difficulty(cls, item: Dict[str, Any]) -> str:

        # Kiểm tra metadata
        if "category" in item:
            category = item["category"].lower()
            if "không được trả lời" in category:
                return cls.PRECISION_CRITICAL
            elif "bắt buộc" in category or "compulsory" in category:
                return cls.COMPULSORY

        # Kiểm tra qid pattern (nếu có quy ước)
        qid = item.get("qid", "")
        if "critical" in qid.lower():
            return cls.PRECISION_CRITICAL
        elif "comp" in qid.lower():
            return cls.COMPULSORY

        return cls.NORMAL


class HybridModelSelector:

    @staticmethod
    def select_model(
        question_type: str,
        difficulty: str,
        context_length: int = 0,
        strategy: str = "hybrid",
    ) -> str:

        if strategy == "cost-optimized":
            return HybridModelSelector._cost_optimized(
                question_type, difficulty, context_length
            )
        elif strategy == "quality-optimized":
            return HybridModelSelector._quality_optimized(
                question_type, difficulty, context_length
            )
        else:  # hybrid (default)
            return HybridModelSelector._hybrid_strategy(
                question_type, difficulty, context_length
            )

    @staticmethod
    def _cost_optimized(
        question_type: str, difficulty: str, context_length: int
    ) -> str:
        """Chiến lược tối ưu chi phí"""
        if question_type == "MATH":
            return "large"  # MATH luôn cần large
        elif question_type == "CONTEXT":
            return "small"  # Context đã filter, small đủ
        elif question_type == "KNOWLEDGE":
            if difficulty == "compulsory":
                return "large"
            else:
                return "small"
        return "small"

    @staticmethod
    def _quality_optimized(
        question_type: str, difficulty: str, context_length: int
    ) -> str:
        return "large"  # Tất cả dùng large

    @staticmethod
    def _hybrid_strategy(
        question_type: str, difficulty: str, context_length: int
    ) -> str:
        # MATH: Luôn large
        if question_type == "MATH":
            return "large"

        # CONTEXT: Dựa vào độ dài
        elif question_type == "CONTEXT":
            if context_length < 1000:
                return "small"
            else:
                return "large"

        # KNOWLEDGE: Dựa vào độ khó
        elif question_type == "KNOWLEDGE":
            if difficulty in ["compulsory", "precision-critical"]:
                return "large"
            else:
                return "small_with_fallback"

        return "large"  # Default


class OptimizedHybridPipeline:
    """
    Pipeline tối ưu với hybrid model selection
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        use_improved_prompts: bool = True,
        large_model_name: str = "large",
        small_model_name: str = "small",
    ):
        """
        Args:
            strategy: Chiến lược lựa chọn model
                     - "cost-optimized": Tối ưu chi phí
                     - "quality-optimized": Tối ưu chất lượng
                     - "hybrid": Cân bằng (mặc định)
            use_improved_prompts: Sử dụng improved prompts (mặc định: True)
            large_model_name: Tên large model (default: "large" cho VNPT, "gpt-4o-mini" cho OpenAI)
            small_model_name: Tên small model (default: "small" cho VNPT, "gpt-3.5-turbo" cho OpenAI)
        """
        self.strategy = strategy
        self.use_improved_prompts = use_improved_prompts and USE_IMPROVED_PROMPTS
        self.large_model_name = large_model_name
        self.small_model_name = small_model_name
        self.small_model = None  # Lazy loading
        self.large_model = None  # Lazy loading

        self.classifier = QuestionClassifier()
        self.context_filter = SemanticContextFilter()

        # Use improved prompts if available
        if self.use_improved_prompts:
            self.prompt_selector = PromptSelector()
        else:
            self.prompt_builder = PromptBuilder()

        self.answer_extractor = AnswerExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.formatter = DynamicChoicesFormatter()

        # Statistics
        self.stats = {
            "small_used": 0,
            "large_used": 0,
            "fallback_triggered": 0,
            "rate_limit_fallback": 0,
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

        # 1. Phân loại câu hỏi
        classification = self.classifier.classify(question)
        question_type = classification["type"]

        # 2. Phân loại độ khó
        difficulty = QuestionDifficulty.classify_difficulty(item)

        # 3. Xử lý context (nếu có)
        context_length = 0
        if question_type == "CONTEXT":
            context = classification["context"]
            context_length = len(context)

            # Apply semantic filtering cho context dài
            if context_length > 1000:
                filtered_context, metadata = self.context_filter.filter_context(
                    context=context,
                    question=classification["question"],
                    max_chunks=4,
                    max_chars=1000,
                )
                classification["context"] = filtered_context
                context_length = len(filtered_context)

                if verbose:
                    print(
                        f"Context filtered: {metadata['original_length']} → {metadata['filtered_length']} chars"
                    )

        # 4. Chọn model
        selected_model = HybridModelSelector.select_model(
            question_type=question_type,
            difficulty=difficulty,
            context_length=context_length,
            strategy=self.strategy,
        )

        if verbose:
            print(f"Type: {question_type}, Difficulty: {difficulty}")
            print(f"Selected model: {selected_model}")

        # 5. Build prompt
        if self.use_improved_prompts:
            # Use improved prompt selector
            prompt = self.prompt_selector.select_prompt(
                question_type=question_type,
                question=classification.get("question", question),
                choices=choices,
                context=classification.get("context"),
                model_type=(
                    selected_model
                    if selected_model != "small_with_fallback"
                    else "small"
                ),
            )
        else:
            # Use original prompt builder
            if question_type == "CONTEXT":
                prompt = self.prompt_builder.build_context_prompt(
                    context=classification["context"],
                    question=classification["question"],
                    choices=choices,
                )
            elif question_type == "MATH":
                prompt = self.prompt_builder.build_math_prompt(
                    question=classification["question"], choices=choices
                )
            else:  # KNOWLEDGE
                prompt = self.prompt_builder.build_knowledge_prompt(
                    question=classification["question"], choices=choices
                )

        # 6. Get response
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

        # 7. Add metadata
        result["qid"] = qid
        result["ground_truth"] = ground_truth
        result["type"] = question_type
        result["difficulty"] = difficulty

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
        if question_type == "MATH":
            max_tokens = 1024
            temperature = 0.05
        else:
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
                predicted = None

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

            # Kiểm tra nếu là lỗi rate limit và đang dùng large model
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
        }


# Convenience function
def create_pipeline(strategy: str = "hybrid") -> OptimizedHybridPipeline:

    return OptimizedHybridPipeline(strategy=strategy)
