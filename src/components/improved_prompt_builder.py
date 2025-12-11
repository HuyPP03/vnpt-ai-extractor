from src.utils import DynamicChoicesFormatter
from typing import List
import re


class ImprovedPromptBuilder:

    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:

        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Bạn là trợ lý AI chuyên phân tích văn bản. Trả lời câu hỏi trắc nghiệm CHỈ DỰA TRÊN đoạn văn bản.

<context>
{context}
</context>

<question>
{question}
</question>

<choices>
{choices_text}
</choices>

QUAN TRỌNG: Chỉ dựa vào văn bản. Suy luận trong đầu, KHÔNG ĐƯỢC giải thích. Chỉ trả lời DUY NHẤT MỘT chữ cái từ: {', '.join(valid_labels)}

Ví dụ đúng: "B" hoặc "A"
KHÔNG ĐƯỢC: "1. Phân tích..." hoặc "Vì..."

Đáp án:"""

    @staticmethod
    def build_math_prompt_improved(question: str, choices: List[str]) -> str:

        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Bạn là chuyên gia Toán - Lý - Hóa. Giải bài toán chính xác.

{question}

{choices_text}

QUAN TRỌNG: Tính toán trong đầu, KHÔNG ĐƯỢC viết từng bước. Chỉ trả lời DUY NHẤT MỘT chữ cái từ: {', '.join(valid_labels)}

Ví dụ đúng: "C" hoặc "A"
KHÔNG ĐƯỢC: "Bước 1:..." hoặc "Dữ kiện:..."

Đáp án:"""

    @staticmethod
    def build_math_prompt_with_verification(question: str, choices: List[str]) -> str:
        
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Giải bài toán phức tạp chính xác tuyệt đối.

{question}

{choices_text}

QUAN TRỌNG: Giải và kiểm tra trong đầu. Nếu thực sự cần, chỉ ghi 1 DÒNG NGẮN nhất về bước quan trọng. Sau đó trả lời DUY NHẤT MỘT chữ cái từ: {', '.join(valid_labels)}

Ví dụ:
- Tốt nhất: "B"
- Chấp nhận được: "Áp dụng công thức x=(-b±√Δ)/2a → B"
- KHÔNG ĐƯỢC: "1. Dữ kiện... 2. Công thức... 3. Tính..."

Đáp án:"""

    @staticmethod
    def build_knowledge_prompt_improved(question: str, choices: List[str]) -> str:

        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Dựa trên kiến thức Văn hóa, Lịch sử, Địa lý, Pháp luật Việt Nam.

{question}

{choices_text}

QUAN TRỌNG: Suy luận trong đầu, KHÔNG ĐƯỢC giải thích. Chỉ trả lời DUY NHẤT MỘT chữ cái từ: {', '.join(valid_labels)}

Ví dụ trả lời đúng: "B" hoặc "A" hoặc "C"
KHÔNG ĐƯỢC trả lời: "1. Phân tích..." hoặc "Đáp án là..."

Đáp án:"""

    @staticmethod
    def build_knowledge_prompt_with_confidence(
        question: str, choices: List[str]
    ) -> str:
        
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Dựa trên kiến thức Văn hóa, Lịch sử, Địa lý, Pháp luật Việt Nam.

{question}

{choices_text}

QUAN TRỌNG: Nếu không chắc chắn, GHI NGẮN lý do (TỐI ĐA 1 câu). Sau đó trả lời DUY NHẤT MỘT chữ cái từ: {', '.join(valid_labels)}

Ví dụ:
- Tốt nhất: "B"
- Chấp nhận: "Theo Luật Cư trú 2020 → B"
- KHÔNG ĐƯỢC: "1. Phân tích đáp án A... 2. Đáp án B..."

Đáp án:"""

    @staticmethod
    def detect_math_complexity(question: str) -> str:
        """
        Phát hiện độ phức tạp của câu toán
        Returns: "simple" hoặc "complex"
        """
        # Complex indicators
        complex_indicators = [
            "tính toán",
            "chứng minh",
            "giải hệ",
            "phương trình bậc",
            "tích phân",
            "đạo hàm",
            "giới hạn",
            "ma trận",
            "xác suất",
            "thống kê",
            "tổ hợp",
            "hoán vị",
        ]

        # Check length
        if len(question) > 500:
            return "complex"

        # Check indicators
        question_lower = question.lower()
        complex_count = sum(
            1 for indicator in complex_indicators if indicator in question_lower
        )

        if complex_count >= 2:
            return "complex"

        # Check number of numbers
        numbers = re.findall(r"\d+[.,]?\d*", question)
        if len(numbers) > 5:
            return "complex"

        return "simple"


class PromptSelector:
    """Chọn prompt phù hợp dựa trên loại và độ phức tạp câu hỏi"""

    @staticmethod
    def select_prompt(
        question_type: str,
        question: str,
        choices: List[str],
        context: str = None,
        model_type: str = "large",
    ) -> str:

        builder = ImprovedPromptBuilder()

        if question_type == "CONTEXT":
            return builder.build_context_prompt(context, question, choices)

        elif question_type == "MATH":
            # Phát hiện độ phức tạp
            complexity = builder.detect_math_complexity(question)

            if complexity == "complex" or model_type == "large":
                # Câu phức tạp hoặc dùng large model → prompt chi tiết với verification
                return builder.build_math_prompt_with_verification(question, choices)
            else:
                # Câu đơn giản với small model → prompt cải tiến
                return builder.build_math_prompt_improved(question, choices)

        elif question_type == "KNOWLEDGE":
            if model_type == "small":
                # Small model cần confidence scoring
                return builder.build_knowledge_prompt_with_confidence(question, choices)
            else:
                # Large model dùng prompt cải tiến
                return builder.build_knowledge_prompt_improved(question, choices)

        # Fallback
        return builder.build_knowledge_prompt_improved(question, choices)


# Backward compatibility
class PromptBuilder:
    """Wrapper để tương thích với code cũ"""

    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_context_prompt(context, question, choices)

    @staticmethod
    def build_math_prompt(question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_math_prompt_improved(question, choices)

    @staticmethod
    def build_knowledge_prompt(question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_knowledge_prompt_improved(question, choices)
