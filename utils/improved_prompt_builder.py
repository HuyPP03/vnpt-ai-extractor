"""
Improved Prompt Builder với prompt tối ưu hơn
Đặc biệt cải thiện cho MATH questions
"""

from utils.format_choices import DynamicChoicesFormatter
from typing import List


class ImprovedPromptBuilder:
    """Xây dựng prompts tối ưu cho các loại câu hỏi khác nhau"""
    
    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:
        """Prompt cho câu hỏi đọc hiểu - đã tối ưu tốt (100% accuracy)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        
        return f"""<instruction>
Bạn là trợ lý AI chuyên phân tích văn bản. Trả lời câu hỏi trắc nghiệm CHỈ DỰA TRÊN đoạn văn bản được cung cấp.

QUY TẮC:
- Không sử dụng kiến thức bên ngoài
- Nếu văn bản không chứa đáp án, suy luận logic từ dữ kiện có sẵn
- Chỉ chọn một đáp án từ: {', '.join(valid_labels)}
</instruction>

<context>
{context}
</context>

<question>
{question}
</question>

<choices>
{choices_text}
</choices>

Hãy chỉ trả lời bằng chữ cái ({', '.join(valid_labels)}) tương ứng với đáp án đúng nhất. Không giải thích thêm.
Đáp án:"""
    
    @staticmethod
    def build_math_prompt_improved(question: str, choices: List[str]) -> str:
        """
        Prompt cải tiến cho câu hỏi toán học - Tối ưu token
        """
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        
        return f"""Bạn là chuyên gia Toán - Lý - Hóa. Giải bài toán chính xác.

{question}

{choices_text}

Giải theo bước:
1. Dữ kiện: [liệt kê ngắn]
2. Công thức: [ghi công thức]
3. Tính: [tính từng bước]
4. Kiểm tra: [xem kết quả hợp lý không]

Đáp án: [GHI DUY NHẤT MỘT CHỮ CÁI từ {', '.join(valid_labels)}]"""
    
    @staticmethod
    def build_math_prompt_with_verification(question: str, choices: List[str]) -> str:
        """
        Prompt với cơ chế tự kiểm tra cho MATH - Tối ưu token
        """
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        
        return f"""Giải bài toán chính xác tuyệt đối.

{question}

{choices_text}

Giải theo bước:
1. Dữ kiện và yêu cầu
2. Công thức áp dụng
3. Tính toán (ghi rõ từng bước)
4. Kiểm tra kết quả

KẾT LUẬN - Đáp án: [GHI DUY NHẤT MỘT CHỮ CÁI từ {', '.join(valid_labels)}]"""
    
    @staticmethod
    def build_knowledge_prompt_improved(question: str, choices: List[str]) -> str:
        """
        Prompt cải tiến cho câu hỏi kiến thức - Tối ưu token
        """
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        
        return f"""Dựa trên kiến thức Văn hóa, Lịch sử, Địa lý, Pháp luật Việt Nam.

{question}

{choices_text}

Phân tích ngắn và chọn đáp án đúng nhất.

Đáp án: [GHI DUY NHẤT MỘT CHỮ CÁI từ {', '.join(valid_labels)}]"""
    
    @staticmethod
    def build_knowledge_prompt_with_confidence(question: str, choices: List[str]) -> str:
        """
        Prompt với confidence scoring - Tối ưu token
        """
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        
        return f"""Dựa trên kiến thức Văn hóa, Lịch sử, Địa lý, Pháp luật Việt Nam.

{question}

{choices_text}

Trả lời theo format:
Lý do: [1 câu ngắn]
Đáp án: [MỘT CHỮ CÁI từ {', '.join(valid_labels)}]
Tin cậy: [Cao/Trung bình/Thấp]"""
    
    @staticmethod
    def detect_math_complexity(question: str) -> str:
        """
        Phát hiện độ phức tạp của câu toán
        Returns: "simple" hoặc "complex"
        """
        # Indicators of complex math
        complex_indicators = [
            "tính toán", "chứng minh", "giải hệ", "phương trình bậc",
            "tích phân", "đạo hàm", "giới hạn", "ma trận",
            "xác suất", "thống kê", "tổ hợp", "hoán vị"
        ]
        
        # Check length
        if len(question) > 500:
            return "complex"
        
        # Check indicators
        question_lower = question.lower()
        complex_count = sum(1 for indicator in complex_indicators if indicator in question_lower)
        
        if complex_count >= 2:
            return "complex"
        
        # Check number of numbers (nhiều số = phức tạp)
        import re
        numbers = re.findall(r'\d+[.,]?\d*', question)
        if len(numbers) > 5:
            return "complex"
        
        return "simple"


class PromptSelector:
    """Chọn prompt phù hợp dựa trên loại và độ phức tạp câu hỏi"""
    
    @staticmethod
    def select_prompt(question_type: str, question: str, choices: List[str],
                     context: str = None, model_type: str = "large") -> str:
        """
        Chọn prompt tối ưu
        
        Args:
            question_type: MATH, CONTEXT, KNOWLEDGE
            question: Câu hỏi
            choices: Danh sách lựa chọn
            context: Context (nếu có)
            model_type: small hoặc large
            
        Returns:
            Prompt string
        """
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


if __name__ == "__main__":
    # Test prompt selection
    selector = PromptSelector()
    
    # Test MATH complex
    math_question = """Một cửa hàng tạp hóa địa phương đã tăng giá một món đồ ăn vặt phổ biến 
    từ 2,00 đô la lên 2,50 đô la, và lượng cầu giảm từ 100 đơn vị xuống 80 đơn vị. 
    Sử dụng công thức trung điểm, hãy xác định độ co giãn của cầu theo giá đối với món đồ ăn vặt này?"""
    
    choices = ["-0,5", "-1,0", "-1,5", "-2,0"]
    
    prompt = selector.select_prompt("MATH", math_question, choices, model_type="large")
    print("MATH Prompt (complex):")
    print(prompt[:500])
    print("\n" + "="*70 + "\n")

