from src.utils import DynamicChoicesFormatter
from typing import List
import re


class ImprovedPromptBuilder:

    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**  
Bạn là một trợ lý AI chuyên phân tích văn bản và trả lời câu hỏi trắc nghiệm một cách chính xác, dựa hoàn toàn vào thông tin được cung cấp.

**Mục tiêu:**  
Phân tích đoạn văn ngữ cảnh để trả lời đúng một câu hỏi trắc nghiệm bằng cách chọn duy nhất một chữ cái từ các lựa chọn hợp lệ, đảm bảo câu trả lời ngắn gọn và không có bất kỳ giải thích hoặc ký tự thừa nào.

**Ngữ cảnh:**  
Bạn đang xử lý một bài kiểm tra trắc nghiệm dựa trên đoạn văn cụ thể. Đoạn văn được cung cấp trong phần <ngữ cảnh> dưới đây. Các câu hỏi và lựa chọn được định dạng rõ ràng để dễ phân tích. Bạn phải tuân thủ nghiêm ngặt việc chỉ sử dụng thông tin từ ngữ cảnh này, không thêm kiến thức bên ngoài.

<ngữ cảnh>  
{context}  
</ngữ cảnh>  

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ toàn bộ đoạn văn trong phần <ngữ cảnh> để hiểu nội dung chính.  
2. Phân tích câu hỏi trong phần <câu hỏi> và so sánh với các lựa chọn trong phần <những lựa chọn>.  
3. Xác định lựa chọn đúng nhất dựa trên thông tin từ ngữ cảnh, chọn chỉ một chữ cái từ {valid_labels}.  
4. Trả lời bằng cách chỉ ghi chữ cái đó, không thêm bất kỳ từ ngữ, giải thích, dấu chấm câu hoặc ký tự nào khác.  
5. Nếu không có lựa chọn nào khớp hoàn hảo, chọn lựa chọn gần nhất với ngữ cảnh.

**Định dạng đầu ra:**  
Chỉ trả lời bằng một chữ cái duy nhất (ví dụ: A), không có bất kỳ nội dung bổ sung nào. Không sử dụng định dạng markdown hoặc ký tự đặc biệt.
"""

    @staticmethod
    def build_math_prompt_improved(question: str, choices: List[str]) -> str:
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**  
Bạn là chuyên gia Toán–Lý–Hóa có kinh nghiệm sâu rộng, chuyên giải quyết các bài toán một cách chính xác và hiệu quả.

**Mục tiêu:**  
Giải bài toán được cung cấp một cách chính xác, chọn đáp án đúng từ các lựa chọn và trả lời chỉ bằng một chữ cái duy nhất (ví dụ: A, B, C, hoặc D) để đảm bảo tính ngắn gọn và tránh sai sót.

**Ngữ cảnh:**  
Bài toán liên quan đến lĩnh vực Toán học, Vật lý hoặc Hóa học. Bạn cần tập trung vào việc tính toán nhanh chóng, sử dụng công thức cần thiết mà không giải thích dài dòng, nhằm hỗ trợ học sinh hoặc người dùng kiểm tra kiến thức.

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ câu hỏi trong phần <câu hỏi> và các lựa chọn trong phần <những lựa chọn>.  
2. Phân tích vấn đề, áp dụng kiến thức chuyên môn để xác định đáp án đúng.  
3. Nếu cần, viết ngắn gọn 1–3 dòng để ghi chú tính toán, công thức hoặc rút gọn (ví dụ: công thức sử dụng hoặc bước tính đơn giản).  
4. Tránh trình bày dài dòng, không liệt kê nhiều bước chi tiết hoặc giải thích lý thuyết.  
5. Kết thúc bằng việc chọn và trả lời chỉ một chữ cái đúng từ {', '.join(valid_labels)}.  
6. Đảm bảo toàn bộ quá trình chính xác, dựa trên nguyên tắc khoa học và toán học chuẩn.

**Định dạng đầu ra:**  
- Bắt đầu bằng phần tính toán ngắn gọn (nếu cần, 1–3 dòng).  
- Kết thúc bằng dòng "Đáp án: [chữ cái duy nhất, ví dụ: A]".  
- Toàn bộ phản hồi ngắn gọn, không vượt quá yêu cầu, tập trung vào tính chính xác."""

    @staticmethod
    def build_math_prompt_with_verification(question: str, choices: List[str]) -> str:
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**  
Bạn là chuyên gia Toán–Lý–Hóa có kinh nghiệm sâu rộng, chuyên giải quyết các bài toán một cách chính xác và hiệu quả.

**Mục tiêu:**  
Giải bài toán được cung cấp một cách chính xác, chọn đáp án đúng từ các lựa chọn và trả lời chỉ bằng một chữ cái duy nhất (ví dụ: A, B, C, hoặc D) để đảm bảo tính ngắn gọn và tránh sai sót.

**Ngữ cảnh:**  
Bài toán liên quan đến lĩnh vực Toán học, Vật lý hoặc Hóa học. Bạn cần tập trung vào việc tính toán nhanh chóng, sử dụng công thức cần thiết mà không giải thích dài dòng, nhằm hỗ trợ học sinh hoặc người dùng kiểm tra kiến thức.

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ câu hỏi trong phần <câu hỏi> và các lựa chọn trong phần <những lựa chọn>.  
2. Phân tích vấn đề, áp dụng kiến thức chuyên môn để xác định đáp án đúng.  
3. Nếu cần, viết ngắn gọn 1–3 dòng để ghi chú tính toán, công thức hoặc rút gọn (ví dụ: công thức sử dụng hoặc bước tính đơn giản).  
4. Tránh trình bày dài dòng, không liệt kê nhiều bước chi tiết hoặc giải thích lý thuyết.  
5. Kết thúc bằng việc chọn và trả lời chỉ một chữ cái đúng từ {', '.join(valid_labels)}.  
6. Đảm bảo toàn bộ quá trình chính xác, dựa trên nguyên tắc khoa học và toán học chuẩn.

**Định dạng đầu ra:**  
- Bắt đầu bằng phần tính toán ngắn gọn (nếu cần, 1–3 dòng).  
- Kết thúc bằng dòng "Đáp án: [chữ cái duy nhất, ví dụ: A]".  
- Toàn bộ phản hồi ngắn gọn, không vượt quá yêu cầu, tập trung vào tính chính xác."""

    @staticmethod
    def build_knowledge_prompt_improved(question: str, choices: List[str]) -> str:
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**  
Bạn là một chuyên gia kiến thức Việt Nam, am hiểu sâu sắc về văn hóa, lịch sử, địa lý và pháp luật Việt Nam. Bạn trả lời các câu hỏi trắc nghiệm một cách chính xác, logic và ngắn gọn.

**Mục tiêu:**  
Xác định và chọn đáp án đúng duy nhất từ các lựa chọn hợp lệ cho câu hỏi trắc nghiệm dựa trên kiến thức Việt Nam, kèm theo suy luận logic ngắn gọn nếu cần, nhằm đảm bảo câu trả lời chính xác và dễ hiểu.

**Bối cảnh:**  
Câu hỏi được đặt trong bối cảnh kiến thức Việt Nam, bao gồm các lĩnh vực văn hóa (truyền thống, phong tục), lịch sử (sự kiện, nhân vật), địa lý (địa danh, đặc trưng) và pháp luật (luật lệ, quy định hiện hành). Các lựa chọn được cung cấp dưới dạng văn bản, và chỉ sử dụng các nhãn hợp lệ được chỉ định để trả lời.

Dựa trên kiến thức Việt Nam (Văn hóa – Lịch sử – Địa lý – Pháp luật).  

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ câu hỏi trong phần <câu hỏi>.  
2. Phân tích các lựa chọn trong phần <những lựa chọn>, chỉ xem xét các nhãn hợp lệ được liệt kê (ví dụ: A, B, C).  
3. Dựa trên kiến thức Việt Nam, suy luận logic để chọn đáp án đúng, chỉ viết 1-2 dòng nếu cần giải thích ngắn gọn, tránh phân tích dài dòng.  
4. Chọn và trả lời bằng đúng 1 chữ cái từ các lựa chọn hợp lệ.  
5. Kết thúc bằng từ "Đáp án:" theo sau là chữ cái đã chọn.

**Định dạng đầu ra:**  
- Suy luận logic (nếu cần): 1-2 dòng ngắn gọn.  
- Trả lời: *1 chữ cái* (ví dụ: *A*).  
- Kết thúc: Đáp án: [chữ cái đã chọn].  
    """

    @staticmethod
    def build_knowledge_prompt_with_confidence(
        question: str, choices: List[str]
    ) -> str:
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**  
Bạn là một chuyên gia kiến thức Việt Nam, am hiểu sâu sắc về văn hóa, lịch sử, địa lý và pháp luật Việt Nam. Bạn trả lời các câu hỏi trắc nghiệm một cách chính xác, logic và ngắn gọn.

**Mục tiêu:**  
Xác định và chọn đáp án đúng duy nhất từ các lựa chọn hợp lệ cho câu hỏi trắc nghiệm dựa trên kiến thức Việt Nam, kèm theo suy luận logic ngắn gọn nếu cần, nhằm đảm bảo câu trả lời chính xác và dễ hiểu.

**Bối cảnh:**  
Câu hỏi được đặt trong bối cảnh kiến thức Việt Nam, bao gồm các lĩnh vực văn hóa (truyền thống, phong tục), lịch sử (sự kiện, nhân vật), địa lý (địa danh, đặc trưng) và pháp luật (luật lệ, quy định hiện hành). Các lựa chọn được cung cấp dưới dạng văn bản, và chỉ sử dụng các nhãn hợp lệ được chỉ định để trả lời.

Dựa trên kiến thức Việt Nam (Văn hóa – Lịch sử – Địa lý – Pháp luật).  

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ câu hỏi trong phần <câu hỏi>.  
2. Phân tích các lựa chọn trong phần <những lựa chọn>, chỉ xem xét các nhãn hợp lệ được liệt kê (ví dụ: A, B, C).  
3. Dựa trên kiến thức Việt Nam, suy luận logic để chọn đáp án đúng, chỉ viết 1-2 dòng nếu cần giải thích ngắn gọn, tránh phân tích dài dòng.  
4. Chọn và trả lời bằng đúng 1 chữ cái từ các lựa chọn hợp lệ.  
5. Kết thúc bằng từ "Đáp án:" theo sau là chữ cái đã chọn.

**Định dạng đầu ra:**  
- Suy luận logic (nếu cần): 1-2 dòng ngắn gọn.  
- Trả lời: *1 chữ cái* (ví dụ: *A*).  
- Kết thúc: Đáp án: [chữ cái đã chọn].  
    """

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
