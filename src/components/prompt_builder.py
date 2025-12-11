from src.utils import DynamicChoicesFormatter
from typing import List


class PromptBuilder:

    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:
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
    def build_math_prompt(question: str, choices: List[str]) -> str:
        """Prompt cho câu hỏi toán học"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""<instruction>
Bạn là chuyên gia Toán học, Vật lý, Hóa học. Giải bài toán theo phương pháp từng bước với tính đúng đắn của kiến thức cao nhất.
</instruction>
<question>
{question}
</question>
<choices>
{choices_text}
</choices>
Hãy suy nghĩ từng bước, sau đó chỉ trả lời bằng chữ cái ({', '.join(valid_labels)}) tương ứng với đáp án đúng nhất.
Đáp án:"""

    @staticmethod
    def build_knowledge_prompt(question: str, choices: List[str]) -> str:
        """Prompt cho câu hỏi kiến thức"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""Dựa trên kiến thức của bạn về Văn hóa, Lịch sử, Pháp luật, Địa lý Việt Nam, hãy trả lời câu hỏi sau đây một cách chính xác nhất.
Câu hỏi:
{question}
Danh sách các đáp án:
{choices_text}
Hãy chỉ trả lời bằng chữ cái ({', '.join(valid_labels)}) tương ứng với đáp án đúng nhất. Không giải thích thêm.
Đáp án:"""

