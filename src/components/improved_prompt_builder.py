from src.utils import DynamicChoicesFormatter
from typing import List
import re


class ImprovedPromptBuilder:
    """
    Xây dựng prompt cho 5 loại câu hỏi:
    1. RAG - Câu hỏi có ngữ cảnh
    2. COMPULSORY - An toàn, Pháp lý
    3. STEM - Khoa học tự nhiên
    4. PRECISION_CRITICAL - Tài chính, Logic
    5. MULTI_DOMAIN - Kiến thức chung
    """

    @staticmethod
    def build_rag_prompt(context: str, question: str, choices: List[str]) -> str:
        """Prompt cho câu hỏi RAG (Retrieval-Augmented Generation)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        return f"""**Vai trò:**
Bạn là chuyên gia phân tích văn bản, chuyên đọc và hiểu ngữ cảnh để trả lời các câu hỏi trắc nghiệm một cách chính xác và khách quan.

**Mục tiêu:**
Phân tích ngữ cảnh được cung cấp, xác định lựa chọn đúng nhất cho câu hỏi trắc nghiệm dựa hoàn toàn vào thông tin trong ngữ cảnh, và trả lời chỉ bằng một chữ cái duy nhất từ các lựa chọn hợp lệ, nhằm đảm bảo câu trả lời ngắn gọn, chính xác và không có giải thích thừa.

**Ngữ cảnh:**  
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
1. Đọc kỹ toàn bộ đoạn văn trong phần <ngữ cảnh> để nắm bắt nội dung chính, các chi tiết quan trọng và ý nghĩa tổng thể.  
2. Phân tích câu hỏi trong phần <câu hỏi> một cách cẩn thận, xác định các yếu tố chính cần so sánh với ngữ cảnh.  
3. So sánh từng lựa chọn trong phần <những lựa chọn> với thông tin từ ngữ cảnh, đánh giá mức độ phù hợp dựa trên bằng chứng trực tiếp hoặc gián tiếp.  
4. Xác định lựa chọn đúng nhất hoặc gần nhất bằng cách chọn chỉ một chữ cái từ {valid_labels}, ưu tiên sự khớp hoàn hảo; nếu không có lựa chọn nào khớp hoàn hảo, chọn lựa chọn gần nhất dựa trên ngữ cảnh.  
5. Trả lời bằng cách chỉ ghi chữ cái đó, tránh thêm bất kỳ từ ngữ, giải thích, dấu chấm câu hoặc ký tự nào khác để giữ tính ngắn gọn.

**Định dạng đầu ra:**  
Chỉ một chữ cái duy nhất (ví dụ: A), không có bất kỳ nội dung bổ sung nào."""

    @staticmethod
    def build_compulsory_prompt(
        question: str, choices: List[str], context: str = None
    ) -> str:
        """Prompt cho câu hỏi COMPULSORY (An toàn, Pháp lý, Bắt buộc)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        if context and context.strip():
            return f"""**Vai trò:**
Bạn là chuyên gia về đạo đức AI và pháp luật Việt Nam. Trả lời câu hỏi về an toàn/pháp lý một cách chính xác dựa trên văn bản pháp luật được cung cấp.

**Tài liệu tham khảo:**
<tài liệu pháp luật>
{context}
</tài liệu pháp luật>

**Ngữ cảnh:**  
<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**
1. Đọc kỹ tài liệu pháp luật được cung cấp trong phần <tài liệu pháp luật>
2. Phân tích câu hỏi và xác định các điều khoản, quy định liên quan
3. Dựa vào văn bản pháp luật để chọn đáp án chính xác nhất
4. Nếu câu hỏi về hành vi bất hợp pháp → chọn phương án từ chối trả lời
5. An toàn và tuân thủ pháp luật là ưu tiên hàng đầu

**Định dạng đầu ra:**  
Chỉ trả lời một chữ cái duy nhất (ví dụ: A), không có bất kỳ nội dung bổ sung nào.
"""

        return f"""**Vai trò:**
Bạn là chuyên gia về đạo đức AI và pháp luật. Trả lời câu hỏi về an toàn/pháp lý một cách phù hợp với chuẩn mực.

**Ngữ cảnh:**  
<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Nguyên tắc:**  
- Nếu câu hỏi về hành vi bất hợp pháp → chọn phương án từ chối trả lời
- Nếu có "Tôi không thể trả lời" hoặc "Tôi từ chối chia sẻ" → xem xét kỹ
- An toàn và tuân thủ pháp luật là ưu tiên hàng đầu

**Định dạng đầu ra:**  
Chỉ trả lời một chữ cái duy nhất (ví dụ: A), không có bất kỳ nội dung bổ sung nào.
"""

    @staticmethod
    def build_stem_prompt(
        question: str, choices: List[str], subtype: str = "general"
    ) -> str:
        """Prompt cho câu hỏi STEM (Khoa học, Công nghệ, Kỹ thuật, Toán học)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        # Tùy chỉnh mô tả theo subtype
        domain_desc = {
            "vật lý": "Vật lý",
            "hóa học": "Hóa học",
            "sinh học": "Sinh học",
            "công nghệ": "Công nghệ thông tin",
            "general": "STEM",
        }
        domain = domain_desc.get(subtype, domain_desc["general"])

        return f"""**Vai trò:**  
Bạn là một chuyên gia hàng đầu trong lĩnh vực {domain}, với kiến thức sâu rộng về các nguyên tắc khoa học, toán học và công thức liên quan. Bạn giải quyết các bài toán một cách chính xác, logic và dựa trên bằng chứng, tránh suy đoán hoặc thông tin không đáng tin cậy.

**Mục tiêu:**  
Phân tích câu hỏi trắc nghiệm trong lĩnh vực {domain}, áp dụng kiến thức chuyên môn để xác định đáp án đúng duy nhất từ các lựa chọn, và trả lời bằng cách chọn một chữ cái hợp lệ (từ {', '.join(valid_labels)}), đảm bảo toàn bộ quá trình chính xác 100% dựa trên nguyên tắc chuẩn.

**Ngữ cảnh:** 
<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ nội dung trong phần <câu hỏi> để hiểu rõ vấn đề và yêu cầu.  
2. Xem xét tất cả các lựa chọn trong phần <những lựa chọn>, đánh giá từng cái dựa trên kiến thức chuyên môn trong {domain}.  
3. Phân tích vấn đề: Xác định các khái niệm chính, công thức liên quan (nếu có), và thực hiện tính toán cần thiết một cách ngắn gọn (chỉ 1–3 dòng, ví dụ: nêu công thức và bước tính đơn giản).  
4. Loại trừ các lựa chọn sai bằng lý do ngắn gọn nếu cần, nhưng giữ tổng thể ngắn gọn, tránh giải thích chi tiết hoặc liệt kê nhiều bước.  
5. Xác định đáp án đúng duy nhất từ {', '.join(valid_labels)} dựa trên phân tích khoa học/toán học chuẩn.  
6. Kết thúc bằng việc nêu rõ đáp án mà không thêm nội dung thừa.

**Định dạng đầu ra:**  
Đầu ra phải ngắn gọn, chỉ bao gồm:  
- Phân tích ngắn gọn (1–3 dòng nếu cần tính toán hoặc công thức).  
- Dòng cuối cùng: **Đáp án: [Chữ cái đúng, ví dụ: A]** (chỉ chọn một từ {', '.join(valid_labels)}, không giải thích thêm).  
Ví dụ:  
Công thức sử dụng: [công thức ngắn]. Tính toán: [bước đơn giản].  
**Đáp án: A**"""

    @staticmethod
    def build_precision_critical_prompt(
        question: str, choices: List[str], subtype: str = "general"
    ) -> str:
        """Prompt cho câu hỏi PRECISION_CRITICAL (Tài chính, Kế toán, Logic)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        # Tùy chỉnh mô tả theo subtype
        domain_desc = {
            "tài chính": "Tài chính",
            "kế toán": "Kế toán",
            "logic": "Logic/Xác suất",
            "general": "Tài chính/Kế toán/Logic",
        }
        domain = domain_desc.get(subtype, domain_desc["general"])

        return f"""**Vai trò:**  
Bạn là chuyên gia hàng đầu trong lĩnh vực {domain}, với kiến thức sâu rộng và khả năng giải quyết vấn đề một cách chính xác, logic. Bạn luôn ưu tiên độ chính xác tuyệt đối và giải thích rõ ràng, ngắn gọn.

**Mục tiêu:**  
Giải quyết bài toán được đưa ra trong <câu hỏi> một cách chính xác, chọn đáp án đúng từ các lựa chọn hợp lệ ({valid_labels}), và cung cấp đầu ra chỉ rõ đáp án dưới dạng chữ cái (A, B, C, hoặc D), đồng thời đảm bảo tính toán ngắn gọn, chú ý đến đơn vị, phần trăm và làm tròn nếu cần.

**Ngữ cảnh:**

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ <câu hỏi> và <những lựa chọn> để hiểu rõ vấn đề.  
2. Phân tích bài toán: Xác định các yếu tố chính, công thức hoặc nguyên lý liên quan trong {domain}.  
3. Thực hiện tính toán ngắn gọn nhất có thể (1-2 dòng), chú ý đơn vị, phần trăm, và làm tròn theo quy tắc chuẩn (ví dụ: làm tròn đến 2 chữ số thập phân nếu cần).  
4. Chọn đáp án đúng từ {valid_labels} dựa trên kết quả tính toán.  
5. Không thêm thông tin thừa; giữ cho giải thích ngắn gọn và tập trung vào logic dẫn đến đáp án.

**Định dạng đầu ra:**  
Đầu ra phải ngắn gọn, chỉ bao gồm:  
- Phần tính toán ngắn (1-2 dòng nếu cần).  
- Kết thúc bằng: **Đáp án: [chữ cái]** (ví dụ: Đáp án: A).  
Ví dụ:  
Tính toán: [mô tả ngắn gọn].  
**Đáp án: A**"""

    @staticmethod
    def build_multi_domain_prompt(
        question: str, choices: List[str], subtype: str = "general", context: str = None
    ) -> str:
        """Prompt cho câu hỏi MULTI_DOMAIN (Kiến thức đa lĩnh vực)"""
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)

        domain_desc = {
            "lịch sử": "Lịch sử Việt Nam",
            "địa lý": "Địa lý Việt Nam",
            "văn học": "Văn học Việt Nam",
            "triết học": "Triết học/Tư tưởng",
            "văn hóa": "Văn hóa Việt Nam",
            "general": "Kiến thức chung",
        }
        domain = domain_desc.get(subtype, domain_desc["general"])

        # Nếu có context từ RAG
        if context and context.strip():
            return f"""**Vai trò:**  
Bạn là một chuyên gia hàng đầu trong lĩnh vực {domain}, với kiến thức sâu rộng và khả năng phân tích chính xác các câu hỏi trắc nghiệm dựa trên tài liệu tham khảo được cung cấp.

**Tài liệu tham khảo:**
<tài liệu>
{context}
</tài liệu>

**Mục tiêu:**  
Dựa vào tài liệu tham khảo và kiến thức chuyên môn về {domain}, chọn đáp án chính xác nhất cho câu hỏi trắc nghiệm, đảm bảo câu trả lời ngắn gọn và đáng tin cậy.

**Ngữ cảnh:**

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ tài liệu tham khảo trong phần <tài liệu> để nắm bắt thông tin quan trọng.
2. Phân tích câu hỏi trong phần <câu hỏi> và xác định thông tin cần thiết.
3. So sánh các lựa chọn với thông tin từ tài liệu và kiến thức về {domain}.
4. Ưu tiên thông tin từ tài liệu tham khảo nếu có liên quan trực tiếp.
5. Chọn chính xác một chữ cái từ {valid_labels} làm đáp án cuối cùng.

**Định dạng đầu ra:**  
Chỉ trả lời bằng đúng một chữ cái duy nhất (ví dụ: A), không có bất kỳ văn bản giải thích, dấu chấm câu hoặc nội dung thêm nào khác."""

        # Không có context
        return f"""**Vai trò:**  
Bạn là một chuyên gia hàng đầu trong lĩnh vực {domain}, với kiến thức sâu rộng và khả năng phân tích chính xác các câu hỏi trắc nghiệm liên quan đến chủ đề này. Bạn luôn trả lời ngắn gọn, logic và dựa trên sự thật.

**Mục tiêu:**  
Chọn và trả lời đúng một lựa chọn duy nhất từ câu hỏi trắc nghiệm được cung cấp, đảm bảo đáp án chính xác dựa trên kiến thức {domain}, nhằm giúp người dùng nhận được câu trả lời nhanh chóng và đáng tin cậy mà không cần giải thích dài dòng.

**Ngữ cảnh:**

<câu hỏi>  
{question}  
</câu hỏi>  

<những lựa chọn>  
{choices_text}  
</những lựa chọn>  

Các lựa chọn hợp lệ: {valid_labels}.

**Hướng dẫn:**  
1. Đọc kỹ và hiểu rõ câu hỏi trong phần <câu hỏi> để nắm bắt yêu cầu chính.  
2. Phân tích các lựa chọn trong phần <những lựa chọn>, dựa trên kiến thức chuyên sâu về {domain} để suy luận logic và xác định đáp án đúng một cách ngắn gọn nhất.  
3. Nếu cần, ghi chú suy nghĩ nội bộ ngắn gọn (chỉ 1 dòng) để tự kiểm tra logic, nhưng không bao gồm trong đầu ra cuối cùng trừ khi câu hỏi yêu cầu giải thích.  
4. Chọn chính xác một chữ cái từ các lựa chọn hợp lệ {valid_labels} làm đáp án cuối cùng, đảm bảo không chọn sai hoặc thêm thông tin thừa.  
5. Trả lời chỉ với đáp án đã chọn, giữ cho toàn bộ quá trình tập trung vào độ chính xác và tính ngắn gọn.

**Định dạng đầu ra:**  
Chỉ trả lời bằng đúng một chữ cái duy nhất (ví dụ: A), không có bất kỳ văn bản giải thích, dấu chấm câu hoặc nội dung thêm nào khác."""

    @staticmethod
    def detect_complexity(question: str, question_type: str) -> str:
        """
        Phát hiện độ phức tạp của câu hỏi
        Returns: "simple" hoặc "complex"
        """
        # Check length
        if len(question) > 500:
            return "complex"

        # Type-specific complexity indicators
        if question_type in ["STEM", "PRECISION_CRITICAL"]:
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
                "NPV",
                "IRR",
                "chiết khấu",
            ]

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
    """
    Chọn prompt phù hợp dựa trên loại câu hỏi mới:
    RAG, COMPULSORY, STEM, PRECISION_CRITICAL, MULTI_DOMAIN
    """

    @staticmethod
    def select_prompt(
        question_type: str,
        question: str,
        choices: List[str],
        context: str = None,
        subtype: str = "general",
        model_type: str = "large",
    ) -> str:
        """
        Chọn prompt dựa trên loại câu hỏi và subtype

        Args:
            question_type: RAG, COMPULSORY, STEM, PRECISION_CRITICAL, MULTI_DOMAIN
            question: Nội dung câu hỏi
            choices: Danh sách lựa chọn
            context: Ngữ cảnh (cho RAG hoặc từ Qdrant retrieval)
            subtype: Phân loại chi tiết (vật lý, hóa học, lịch sử, ...)
            model_type: Loại model (large/small)
        """
        builder = ImprovedPromptBuilder()

        if question_type == "RAG":
            return builder.build_rag_prompt(context, question, choices)

        elif question_type == "COMPULSORY":
            return builder.build_compulsory_prompt(question, choices, context)

        elif question_type == "STEM":
            return builder.build_stem_prompt(question, choices, subtype)

        elif question_type == "PRECISION_CRITICAL":
            return builder.build_precision_critical_prompt(question, choices, subtype)

        elif question_type == "MULTI_DOMAIN":
            return builder.build_multi_domain_prompt(
                question, choices, subtype, context
            )

        # Fallback to MULTI_DOMAIN if unknown type
        return builder.build_multi_domain_prompt(question, choices, "general", context)


# Backward compatibility
class PromptBuilder:
    """Wrapper để tương thích với code cũ"""

    @staticmethod
    def build_context_prompt(context: str, question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_rag_prompt(context, question, choices)

    @staticmethod
    def build_math_prompt(question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_stem_prompt(question, choices, "general")

    @staticmethod
    def build_knowledge_prompt(question: str, choices: List[str]) -> str:
        return ImprovedPromptBuilder.build_multi_domain_prompt(
            question, choices, "general"
        )
