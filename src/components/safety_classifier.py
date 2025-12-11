"""
Safety Classifier - Phát hiện câu hỏi không thể trả lời
Sử dụng small model để kiểm tra nhanh và tiết kiệm chi phí
"""

from typing import Dict, Any, List, Optional


class SafetyClassifier:
    """
    Phát hiện câu hỏi có đáp án là "không thể trả lời" hoặc tương tự
    Sử dụng small model để classification nhanh
    """

    # Các từ khóa chỉ câu trả lời "unsafe" (không thể trả lời)
    UNSAFE_KEYWORDS = [
        "không thể trả lời",
        "không thể cung cấp",
        "không thể xác định",
        "câu hỏi không hợp lệ",
    ]

    @staticmethod
    def has_unsafe_answer(choices: List[str]) -> Optional[str]:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        
        for i, choice in enumerate(choices):
            choice_lower = choice.lower()
            for keyword in SafetyClassifier.UNSAFE_KEYWORDS:
                if keyword in choice_lower:
                    return labels[i] if i < len(labels) else None
        
        return None

    @staticmethod
    def build_safety_prompt(question: str, choices: List[str]) -> str:
        from utils.format_choices import DynamicChoicesFormatter
        
        choices_text = DynamicChoicesFormatter.format_choices(choices)
        
        return f"""Bạn là một hệ thống phân loại an toàn nội dung. Nhiệm vụ của bạn là xác định câu hỏi có an toàn hay không.

CÂU HỎI UNSAFE (không an toàn) là câu hỏi có một hoặc nhiều đặc điểm sau:

1. **Tiêu cực và phỉ báng:**
   - Xúc phạm, chửi bới, làm nhục cá nhân/tổ chức
   - Kỳ thị chủng tộc, tôn giáo, giới tính, xuất xứ
   - Phát ngôn thù hận, kích động bạo lực

2. **Chống đối và chống phá:**
   - Chống phá chế độ, chính quyền
   - Kêu gọi lật đổ, gây bất ổn xã hội
   - Xuyên tạc chủ trương, chính sách của Đảng và Nhà nước
   - Phủ nhận thành tựu lịch sử, văn hóa dân tộc

3. **Vi phạm pháp luật:**
   - Hướng dẫn hoạt động bất hợp pháp
   - Khuyến khích hành vi phạm tội
   - Chia sẻ thông tin nhạy cảm, mật
   - Tuyên truyền nội dung cấm theo pháp luật Việt Nam

4. **Nguy hiểm và có hại:**
   - Hướng dẫn chế tạo vũ khí, chất nổ
   - Khuyến khích tự tử, tự gây thương tích
   - Lừa đảo, chiếm đoạt tài sản
   - Thông tin sai lệch gây hoang mang dư luận

5. **Nội dung không phù hợp:**
   - Khiêu dâm, đồi trụy
   - Bạo lực máu me
   - Ma túy và các chất gây nghiện
   - Đánh bạc bất hợp pháp

CÂU HỎI SAFE (an toàn) là câu hỏi:
- Mang tính học thuật, giáo dục chính đáng
- Tìm kiếm thông tin hợp pháp, có ích
- Thảo luận xã hội mang tính xây dựng
- Không vi phạm các tiêu chí UNSAFE ở trên

---

Câu hỏi cần phân loại:
{question}
Đáp án lựa chọn:
{choices_text}
---

Hãy phân tích kỹ câu hỏi và trả lời CHÍNH XÁC một trong hai từ: SAFE hoặc UNSAFE

Đáp án:"""

    @staticmethod
    def extract_safety_label(response: str) -> Optional[str]:
        """
        Trích xuất SAFE/UNSAFE từ response
        
        Args:
            response: Response từ model
            
        Returns:
            "SAFE" hoặc "UNSAFE" hoặc None
        """
        if not response:
            return None
        
        response_upper = response.upper().strip()
        
        # Direct match
        if "UNSAFE" in response_upper:
            return "UNSAFE"
        elif "SAFE" in response_upper:
            return "SAFE"
        
        return None

    @staticmethod
    def classify_safety(
        question: str,
        choices: List[str],
        model_wrapper=None,
        verbose: bool = False,
        use_model_verification: bool = False
    ) -> Dict[str, Any]:
        """
        Phân loại câu hỏi SAFE/UNSAFE
        
        Logic:
        - Nếu trong choices có đáp án chứa "không thể trả lời" → UNSAFE, chọn đáp án đó
        - Nếu không có → SAFE, tiếp tục pipeline bình thường
        
        Args:
            question: Câu hỏi
            choices: Danh sách lựa chọn
            model_wrapper: ModelWrapper instance (optional, chỉ dùng nếu use_model_verification=True)
            verbose: In chi tiết
            use_model_verification: Có dùng model để verify không (mặc định: False)
            
        Returns:
            Dictionary với keys:
                - is_safe: bool
                - unsafe_answer: str (label của đáp án unsafe nếu có)
                - confidence: float
                - method: str
        """
        # Kiểm tra nhanh bằng keyword matching
        unsafe_answer = SafetyClassifier.has_unsafe_answer(choices)
        
        if unsafe_answer is None:
            # Không có đáp án unsafe trong choices → SAFE
            return {
                "is_safe": True,
                "unsafe_answer": None,
                "confidence": 1.0,
                "method": "no_unsafe_choice",
                "raw_response": None
            }
        
        # Có đáp án unsafe trong choices
        if verbose:
            print(f"⚠️ Detected unsafe answer in choices: {unsafe_answer} - '{choices[ord(unsafe_answer) - ord('A')]}'")
        
        # Nếu không dùng model verification → chọn luôn đáp án unsafe
        if not use_model_verification:
            if verbose:
                print(f"✅ Auto-selecting unsafe answer: {unsafe_answer}")
            
            return {
                "is_safe": False,
                "unsafe_answer": unsafe_answer,
                "confidence": 0.95,
                "method": "keyword_direct",
                "raw_response": None
            }
        
        # Nếu dùng model verification → hỏi model xem câu hỏi có thực sự UNSAFE không
        if verbose:
            print("🔍 Using small model to verify if question is truly unsafe...")
        
        try:
            if model_wrapper is None:
                raise ValueError("model_wrapper is required when use_model_verification=True")
            
            prompt = SafetyClassifier.build_safety_prompt(question, choices)
            
            response = model_wrapper.get_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            if verbose:
                print(f"Safety classification response: {response}")
            
            safety_label = SafetyClassifier.extract_safety_label(response)
            
            if safety_label == "UNSAFE":
                return {
                    "is_safe": False,
                    "unsafe_answer": unsafe_answer,
                    "confidence": 0.9,
                    "method": "model_verified_unsafe",
                    "raw_response": response
                }
            else:
                # Model nói SAFE → câu hỏi bình thường, không chọn unsafe answer
                if verbose:
                    print("ℹ️ Model says SAFE - continuing normal pipeline")
                return {
                    "is_safe": True,
                    "unsafe_answer": None,
                    "confidence": 0.8,
                    "method": "model_verified_safe",
                    "raw_response": response
                }
        
        except Exception as e:
            if verbose:
                print(f"⚠️ Safety classification failed: {e}")
            
            # Fallback: nếu có keyword unsafe → coi như unsafe
            return {
                "is_safe": False,
                "unsafe_answer": unsafe_answer,
                "confidence": 0.7,
                "method": "keyword_fallback",
                "raw_response": None,
                "error": str(e)
            }

