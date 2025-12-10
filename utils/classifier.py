from typing import Dict, Any
import re

class QuestionClassifier:
    """Phân loại câu hỏi dựa trên đặc điểm cấu trúc"""
    
    CONTEXT_INDICATORS = [
        "Đoạn thông tin:", "Tiêu đề:", "Title:", "Document 1", "Document 2",
        "-- Đoạn văn", "Văn bản:", "Bài đọc:", "Context:", "Passage:"
    ]
    
    MATH_SYMBOLS = [
        "$", "\\", "\\frac", "\\int", "\\sum", "\\sqrt", "\\pi", "^", "²", "³", '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc', '\\sinh', '\\cosh', '\\tanh', '\\coth', '\\sech', '\\csch',
        "≈", "≤", "≥", "≠", "∞", "°", "α", "β", "γ", "Δ", "∑", "∏"
    ]
    
    MATH_KEYWORDS = [
        # Toán học
        "tính toán", "tính", "giá trị kỳ vọng", "xác suất",
        "đạo hàm", "tích phân", "công thức", "phương trình",
        "bất phương trình", "hàm số",
        "ma trận", "vector", "định thức", "số phức",
        "giới hạn", "cực trị", "tiệm cận", "đồ thị", "tọa độ"
        
        # Vật lý
        "vận tốc", "gia tốc",
        "động năng", "thế năng", "nhiệt độ", "áp suất", "điện trở",
        "cường độ dòng điện", "hiệu điện thế", "từ trường", "điện trường",
        "quang học", "tần số", "bước sóng", "sóng dọc", "sóng ngang", "khối lượng riêng",
        "gia tốc trọng trường", "động lượng", "xung lượng",
        
        # Hóa học
        "mol", "nồng độ", "khối lượng mol", "nguyên tử",
        "phân tử", "ion", "hóa trị", "oxi hóa",
        "chất tan", "dung môi", "cân bằng hóa học",
        "tốc độ phản ứng", "chất xúc tác", "electron",
        "este", "axit", "bazơ", "nguyên tố",
        "bảng tuần hoàn", "đồng vị", "polyme", "hidrocacbon",
        
        # Kinh tế
        "tổng chi phí", "vốn lưu động", "lợi nhuận", "tỷ lệ", "phần trăm",
        "chi phí khấu hao", "lợi nhuận",
        # Đo lường
        "khoảng cách", "diện tích", "thể tích", "độ dài", "khối lượng",
        "thời gian", "tốc độ", "chu vi", "bán kính", "đường kính"
    ]
    
    CHEMISTRY_PATTERNS = [
        r'[A-Z][a-z]?\d+',  # Công thức hóa học có số: H2O, CO2, H2SO4
        r'\d+%',  # Phần trăm: 40%, 25%
    ]
    
    PHYSICS_UNITS = [
        "m/s", "km/h", "m/s²", "Ω", "°C", "Pa", "Hz", "eV", "kg", "mol", "mL"
    ]
    
    @classmethod
    def classify(cls, question_text: str) -> Dict[str, Any]:
        """
        Phân loại câu hỏi thành 3 loại:
        - CONTEXT: Câu hỏi có ngữ cảnh/đoạn văn
        - MATH: Câu hỏi toán/lý/hóa/STEM
        - KNOWLEDGE: Câu hỏi kiến thức thông thường
        """
        # 1. Check CONTEXT TYPE
        if any(indicator in question_text for indicator in cls.CONTEXT_INDICATORS):
            return cls._extract_context_question(question_text)
        
        # 2. Check MATH TYPE
        if cls._is_math_question(question_text):
            return {"type": "MATH", "question": question_text}
        
        # 3. Default to KNOWLEDGE TYPE
        return {"type": "KNOWLEDGE", "question": question_text}
    
    @classmethod
    def _extract_context_question(cls, text: str) -> Dict[str, Any]:
        """Tách context và câu hỏi"""
        if "Câu hỏi:" in text:
            parts = text.rsplit("Câu hỏi:", 1)
            return {
                "type": "CONTEXT",
                "context": parts[0].strip(),
                "question": parts[1].strip() if len(parts) > 1 else text
            }
        return {"type": "CONTEXT", "context": text, "question": text}
    
    @classmethod
    def _is_math_question(cls, text: str) -> bool:
        """Kiểm tra có phải câu hỏi toán/STEM không"""
        # 1. Kiểm tra ký hiệu toán học
        has_math_symbol = any(sym in text for sym in cls.MATH_SYMBOLS)
        
        # 2. Kiểm tra từ khóa STEM
        text_lower = text.lower()
        has_math_keyword = any(kw in text_lower for kw in cls.MATH_KEYWORDS)
        
        # 3. Kiểm tra đơn vị vật lý
        has_physics_unit = any(unit in text for unit in cls.PHYSICS_UNITS)
        
        # 4. Kiểm tra công thức hóa học
        has_chemistry = any(re.search(pattern, text) for pattern in cls.CHEMISTRY_PATTERNS)
        
        # 5. Kiểm tra nhiều số (đặc trưng của bài toán tính toán)
        numbers = re.findall(r'\d+[.,]?\d*%?', text)
        has_multiple_numbers = len(numbers) >= 2
        
        # 6. Phát hiện công thức hóa học (chữ in hoa + số)
        chemical_formulas = re.findall(r'\b[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*\b', text)
        has_chemical_formula = len(chemical_formulas) >= 1
        
        # Logic tổng hợp
        return (
            has_math_symbol or 
            has_physics_unit or 
            has_chemical_formula or
            (has_math_keyword and has_multiple_numbers) or
            (has_chemistry and has_math_keyword)
        )