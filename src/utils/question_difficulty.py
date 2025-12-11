from dataclasses import dataclass
from typing import Any, Dict, List
import re

@dataclass
class DifficultyScore:
    """Điểm đánh giá độ khó"""
    total_score: float = 0.0
    is_hard: bool = False
    model_size: str = "small"  # "small" or "large"
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class QuestionDifficulty:
    """
    Phân loại câu hỏi: SMALL model hoặc LARGE model
    
    Scoring System:
    - Score < 50:  SMALL model (fast, cheap)
    - Score >= 50: LARGE model (better reasoning)
    """
    
    # Keywords chỉ ra câu hỏi phức tạp (cần large model)
    COMPLEX_KEYWORDS = [
        # Reasoning keywords
        'tại sao', 'vì sao', 'nguyên nhân', 'hậu quả', 
        'phân tích', 'đánh giá', 'so sánh', 'nhận xét', 
        'giải thích', 'luận giải', 'ý nghĩa', 'vai trò',
        'tác động', 'ảnh hưởng', 'liên hệ', 'mối quan hệ',
        
        # Hypothetical/complex reasoning
        'nếu', 'giả sử', 'trong trường hợp', 'khi mà',
        'sự khác biệt', 'điểm giống và khác',
        'phê phán', 'tranh luận', 'lập luận', 'chứng minh',
        'dự đoán', 'suy diễn', 'kết luận'
    ]
    
    # Keywords chỉ ra câu hỏi đơn giản (dùng small model)
    SIMPLE_KEYWORDS = [
        'là gì', 'bao nhiêu', 'khi nào', 'ở đâu', 'ai là',
        'định nghĩa', 'thế nào là', 'có phải', 'đúng hay sai',
        'kể tên', 'liệt kê', 'cho biết'
    ]
    
    # Lĩnh vực phức tạp cần large model
    COMPLEX_DOMAINS = [
        'pháp luật', 'luật', 'điều', 'khoản', 'nghị định', 'thông tư',
        'triết học', 'tư tưởng', 'lý luận', 'học thuyết', 'chủ nghĩa',
        'kinh tế', 'tài chính', 'ngân hàng', 'chứng khoán',
        'y học', 'dược', 'bệnh lý', 'chẩn đoán'
    ]
    
    @classmethod
    def classify_difficulty(cls, item: Dict[str, Any]) -> str:
        """
        Phân loại câu hỏi và trả về kích thước model cần dùng
        
        Args:
            item: Dictionary chứa 'question' và 'choices'
            
        Returns:
            str: 'small' hoặc 'large'
        """
        score = cls.calculate_difficulty_score(item)
        return score.model_size
    
    @classmethod
    def calculate_difficulty_score(cls, item: Dict[str, Any]) -> DifficultyScore:
        """
        Tính toán điểm độ khó
        
        Returns:
            DifficultyScore object
        """
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        score = DifficultyScore()
        total = 0.0
        
        # 1. Kiểm tra keywords (0-30 điểm)
        keyword_score, keyword_reasons = cls._score_keywords(question)
        total += keyword_score
        score.reasons.extend(keyword_reasons)
        
        # 2. Độ phức tạp câu hỏi (0-25 điểm)
        complexity_score, complexity_reasons = cls._score_complexity(question)
        total += complexity_score
        score.reasons.extend(complexity_reasons)
        
        # 3. Lĩnh vực chuyên sâu (0-20 điểm)
        domain_score, domain_reasons = cls._score_domain(question)
        total += domain_score
        score.reasons.extend(domain_reasons)
        
        # 4. Tính mơ hồ của đáp án (0-25 điểm)
        ambiguity_score, ambiguity_reasons = cls._score_ambiguity(choices)
        total += ambiguity_score
        score.reasons.extend(ambiguity_reasons)
        
        score.total_score = total
        score.is_hard = total >= 50
        score.model_size = "large" if score.is_hard else "small"
        
        return score
    
    @classmethod
    def _score_keywords(cls, question: str) -> tuple[float, List[str]]:
        """Đánh giá dựa trên keywords (0-30)"""
        score = 0.0
        reasons = []
        question_lower = question.lower()
        
        # Check complex keywords
        complex_found = [kw for kw in cls.COMPLEX_KEYWORDS if kw in question_lower]
        if complex_found:
            score += 20
            reasons.append(f"Từ khóa phức tạp: {', '.join(complex_found[:3])}")
        
        # Check simple keywords (negative score for simple)
        simple_found = [kw for kw in cls.SIMPLE_KEYWORDS if kw in question_lower]
        if simple_found and not complex_found:
            score -= 10  # Reduce score for simple questions
            reasons.append(f"Câu hỏi đơn giản: {simple_found[0]}")
        
        # Multiple complex keywords = very hard
        if len(complex_found) >= 2:
            score += 10
            reasons.append("Nhiều yếu tố phức tạp")
        
        return max(score, 0), reasons
    
    @classmethod
    def _score_complexity(cls, question: str) -> tuple[float, List[str]]:
        """Đánh giá độ phức tạp (0-25)"""
        score = 0.0
        reasons = []
        
        # 1. Độ dài câu hỏi
        word_count = len(question.split())
        if word_count > 40:
            score += 15
            reasons.append(f"Câu hỏi rất dài ({word_count} từ)")
        elif word_count > 25:
            score += 8
            reasons.append(f"Câu hỏi dài ({word_count} từ)")
        
        # 2. Nhiều mệnh đề
        clause_indicators = [',', ';', 'và', 'hoặc', 'nhưng', 'tuy nhiên', 'do đó', 'vì']
        clause_count = sum(question.count(ind) for ind in clause_indicators)
        if clause_count >= 3:
            score += 10
            reasons.append(f"Nhiều mệnh đề ({clause_count})")
        
        return score, reasons
    
    @classmethod
    def _score_domain(cls, question: str) -> tuple[float, List[str]]:
        """Đánh giá lĩnh vực chuyên môn (0-20)"""
        score = 0.0
        reasons = []
        question_lower = question.lower()
        
        # Check complex domains
        for domain in cls.COMPLEX_DOMAINS:
            if domain in question_lower:
                score += 15
                reasons.append(f"Lĩnh vực chuyên sâu: {domain}")
                break
        
        # Kiến thức cụ thể (số liệu, điều khoản...)
        specific_patterns = [
            r'\bđiều\s+\d+',  # Điều 12
            r'\bnghị định\s+\d+',  # Nghị định 123
            r'\bkhoản\s+\d+',  # Khoản 2
            r'\bnăm\s+\d{4}',  # Năm 2024
        ]
        
        if any(re.search(pattern, question_lower) for pattern in specific_patterns):
            score += 5
            reasons.append("Yêu cầu kiến thức cụ thể")
        
        return score, reasons
    
    @classmethod
    def _score_ambiguity(cls, choices: List[str]) -> tuple[float, List[str]]:
        """Đánh giá tính mơ hồ (0-25)"""
        score = 0.0
        reasons = []
        
        if not choices:
            return 0.0, reasons
        
        # Đáp án "không thể trả lời" - khó hơn
        uncertain_patterns = [
            'không thể trả lời', 'không thể chia sẻ', 
            'tôi không biết', 'không có đáp án',
            'tất cả đều đúng', 'tất cả đều sai',
            'không thể xác định'
        ]
        
        has_uncertain = any(
            any(pattern in choice.lower() for pattern in uncertain_patterns)
            for choice in choices
        )
        
        if has_uncertain:
            score += 15
            reasons.append("Có đáp án mơ hồ")
        
        # Đáp án dài và phức tạp
        avg_choice_length = sum(len(c.split()) for c in choices) / len(choices)
        if avg_choice_length > 10:
            score += 10
            reasons.append("Đáp án dài và phức tạp")
        
        return score, reasons
    
    @classmethod
    def is_hard(cls, item: Dict[str, Any]) -> bool:
        """
        Kiểm tra câu hỏi có khó không (cần large model)
        
        Returns:
            bool: True nếu cần large model
        """
        score = cls.calculate_difficulty_score(item)
        return score.is_hard
    
    @classmethod
    def print_analysis(cls, item: Dict[str, Any]) -> None:
        """In phân tích chi tiết"""
        score = cls.calculate_difficulty_score(item)
        
        print("="*70)
        print(f"Question: {item.get('question', '')[:80]}...")
        print("-"*70)
        print(f"Total Score:     {score.total_score:.1f}/100")
        print(f"Model Size:      {score.model_size.upper()}")
        print(f"Is Hard:         {'YES' if score.is_hard else 'NO'}")
        print("-"*70)
        print("Reasons:")
        for reason in score.reasons:
            print(f"  • {reason}")
        print("="*70)