from typing import List, Optional
import re


class AnswerExtractor:
    """Trích xuất đáp án từ response của model"""
    
    @staticmethod
    def extract(model_response: str, valid_labels: List[str]) -> Optional[str]:
        """
        Trích xuất đáp án từ response
        
        Xử lý các trường hợp:
        - "Đáp án: A"
        - "Đáp án: A. Text thêm"
        - "KẾT LUẬN: ... đáp án chính xác là: A. Text"
        """
        if not model_response or not valid_labels:
            return None
        
        # Tạo pattern cho valid labels (case-insensitive)
        labels_pattern = '|'.join(re.escape(label) for label in valid_labels)
        
        # Pattern 1: "Đáp án: A" hoặc "Đáp án là A" (có thể có text sau)
        # Ưu tiên cao nhất vì rõ ràng nhất
        pattern1 = rf'[Đđ]áp\s*án\s*(?:là|:|chính\s*xác\s*là)?[\s:]*({labels_pattern})(?:\.|,|\s|$)'
        match = re.search(pattern1, model_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: "KẾT LUẬN" section - tìm chữ cái sau "là:" hoặc ":"
        pattern2 = rf'(?:[Kk]ết\s*luận|KẾT\s*LUẬN).*?(?:là|:)\s*({labels_pattern})(?:\.|,|\s|$)'
        match = re.search(pattern2, model_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: "Answer: A" hoặc "Answer is A"
        pattern3 = rf'[Aa]nswer\s*(?:is|:)?\s*({labels_pattern})(?:\.|,|\s|$)'
        match = re.search(pattern3, model_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 4: "[A]" hoặc "(A)" - trong ngoặc
        pattern4 = rf'[\[\(]({labels_pattern})[\]\)]'
        matches = re.findall(pattern4, model_response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # Pattern 5: Standalone letter at start of line
        pattern5 = rf'^({labels_pattern})(?:\.|,|\s|$)'
        match = re.search(pattern5, model_response.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
        
        # Pattern 6: Tìm trong 200 ký tự cuối (thường là kết luận)
        tail = model_response[-200:]
        # Tìm pattern "là X" hoặc ": X" trong tail
        pattern6 = rf'(?:là|:)\s*({labels_pattern})(?:\.|,|\s|$)'
        matches = re.findall(pattern6, tail, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # Pattern 7: Tìm chữ cái đứng độc lập trong tail
        pattern7 = rf'\b({labels_pattern})(?:\.|,|\s|$)'
        matches = re.findall(pattern7, tail, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # Pattern 8: Most frequent label trong toàn bộ response
        matches = re.findall(rf'\b({labels_pattern})\b', model_response, re.IGNORECASE)
        if matches:
            from collections import Counter
            label_counts = Counter([m.upper() for m in matches])
            # Chỉ return nếu có ít nhất 2 lần xuất hiện
            most_common = label_counts.most_common(1)[0]
            if most_common[1] >= 2:
                return most_common[0]
        
        return None