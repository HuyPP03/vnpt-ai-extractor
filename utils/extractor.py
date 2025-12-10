from typing import List, Optional
import re


class AnswerExtractor:
    """Trích xuất đáp án từ response của model"""
    
    @staticmethod
    def extract(model_response: str, valid_labels: List[str]) -> Optional[str]:
        """Trích xuất đáp án từ response"""
        if not model_response or not valid_labels:
            return None
        
        # Tạo pattern cho valid labels
        labels_pattern = '|'.join(re.escape(label) for label in valid_labels)
        
        # Pattern 1: "Đáp án: A" hoặc "Đáp án là A"
        pattern1 = rf'[Đđ]áp\s*án\s*(?:là|:)?\s*({labels_pattern})\b'
        match = re.search(pattern1, model_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: "Kết luận: A" hoặc "Answer: A"
        pattern2 = rf'(?:[Kk]ết\s*luận|[Aa]nswer)\s*(?:là|:)?\s*({labels_pattern})\b'
        match = re.search(pattern2, model_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: "[A]" hoặc "(A)"
        pattern3 = rf'[\[\(]({labels_pattern})[\]\)]'
        matches = re.findall(pattern3, model_response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # Pattern 4: Standalone letter at start
        pattern4 = rf'^({labels_pattern})[.\s\)]'
        match = re.search(pattern4, model_response.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 5: First occurrence in last 100 chars
        tail = model_response[-100:]
        pattern5 = rf'\b({labels_pattern})\b'
        matches = re.findall(pattern5, tail, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # Pattern 6: Most frequent label
        matches = re.findall(rf'\b({labels_pattern})\b', model_response, re.IGNORECASE)
        if matches:
            from collections import Counter
            label_counts = Counter([m.upper() for m in matches])
            return label_counts.most_common(1)[0][0]
        
        return None