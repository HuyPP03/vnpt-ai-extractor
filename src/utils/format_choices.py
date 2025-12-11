from typing import List
import string


class DynamicChoicesFormatter:

    @staticmethod
    def format_choices(choices: List[str]) -> str:
        """Format choices từ list thành chuỗi với labels A, B, C, ..."""
        if not choices:
            return ""

        formatted_lines = []
        for i, content in enumerate(choices):
            label = string.ascii_uppercase[i]
            formatted_lines.append(f"{label}. {content}")

        return "\n".join(formatted_lines)

    @staticmethod
    def get_valid_labels(choices: List[str]) -> List[str]:
        num_choices = len(choices)
        return list(string.ascii_uppercase[:num_choices])

    @staticmethod
    def validate_answer(answer: str, choices: List[str]) -> bool:
        """Kiểm tra answer có hợp lệ không"""
        if not answer:
            return False
        valid_labels = DynamicChoicesFormatter.get_valid_labels(choices)
        return answer.upper() in valid_labels
