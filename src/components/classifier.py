from typing import Dict, Any, List
import re


class QuestionClassifier:
    """
    Phân loại câu hỏi thành 5 loại chính:
    1. RAG (Retrieval-Augmented Generation) - Câu hỏi có ngữ cảnh sẵn
    2. COMPULSORY (An toàn, Pháp lý, Bắt buộc) - Cần RAG từ Law_and_Government
    3. STEM (Khoa học, Công nghệ, Kỹ thuật, Toán học) - Cần RAG từ Science
    4. PRECISION_CRITICAL (Tài chính, Y tế, Kế toán) - Cần RAG từ Finance/Health
    5. MULTI_DOMAIN (Đa lĩnh vực) - Cần RAG từ People_and_Society, Books_and_Literature, etc.

    Domains trong Qdrant:
    - People_and_Society: Lịch sử, Địa lý, Xã hội, Chính trị
    - Law_and_Government: Pháp luật, Chính phủ
    - Health: Y tế, Sức khỏe
    - Finance: Tài chính, Kinh tế
    - News: Tin tức
    - Books_and_Literature: Văn học, Sách
    - Arts_and_Entertainment: Nghệ thuật, Giải trí
    - Science: Khoa học (Vật lý, Hóa học, Sinh học)
    - Sensitive_Subjects: Chủ đề nhạy cảm
    - Jobs_and_Education: Nghề nghiệp, Giáo dục
    """

    RAG_INDICATORS = [
        "Đoạn thông tin:",
        "Tiêu đề:",
        "Title:",
        "Content:",
        "Document 1",
        "Document 2",
        "Document [",
        "-- Đoạn văn",
        "Văn bản:",
        "Bài đọc:",
        "Context:",
        "Passage:",
        "Theo đoạn văn",
        "Dựa vào đoạn văn",
        "Đoạn văn sau",
    ]
    COMPULSORY_KEYWORDS = {
        # Pháp luật - Law_and_Government
        "pháp luật": [
            "nghị định",
            "luật",
            "bộ luật",
            "thông tư",
            "quyết định",
            "vi phạm",
            "phạt",
            "mức phạt",
            "xử phạt",
            "hành vi vi phạm",
            "quy định",
            "điều khoản",
            "pháp lý",
            "hình sự",
            "dân sự",
            "hành chính",
            "hiến pháp",
            "án",
            "tòa án",
            "tội danh",
            "truy tố",
            "kháng cáo",
        ],
        # An toàn - Sensitive_Subjects
        "safety": [
            "tôi không thể chia sẻ",
            "tôi không thể trả lời",
            "tôi từ chối trả lời",
            "không thể cung cấp",
            "không thể hỗ trợ",
            "vi phạm đạo đức",
            "không thể chia sẻ",
            "trốn thuế",
            "làm giả",
            "giả mạo",
            "gian lận",
            "phá hoại",
            "chống phá",
            "lật đổ",
            "bạo loạn",
            "khủng bố",
            "ma túy",
            "tội phạm",
        ],
    }
    STEM_MATH_SYMBOLS = [
        "$",
        "\\frac",
        "\\int",
        "\\sum",
        "\\sqrt",
        "\\pi",
        "\\theta",
        "\\alpha",
        "\\beta",
        "\\gamma",
        "\\Delta",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\log",
        "\\ln",
        "^{",
        "_{",
        "²",
        "³",
        "≈",
        "≤",
        "≥",
        "≠",
        "∞",
        "∑",
        "∏",
        "∫",
    ]

    STEM_KEYWORDS = {
        # Science domain
        "vật lý": [
            "điện trở",
            "tụ điện",
            "cuộn cảm",
            "gia tốc",
            "vận tốc",
            "động năng",
            "thế năng",
            "lượng tử",
            "photon",
            "electron",
            "từ trường",
            "điện trường",
            "cường độ dòng điện",
            "hiệu điện thế",
            "công suất",
            "tần số",
            "bước sóng",
            "quang học",
            "nhiệt độ tuyệt đối",
            "áp suất",
            "lực",
            "khối lượng",
            "quán tính",
            "ma sát",
        ],
        "hóa học": [
            "phản ứng hóa học",
            "nồng độ mol",
            "khối lượng mol",
            "phương trình hóa học",
            "cân bằng hóa học",
            "oxi hóa khử",
            "axit bazơ",
            "pH",
            "este",
            "hidrocacbon",
            "polyme",
            "nguyên tử khối",
            "số oxi hóa",
            "chất xúc tác",
            "tốc độ phản ứng",
            "hợp chất",
            "nguyên tố",
            "ion",
            "liên kết",
        ],
        "sinh học": [
            "gen",
            "nhiễm sắc thể",
            "DNA",
            "RNA",
            "enzym",
            "protein",
            "quang hợp",
            "hô hấp tế bào",
            "phân bào",
            "đột biến",
            "di truyền",
            "tế bào",
            "mô",
            "cơ quan",
            "hệ cơ quan",
            "sinh thái",
            "quần xã",
        ],
        "toán học": [
            "phương trình",
            "bất phương trình",
            "đạo hàm",
            "tích phân",
            "giới hạn",
            "hàm số",
            "đồ thị",
            "véc tơ",
            "ma trận",
            "định thức",
            "chuỗi",
            "số học",
        ],
        "công nghệ": [
            "socket",
            "thread",
            "algorithm",
            "thuật toán",
            "độ phức tạp",
            "big o",
            "binary tree",
            "linked list",
            "hash table",
            "database",
            "sql",
            "api",
            "http",
            "tcp/ip",
            "mạng",
            "server",
            "client",
        ],
    }
    STEM_UNITS = [
        "m/s",
        "km/h",
        "m/s²",
        "Ω",
        "°C",
        "K",
        "Pa",
        "Hz",
        "eV",
        "J",
        "W",
        "V",
        "A",
        "mol/L",
        "g/mol",
        "nm",
        "μm",
    ]
    PRECISION_KEYWORDS = {
        # Finance domain
        "tài chính": [
            "lãi suất thực",
            "lãi suất danh nghĩa",
            "GDP",
            "GNP",
            "lạm phát",
            "giảm phát",
            "P/E",
            "ROE",
            "ROA",
            "NPV",
            "IRR",
            "chi phí cơ hội",
            "chi phí chìm",
            "khấu hao",
            "EBITDA",
            "dòng tiền chiết khấu",
            "đầu tư",
            "cổ phiếu",
            "trái phiếu",
            "ngân hàng",
            "tín dụng",
            "vay vốn",
            "lãi suất",
            "thị trường",
            "kinh tế",
            "tiền tệ",
            "ngoại hối",
        ],
        "kế toán": [
            "bảng cân đối kế toán",
            "báo cáo tài chính",
            "tài sản ngắn hạn",
            "tài sản dài hạn",
            "nợ phải trả",
            "vốn chủ sở hữu",
            "doanh thu thuần",
            "giá vốn hàng bán",
            "lợi nhuận gộp",
            "lợi nhuận ròng",
            "kế toán",
            "thuế",
        ],
        # Health domain
        "y tế": [
            "bệnh",
            "triệu chứng",
            "điều trị",
            "thuốc",
            "dược",
            "bệnh viện",
            "bác sĩ",
            "chẩn đoán",
            "khám bệnh",
            "sức khỏe",
            "dinh dưỡng",
            "vitamin",
            "kháng sinh",
            "vaccine",
            "tiêm chủng",
            "y học",
            "phòng bệnh",
            "chăm sóc",
        ],
        "logic": [
            "xác suất",
            "tổ hợp",
            "chỉnh hợp",
            "hoán vị",
            "biến cố",
            "không gian mẫu",
            "kỳ vọng",
            "phương sai",
            "độ lệch chuẩn",
            "phân phối chuẩn",
        ],
    }
    MULTI_DOMAIN_KEYWORDS = {
        # People_and_Society domain
        "lịch sử": [
            "triều đại",
            "vua",
            "chiến tranh",
            "cách mạng",
            "khởi nghĩa",
            "thế kỷ",
            "thời kỳ",
            "sự kiện lịch sử",
            "lịch sử",
            "năm",
            "thời đại",
            "nhà",  # Nhà Trần, Nhà Lê
            "hoàng đế",
            "vương triều",
        ],
        "địa lý": [
            "tỉnh",
            "thành phố",
            "sông",
            "núi",
            "đồng bằng",
            "cao nguyên",
            "khí hậu",
            "địa hình",
            "địa lý",
            "vùng",
            "miền",
            "đồi",
            "biển",
            "đảo",
            "hồ",
            "thác",
        ],
        # Books_and_Literature domain
        "văn học": [
            "tác phẩm",
            "tác giả",
            "thơ",
            "văn xuôi",
            "truyện",
            "nhân vật",
            "tư tưởng nghệ thuật",
            "văn học",
            "tiểu thuyết",
            "truyện ngắn",
            "thể thơ",
            "ca dao",
            "tục ngữ",
            "sách",
        ],
        # People_and_Society domain
        "triết học": [
            "tư tưởng hồ chí minh",
            "chủ nghĩa",
            "học thuyết",
            "triết học",
            "triết lý",
            "đạo đức",
            "nhân sinh quan",
            "thế giới quan",
        ],
        "chính trị": [
            "chính trị",
            "đảng",
            "nhà nước",
            "chính phủ",
            "quốc hội",
            "chính sách",
            "đối ngoại",
            "ngoại giao",
            "hiệp ước",
        ],
        "văn hóa": [
            "văn hóa",
            "truyền thống",
            "phong tục",
            "tập quán",
            "lễ hội",
            "tôn giáo",
            "tín ngưỡng",
            "nghi lễ",
            "tục lệ",
            "di sản",
        ],
        # Arts_and_Entertainment domain
        "nghệ thuật": [
            "nghệ thuật",
            "hội họa",
            "điêu khắc",
            "kiến trúc",
            "âm nhạc",
            "ca múa nhạc",
            "sân khấu",
            "điện ảnh",
            "phim",
            "tranh",
            "tượng",
        ],
        # Jobs_and_Education domain
        "giáo dục": [
            "giáo dục",
            "trường học",
            "học sinh",
            "sinh viên",
            "giáo viên",
            "giảng dạy",
            "đào tạo",
            "chương trình",
            "môn học",
            "bài học",
        ],
        "nghề nghiệp": [
            "nghề",
            "nghề nghiệp",
            "công việc",
            "làm việc",
            "tuyển dụng",
            "nhân sự",
            "lương",
            "thu nhập",
        ],
        # News domain
        "tin tức": [
            "tin tức",
            "báo chí",
            "thông tin",
            "sự kiện",
            "hiện tại",
            "mới",
        ],
    }

    @classmethod
    def classify(cls, question_text: str, choices: List[str] = None) -> Dict[str, Any]:
        """
        Phân loại câu hỏi theo thứ tự ưu tiên:
        1. RAG (có context dài sẵn)
        2. COMPULSORY (pháp luật/safety) - cần RAG từ Law_and_Government/Sensitive_Subjects
        3. STEM (khoa học) - cần RAG từ Science
        4. PRECISION_CRITICAL (tài chính/y tế) - cần RAG từ Finance/Health
        5. MULTI_DOMAIN (kiến thức chung) - cần RAG từ các domain khác
        """
        # 1. Kiểm tra RAG question (có context sẵn)
        if cls._is_rag_question(question_text):
            return cls._extract_rag_question(question_text)

        # 2. Kiểm tra COMPULSORY (pháp luật, safety)
        compulsory_result = cls._is_compulsory_question(question_text, choices)
        if compulsory_result["is_compulsory"]:
            return {
                "type": "COMPULSORY",
                "question": question_text,
                "subtype": compulsory_result["subtype"],
            }

        # 3. Kiểm tra STEM (khoa học)
        stem_result = cls._is_stem_question(question_text)
        if stem_result["is_stem"]:
            return {
                "type": "STEM",
                "question": question_text,
                "subtype": stem_result["subtype"],
            }

        # 4. Kiểm tra PRECISION_CRITICAL (tài chính, y tế, logic)
        precision_result = cls._is_precision_critical(question_text)
        if precision_result["is_precision"]:
            return {
                "type": "PRECISION_CRITICAL",
                "question": question_text,
                "subtype": precision_result["subtype"],
            }

        # 5. Mặc định MULTI_DOMAIN
        return {
            "type": "MULTI_DOMAIN",
            "question": question_text,
            "subtype": cls._detect_multi_domain_subtype(question_text),
        }

    @classmethod
    def _is_rag_question(cls, text: str) -> bool:
        has_indicator = any(indicator in text for indicator in cls.RAG_INDICATORS)
        is_long = len(text) > 500
        return has_indicator or is_long

    @classmethod
    def _extract_rag_question(cls, text: str) -> Dict[str, Any]:
        if "Câu hỏi:" in text:
            parts = text.rsplit("Câu hỏi:", 1)
            return {
                "type": "RAG",
                "context": parts[0].strip(),
                "question": parts[1].strip() if len(parts) > 1 else text,
            }
        return {
            "type": "RAG",
            "context": text,
            "question": text,
        }

    @classmethod
    def _is_compulsory_question(
        cls, text: str, choices: List[str] = None
    ) -> Dict[str, Any]:
        """
        Kiểm tra xem câu hỏi có phải COMPULSORY không
        Returns: {"is_compulsory": bool, "subtype": str}
        """
        text_lower = text.lower()

        # Đếm điểm cho từng subtype
        subtype_scores = {}
        for domain, keywords in cls.COMPULSORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                subtype_scores[domain] = score

        # Kiểm tra choices có chứa refusal patterns không
        has_refusal_choice = False
        if choices:
            refusal_patterns = [
                "tôi không thể",
                "tôi từ chối",
                "không thể chia sẻ",
                "không thể trả lời",
                "không thể cung cấp",
            ]
            choices_text = " ".join(choices).lower()
            has_refusal_choice = any(
                pattern in choices_text for pattern in refusal_patterns
            )
            if has_refusal_choice:
                subtype_scores["safety"] = subtype_scores.get("safety", 0) + 2

        is_compulsory = bool(subtype_scores)

        # Xác định subtype
        if is_compulsory:
            subtype = max(subtype_scores, key=subtype_scores.get)
        else:
            subtype = "general"

        return {"is_compulsory": is_compulsory, "subtype": subtype}

    @classmethod
    def _is_stem_question(cls, text: str) -> Dict[str, bool]:
        text_lower = text.lower()
        has_math_symbol = any(sym in text for sym in cls.STEM_MATH_SYMBOLS)
        has_unit = any(unit in text for unit in cls.STEM_UNITS)
        subtype_scores = {}
        for domain, keywords in cls.STEM_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                subtype_scores[domain] = score
        has_code = bool(re.search(r"(def |class |function |import |#include)", text))
        if has_code:
            subtype_scores["công nghệ"] = subtype_scores.get("công nghệ", 0) + 3
        is_stem = has_math_symbol or has_unit or bool(subtype_scores)
        subtype = (
            max(subtype_scores, key=subtype_scores.get) if subtype_scores else "general"
        )
        return {"is_stem": is_stem, "subtype": subtype}

    @classmethod
    def _is_precision_critical(cls, text: str) -> Dict[str, bool]:
        text_lower = text.lower()
        subtype_scores = {}
        for domain, keywords in cls.PRECISION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                subtype_scores[domain] = score
        financial_numbers = re.findall(
            r"\d+[.,]?\d*\s*(%|USD|VND|\$|đồng|triệu|tỷ)", text
        )
        if len(financial_numbers) >= 2:
            subtype_scores["tài chính"] = subtype_scores.get("tài chính", 0) + 2
        is_precision = bool(subtype_scores)
        subtype = (
            max(subtype_scores, key=subtype_scores.get) if subtype_scores else "general"
        )
        return {"is_precision": is_precision, "subtype": subtype}

    @classmethod
    def _detect_multi_domain_subtype(cls, text: str) -> str:
        """
        Phát hiện subtype cho MULTI_DOMAIN
        Returns: subtype name (lịch sử, địa lý, văn học, etc.)
        """
        text_lower = text.lower()
        domain_counts = {}

        # Đếm điểm cho từng domain
        for domain, keywords in cls.MULTI_DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_counts[domain] = score

        # Trả về domain có điểm cao nhất
        if domain_counts:
            max_domain = max(domain_counts, key=domain_counts.get)
            return max_domain

        return "general"
