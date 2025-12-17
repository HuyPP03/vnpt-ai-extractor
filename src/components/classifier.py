from typing import Dict, Any
import re


class QuestionClassifier:

    CONTEXT_INDICATORS = [
        "Đoạn thông tin:",
        "Tiêu đề:",
        "Title:",
        "Document 1",
        "Document 2",
        "-- Đoạn văn",
        "Văn bản:",
        "Bài đọc:",
        "Context:",
        "Passage:",
    ]

    MATH_SYMBOLS = [
        "$",
        "\\",
        "\\frac",
        "\\int",
        "\\sum",
        "\\sqrt",
        "\\pi",
        "^",
        "²",
        "³",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\cot",
        "\\sec",
        "\\csc",
        "\\sinh",
        "\\cosh",
        "\\tanh",
        "\\coth",
        "\\sech",
        "\\csch",
        "≈",
        "≤",
        "≥",
        "≠",
        "∞",
        "°",
        "α",
        "β",
        "γ",
        "Δ",
        "∑",
        "∏",
    ]

    MATH_KEYWORDS = [
        # Toán học
        "tính toán",
        "tính",
        "giá trị kỳ vọng",
        "xác suất",
        "đạo hàm",
        "tích phân",
        "công thức",
        "phương trình",
        "bất phương trình",
        "hàm số",
        "ma trận",
        "vector",
        "định thức",
        "số phức",
        "giới hạn",
        "cực trị",
        "tiệm cận",
        "đồ thị",
        "tọa độ"
        # Vật lý
        "vận tốc",
        "gia tốc",
        "động năng",
        "thế năng",
        "nhiệt độ",
        "áp suất",
        "điện trở",
        "cường độ dòng điện",
        "hiệu điện thế",
        "từ trường",
        "điện trường",
        "quang học",
        "tần số",
        "bước sóng",
        "sóng dọc",
        "sóng ngang",
        "khối lượng riêng",
        "gia tốc trọng trường",
        "động lượng",
        "xung lượng",
        # Hóa học
        "mol",
        "nồng độ",
        "khối lượng mol",
        "nguyên tử",
        "phân tử",
        "ion",
        "hóa trị",
        "oxi hóa",
        "chất tan",
        "dung môi",
        "cân bằng hóa học",
        "tốc độ phản ứng",
        "chất xúc tác",
        "electron",
        "este",
        "axit",
        "bazơ",
        "nguyên tố",
        "bảng tuần hoàn",
        "đồng vị",
        "polyme",
        "hidrocacbon",
        # Kinh tế
        "tổng chi phí",
        "vốn lưu động",
        "lợi nhuận",
        "tỷ lệ",
        "phần trăm",
        "chi phí khấu hao",
        "lợi nhuận",
        # Đo lường
        "khoảng cách",
        "diện tích",
        "thể tích",
        "độ dài",
        "khối lượng",
        "thời gian",
        "tốc độ",
        "chu vi",
        "bán kính",
        "đường kính",
    ]

    CHEMISTRY_PATTERNS = [
        r"[A-Z][a-z]?\d+",  # Công thức hóa học có số: H2O, CO2, H2SO4
        r"\d+%",  # Phần trăm: 40%, 25%
    ]

    PHYSICS_UNITS = [
        "m/s",
        "km/h",
        "m/s²",
        "Ω",
        "°C",
        "Pa",
        "Hz",
        "eV",
        "kg",
        "mol",
        "mL",
    ]

    @classmethod
    def classify(cls, question_text: str) -> Dict[str, Any]:

        # 1. Check CONTEXT TYPE
        if any(indicator in question_text for indicator in cls.CONTEXT_INDICATORS):
            return cls._extract_context_question(question_text)

        # 2. Check MATH TYPE
        if cls._is_math_question(question_text):
            return {"type": "MATH", "question": question_text}

        # 3. Default to KNOWLEDGE TYPE with subtype classification
        subtype = cls._classify_knowledge_subtype(question_text)
        return {"type": "KNOWLEDGE", "question": question_text, "subtype": subtype}

    @classmethod
    def _extract_context_question(cls, text: str) -> Dict[str, Any]:
        """Tách context và câu hỏi"""
        if "Câu hỏi:" in text:
            parts = text.rsplit("Câu hỏi:", 1)
            return {
                "type": "CONTEXT",
                "context": parts[0].strip(),
                "question": parts[1].strip() if len(parts) > 1 else text,
            }
        return {"type": "CONTEXT", "context": text, "question": text}

    @classmethod
    def _is_math_question(cls, text: str) -> bool:

        # 1. Kiểm tra ký hiệu toán học
        has_math_symbol = any(sym in text for sym in cls.MATH_SYMBOLS)

        # 2. Kiểm tra từ khóa STEM
        text_lower = text.lower()
        has_math_keyword = any(kw in text_lower for kw in cls.MATH_KEYWORDS)

        # 3. Kiểm tra đơn vị vật lý
        has_physics_unit = any(unit in text for unit in cls.PHYSICS_UNITS)

        # 4. Kiểm tra công thức hóa học
        has_chemistry = any(
            re.search(pattern, text) for pattern in cls.CHEMISTRY_PATTERNS
        )

        # 5. Kiểm tra nhiều số (đặc trưng của bài toán tính toán)
        numbers = re.findall(r"\d+[.,]?\d*%?", text)
        has_multiple_numbers = len(numbers) >= 2

        # 6. Phát hiện công thức hóa học (chữ in hoa + số)
        chemical_formulas = re.findall(r"\b[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*\b", text)
        has_chemical_formula = len(chemical_formulas) >= 1

        # Logic tổng hợp
        return (
            has_math_symbol
            or has_physics_unit
            or has_chemical_formula
            or (has_math_keyword and has_multiple_numbers)
            or (has_chemistry and has_math_keyword)
        )

    @classmethod
    def _classify_knowledge_subtype(cls, text: str) -> str:
        classifier = ImprovedKnowledgeClassifier()
        result = classifier.classify_with_explanation(text, top_k=3)
        return result["best_match"]


class ImprovedKnowledgeClassifier:
    """
    Phân loại subtype cho câu hỏi KNOWLEDGE với scoring system
    """

    # Định nghĩa keywords với trọng số (weight)
    CATEGORY_KEYWORDS = {
        # COMPULSORY subtypes
        "safety": {
            "high": ["an toàn", "nguy hiểm", "độc hại", "cấm", "cảnh báo"],
            "medium": ["tai nạn", "phòng chống", "bảo vệ", "rủi ro", "tránh"],
            "low": ["cẩn thận", "chú ý"],
        },
        "pháp luật": {
            "high": [
                "luật",
                "nghị định",
                "thông tư",
                "bộ luật",
                "hiến pháp",
                "điều luật",
            ],
            "medium": ["quy định", "vi phạm", "xử phạt", "trách nhiệm pháp lý"],
            "low": ["điều", "khoản", "theo quy định"],
        },
        # STEM subtypes
        "vật lý": {
            "high": [
                "vật lý",
                "lực học",
                "động lực học",
                "nhiệt động lực học",
                "quang học",
                "điện từ học",
            ],
            "medium": [
                "lực",
                "năng lượng",
                "động năng",
                "thế năng",
                "điện",
                "từ trường",
                "ánh sáng",
            ],
            "low": ["chuyển động", "tốc độ", "gia tốc"],
        },
        "hóa học": {
            "high": ["hóa học", "phản ứng hóa học", "hợp chất", "nguyên tố"],
            "medium": ["axit", "bazơ", "muối", "dung dịch", "chất", "phân tử"],
            "low": ["hòa tan", "kết tủa", "pH"],
        },
        "sinh học": {
            "high": [
                "sinh học",
                "di truyền học",
                "tiến hóa",
                "sinh thái học",
                "vi sinh",
            ],
            "medium": ["tế bào", "gen", "DNA", "protein", "enzyme", "sinh vật"],
            "low": ["cơ thể", "động vật", "thực vật", "vi khuẩn"],
        },
        "toán học": {
            "high": [
                "toán học",
                "đại số",
                "hình học",
                "giải tích",
                "xác suất",
                "thống kê",
            ],
            "medium": ["phương trình", "hàm số", "đạo hàm", "tích phân", "ma trận"],
            "low": ["tính toán", "công thức", "số"],
        },
        "công nghệ": {
            "high": [
                "công nghệ thông tin",
                "khoa học máy tính",
                "lập trình",
                "thuật toán",
                "AI",
                "machine learning",
            ],
            "medium": ["phần mềm", "phần cứng", "internet", "mạng", "cơ sở dữ liệu"],
            "low": ["máy tính", "website", "ứng dụng"],
        },
        # PRECISION_CRITICAL subtypes
        "tài chính": {
            "high": [
                "tài chính",
                "đầu tư",
                "chứng khoán",
                "cổ phiếu",
                "trái phiếu",
                "quỹ đầu tư",
            ],
            "medium": ["ngân hàng", "tiền tệ", "lãi suất", "tín dụng", "vốn"],
            "low": ["tiền", "thu nhập", "chi tiêu"],
        },
        "kế toán": {
            "high": ["kế toán", "báo cáo tài chính", "bảng cân đối kế toán", "sổ sách"],
            "medium": [
                "tài sản",
                "nợ",
                "vốn chủ sở hữu",
                "doanh thu",
                "chi phí",
                "lợi nhuận",
            ],
            "low": ["khấu hao", "phân bổ"],
        },
        "y tế": {
            "high": [
                "y tế",
                "y học",
                "bệnh viện",
                "phòng khám",
                "chẩn đoán",
                "điều trị",
            ],
            "medium": ["bệnh", "thuốc", "triệu chứng", "sức khỏe", "bác sĩ"],
            "low": ["đau", "mệt mỏi", "khám"],
        },
        "logic": {
            "high": ["logic", "suy luận", "luận lý", "tam đoạn luận", "mệnh đề"],
            "medium": ["lập luận", "chứng minh", "giả thiết", "kết luận"],
            "low": ["suy nghĩ", "phân tích"],
        },
        # MULTI_DOMAIN subtypes
        "lịch sử": {
            "high": [
                "lịch sử",
                "sử",
                "triều đại",
                "cách mạng",
                "chiến tranh",
                "thời kỳ lịch sử",
            ],
            "medium": ["vua", "chúa", "vương triều", "nhà", "thế kỷ"],
            "low": ["năm", "thời", "quá khứ"],
        },
        "địa lý": {
            "high": [
                "địa lý",
                "bản đồ",
                "kinh tuyến",
                "vĩ tuyến",
                "địa hình",
                "khí hậu",
            ],
            "medium": [
                "vùng",
                "miền",
                "tỉnh",
                "thành phố",
                "sông",
                "núi",
                "biển",
                "đảo",
            ],
            "low": ["đồng bằng", "cao nguyên", "châu lục", "nước"],
        },
        "chính trị": {
            "high": [
                "chính trị",
                "chính quyền",
                "quốc hội",
                "chính phủ",
                "bầu cử",
                "dân chủ",
            ],
            "medium": ["đảng", "nhà nước", "chủ tịch", "thủ tướng", "bộ trưởng"],
            "low": ["lãnh đạo", "quyết định"],
        },
        "triết học": {
            "high": [
                "triết học",
                "triết lý",
                "siêu hình",
                "nhận thức luận",
                "bản thể luận",
            ],
            "medium": ["tư tưởng", "học thuyết", "quan điểm triết học", "triết gia"],
            "low": ["quan điểm", "nhìn nhận", "ý nghĩa"],
        },
        "văn hóa": {
            "high": ["văn hóa", "di sản văn hóa", "bản sắc văn hóa", "văn minh"],
            "medium": ["phong tục", "tập quán", "lễ hội", "truyền thống", "dân tộc"],
            "low": ["tín ngưỡng", "nghi lễ", "tục lệ"],
        },
        "văn học": {
            "high": [
                "văn học",
                "tác phẩm văn học",
                "trường phái văn học",
                "thể loại văn học",
            ],
            "medium": [
                "tác giả",
                "nhà văn",
                "thi sĩ",
                "thơ",
                "văn",
                "truyện",
                "tiểu thuyết",
            ],
            "low": ["viết", "sáng tác", "câu chuyện"],
        },
        "nghệ thuật": {
            "high": [
                "nghệ thuật",
                "mỹ thuật",
                "hội họa",
                "điêu khắc",
                "kiến trúc",
                "âm nhạc",
                "múa",
            ],
            "medium": ["tranh", "bức họa", "bài hát", "nhạc sĩ", "ca sĩ", "họa sĩ"],
            "low": ["đẹp", "sáng tạo", "thẩm mỹ"],
        },
        "giáo dục": {
            "high": [
                "giáo dục",
                "đào tạo",
                "chương trình giáo dục",
                "phương pháp giảng dạy",
            ],
            "medium": [
                "học sinh",
                "sinh viên",
                "trường",
                "đại học",
                "giáo viên",
                "giảng viên",
            ],
            "low": ["học", "dạy", "lớp"],
        },
        "nghề nghiệp": {
            "high": [
                "nghề nghiệp",
                "nghề",
                "ngành nghề",
                "tuyển dụng",
                "thị trường lao động",
            ],
            "medium": [
                "việc làm",
                "công việc",
                "lương",
                "thu nhập",
                "kỹ năng nghề nghiệp",
            ],
            "low": ["làm việc", "nhân viên"],
        },
        "tin tức": {
            "high": ["tin tức", "thời sự", "báo chí", "truyền thông", "sự kiện"],
            "medium": ["thông tin", "báo", "tạp chí", "phóng viên"],
            "low": ["mới", "hiện nay", "gần đây"],
        },
    }

    # Trọng số cho mỗi level
    WEIGHTS = {
        "high": 3.0,
        "medium": 1.5,
        "low": 0.5,
    }

    # Patterns đặc biệt
    SPECIAL_PATTERNS = {
        "pháp luật": [
            r"điều\s+\d+",  # điều 5, điều 10
            r"khoản\s+\d+",  # khoản 2, khoản 3
            r"luật\s+\w+",  # luật giao thông, luật hình sự
            r"nghị định\s+\d+",  # nghị định 100
        ],
        "toán học": [
            r"\d+\s*[\+\-\*\/\=]\s*\d+",  # phép tính: 2 + 3
            r"[a-z]\s*[\+\-\*\/\=]\s*[a-z]",  # biến: x + y
            r"[a-z]\^\d+",  # mũ: x^2
        ],
        "lịch sử": [
            r"năm\s+\d{3,4}",  # năm 1945
            r"thế kỷ\s+[IVX]+",  # thế kỷ XX
            r"triều\s+\w+",  # triều Nguyễn
        ],
    }

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Chuẩn hóa text"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _calculate_category_score(cls, text: str, category: str) -> float:
        """Tính điểm cho một category"""
        score = 0.0
        text_normalized = cls._normalize_text(text)

        # Điểm từ keywords
        if category in cls.CATEGORY_KEYWORDS:
            keywords_config = cls.CATEGORY_KEYWORDS[category]

            for level, keywords in keywords_config.items():
                weight = cls.WEIGHTS[level]
                for keyword in keywords:
                    # Count occurrences (nhưng không cộng quá nhiều)
                    count = min(text_normalized.count(keyword), 3)
                    if count > 0:
                        score += weight * count

        # Điểm từ special patterns
        if category in cls.SPECIAL_PATTERNS:
            for pattern in cls.SPECIAL_PATTERNS[category]:
                matches = re.findall(pattern, text_normalized)
                if matches:
                    score += 2.0 * min(len(matches), 3)

        return score

    @classmethod
    def _get_all_scores(cls, text: str) -> Dict[str, float]:
        """Tính điểm cho tất cả categories"""
        scores = {}
        for category in cls.CATEGORY_KEYWORDS.keys():
            score = cls._calculate_category_score(text, category)
            if score > 0:
                scores[category] = score
        return scores

    @classmethod
    def classify_with_confidence(
        cls, text: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Phân loại với confidence score

        Returns:
            List of (category, confidence_score) sorted by score
        """
        scores = cls._get_all_scores(text)

        if not scores:
            return [("general", 0.0)]

        # Normalize scores to 0-1
        max_score = max(scores.values())
        normalized_scores = {cat: score / max_score for cat, score in scores.items()}

        # Sort by score
        sorted_categories = sorted(
            normalized_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_categories[:top_k]

    @classmethod
    def classify_knowledge_subtype(cls, text: str, threshold: float = 0.3) -> str:
        """
        Phân loại subtype cho câu hỏi KNOWLEDGE

        Args:
            text: Câu hỏi cần phân loại
            threshold: Ngưỡng confidence tối thiểu (0-1)

        Returns:
            Subtype phù hợp nhất
        """
        results = cls.classify_with_confidence(text, top_k=1)

        if not results:
            return "general"

        best_category, confidence = results[0]

        # Nếu confidence quá thấp, trả về general
        if confidence < threshold:
            return "general"

        return best_category

    @classmethod
    def classify_with_explanation(cls, text: str, top_k: int = 3) -> Dict:
        """
        Phân loại kèm giải thích chi tiết

        Returns:
            Dictionary với thông tin chi tiết về classification
        """
        all_scores = cls._get_all_scores(text)
        top_results = cls.classify_with_confidence(text, top_k=top_k)

        return {
            "best_match": top_results[0][0] if top_results else "general",
            "confidence": top_results[0][1] if top_results else 0.0,
            "top_matches": [
                {
                    "category": cat,
                    "confidence": conf,
                    "raw_score": all_scores.get(cat, 0.0),
                }
                for cat, conf in top_results
            ],
            "all_scores": all_scores,
        }
