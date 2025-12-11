# VNPT RAG System

Hệ thống RAG (Retrieval-Augmented Generation) đánh giá hiệu năng của các mô hình AI VNPT và OpenAI trên bộ dữ liệu Q&A với nhiều chiến lược tối ưu.

---

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Cài đặt môi trường](#-cài-đặt-môi-trường)
- [Cấu hình API Keys](#-cấu-hình-api-keys)
- [Cách sử dụng](#-cách-sử-dụng)
- [Các chiến lược đánh giá](#-các-chiến-lược-đánh-giá)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)

---

## ✨ Tính năng chính

- ✅ **Unified Pipeline**: Hệ thống đánh giá thống nhất với 4 chiến lược
- ✅ **Multi-Provider**: Hỗ trợ cả VNPT AI và OpenAI models
- ✅ **Improved Prompts**: Tối ưu prompts cho từng loại câu hỏi (MATH, CONTEXT, KNOWLEDGE)
- ✅ **Hybrid Strategy**: Kết hợp model lớn và nhỏ để tối ưu chi phí/hiệu năng
- ✅ **Safety Classifier**: Phát hiện và xử lý câu hỏi không an toàn
- ✅ **Semantic Filtering**: Lọc context dựa trên độ tương đồng ngữ nghĩa
- ✅ **Flexible Configuration**: Cấu hình linh hoạt qua command line

---

## 🐍 Cài đặt môi trường

### 1. Cài đặt Conda (nếu chưa có)

```bash
# Tải Miniconda (phiên bản nhẹ của Conda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Khởi động lại terminal hoặc chạy
source ~/.bashrc
```

### 2. Tạo môi trường Conda mới

```bash
# Tạo môi trường với Python 3.10
conda create -n vnpt-ai python=3.10 -y

# Kích hoạt môi trường
conda activate vnpt-ai
```

### 3. Cài đặt các thư viện cần thiết

```bash
# Cài đặt các package chính
pip install requests openai numpy
```

**Các thư viện cần thiết:**
- `requests` - Gọi API HTTP
- `openai` - OpenAI Python SDK
- `numpy` - Xử lý vector cho semantic filtering

---

## 🔑 Cấu hình API Keys

### VNPT AI API Configuration

File: `src/configs/api-keys.json`

**Cấu trúc:**

```json
[
  {
    "authorization": "Bearer <JWT_TOKEN>",
    "tokenKey": "<TOKEN_KEY>",
    "llmApiName": "LLM large",
    "tokenId": "<TOKEN_ID>"
  },
  {
    "authorization": "Bearer <JWT_TOKEN>",
    "tokenKey": "<TOKEN_KEY>",
    "llmApiName": "LLM small",
    "tokenId": "<TOKEN_ID>"
  },
  {
    "authorization": "Bearer <JWT_TOKEN>",
    "tokenKey": "<TOKEN_KEY>",
    "llmApiName": "LLM embedings",
    "tokenId": "<TOKEN_ID>"
  }
]
```

**Lưu ý:**
- Index 0: Model **large** (mạnh nhất, cho câu hỏi khó)
- Index 1: Model **small** (nhẹ hơn, cho câu hỏi đơn giản)
- Index 2: Model **embeddings** (tạo vector cho semantic search)

**Cách thiết lập:**

```bash
cd src/configs
cp api-keys.json.example api-keys.json
# Sau đó chỉnh sửa file api-keys.json với thông tin của bạn
```

---

### OpenAI API Configuration

File: `src/configs/openai_config.json`

**Cấu trúc:**

```json
{
  "openai": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-proj-xxxxxxxxxxxx",
    "model": "gpt-4o-mini"
  }
}
```

**Các trường:**
- `base_url`: URL endpoint của OpenAI API
- `api_key`: API key của bạn từ [OpenAI Platform](https://platform.openai.com/api-keys)
- `model`: Tên model mặc định (ví dụ: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`)

**Cách thiết lập:**

```bash
cd src/configs
cp openai_config.json.example openai_config.json
# Sau đó chỉnh sửa file openai_config.json với API key của bạn
```

---

## 🚀 Cách sử dụng

### Quick Start

```bash
# Kích hoạt môi trường
conda activate vnpt-ai

# Chạy với Hybrid strategy và VNPT models (Khuyến nghị)
python main.py --strategy hybrid --provider vnpt

# Chạy với OpenAI models
python main.py --strategy hybrid --provider openai

# Test nhanh với 50 câu hỏi
python main.py --strategy hybrid --max-questions 50 --verbose
```

### Các tham số dòng lệnh

#### Strategy Arguments

| Tham số | Mô tả | Giá trị | Mặc định |
|---------|-------|---------|----------|
| `--strategy` | Chiến lược đánh giá | `baseline`, `hybrid`, `cost-optimized`, `quality-optimized` | `baseline` |
| `--provider` | Nhà cung cấp model | `vnpt`, `openai` | `vnpt` |
| `--large-model` | Tên model lớn | `large` (VNPT), `gpt-4o-mini` (OpenAI) | `large` |
| `--small-model` | Tên model nhỏ | `small` (VNPT), `gpt-3.5-turbo` (OpenAI) | `small` |

#### Data Arguments

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--input` | File JSON chứa câu hỏi | `data/val.json` |
| `--max-questions` | Số câu hỏi tối đa để đánh giá | Tất cả |
| `--start-index` | Index bắt đầu (0-based) | `0` |
| `--end-index` | Index kết thúc (exclusive) | Cuối danh sách |

#### Output Arguments

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--verbose` | In chi tiết cho mỗi câu hỏi | `False` |
| `--no-save` | Không lưu kết quả vào file | `False` |
| `--save-predictions` | Lưu predictions vào CSV | `False` |
| `--output-dir` | Thư mục lưu kết quả | `.` (thư mục hiện tại) |

#### Prompt Arguments

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--no-improved-prompts` | Tắt improved prompts | `False` |

---

## 📊 Các chiến lược đánh giá

### 1. Baseline Strategy

Sử dụng model lớn cho tất cả câu hỏi.

```bash
python main.py --strategy baseline --provider vnpt
```

**Đặc điểm:**
- ✅ Accuracy cao nhất
- ❌ Chi phí cao nhất
- 🎯 Phù hợp: Đánh giá baseline, so sánh hiệu năng

### 2. Hybrid Strategy ⭐ (Khuyến nghị)

Kết hợp model lớn và nhỏ dựa trên độ phức tạp câu hỏi.

```bash
python main.py --strategy hybrid --provider vnpt
```

**Đặc điểm:**
- ✅ Cân bằng tốt giữa chi phí và accuracy
- ✅ Tiết kiệm ~35% chi phí so với baseline
- ✅ Giữ được ~90% accuracy
- 🎯 Phù hợp: Production, sử dụng thực tế

**Logic:**
- Câu hỏi MATH (khó) → Model lớn
- Câu hỏi CONTEXT (đơn giản) → Model nhỏ
- Câu hỏi KNOWLEDGE (trung bình) → Model lớn

### 3. Cost-Optimized Strategy

Ưu tiên giảm chi phí, chấp nhận giảm accuracy.

```bash
python main.py --strategy cost-optimized --provider vnpt
```

**Đặc điểm:**
- ✅ Tiết kiệm ~50% chi phí
- ❌ Accuracy giảm ~5-7%
- 🎯 Phù hợp: Môi trường dev/test, budget hạn chế

### 4. Quality-Optimized Strategy

Ưu tiên accuracy cao nhất, không quan tâm chi phí.

```bash
python main.py --strategy quality-optimized --provider vnpt
```

**Đặc điểm:**
- ✅ Accuracy cao nhất (~95%)
- ❌ Chi phí cao nhất
- ✅ Sử dụng improved prompts và safety checks
- 🎯 Phù hợp: Đánh giá cuối cùng, demo quan trọng

---

## 📈 So sánh hiệu năng

| Chiến lược | Chi phí | Accuracy | MATH | CONTEXT | KNOWLEDGE |
|------------|---------|----------|------|---------|-----------|
| **Baseline** | 100% | ~92% | 85% | 100% | 90% |
| **Hybrid** ⭐ | 65% | ~90% | 85%+ | 100% | 88% |
| **Cost-Optimized** | 50% | ~85% | 80% | 95% | 85% |
| **Quality-Optimized** | 100% | ~95% | 95% | 100% | 95% |

---

## 💡 Ví dụ sử dụng

### Đánh giá cơ bản

```bash
# Đánh giá với VNPT models
python main.py --strategy hybrid --provider vnpt --input data/val.json

# Đánh giá với OpenAI models
python main.py --strategy hybrid --provider openai \
  --large-model gpt-4o-mini --small-model gpt-3.5-turbo
```

### Đánh giá với giới hạn

```bash
# Đánh giá 100 câu hỏi đầu tiên
python main.py --strategy hybrid --max-questions 100

# Đánh giá từ câu 10 đến câu 50
python main.py --strategy hybrid --start-index 10 --end-index 50
```

### Đánh giá với verbose và lưu kết quả

```bash
# Verbose mode để debug
python main.py --strategy hybrid --max-questions 10 --verbose

# Lưu predictions vào CSV
python main.py --strategy hybrid --save-predictions

# Lưu kết quả vào thư mục cụ thể
python main.py --strategy hybrid --output-dir ./results
```

### So sánh các chiến lược

```bash
# Chạy tất cả các chiến lược để so sánh
python main.py --strategy baseline --max-questions 100
python main.py --strategy hybrid --max-questions 100
python main.py --strategy cost-optimized --max-questions 100
python main.py --strategy quality-optimized --max-questions 100
```

### So sánh VNPT vs OpenAI

```bash
# VNPT models
python main.py --strategy hybrid --provider vnpt --max-questions 50

# OpenAI models
python main.py --strategy hybrid --provider openai \
  --large-model gpt-4o-mini --small-model gpt-3.5-turbo --max-questions 50
```

### Tắt improved prompts để so sánh

```bash
# Với improved prompts (mặc định)
python main.py --strategy hybrid --max-questions 50

# Không dùng improved prompts
python main.py --strategy hybrid --no-improved-prompts --max-questions 50
```

---

## 📁 Cấu trúc dự án

```
vnpt_rag_system/
├── main.py                      # Entry point chính
├── README.MD                    # File này
├── data/
│   ├── val.json                # Bộ dữ liệu validation
│   └── test.json               # Bộ dữ liệu test
└── src/
    ├── __init__.py
    ├── clients/                # API clients
    │   ├── __init__.py
    │   ├── factory.py          # Factory pattern cho clients
    │   ├── vnpt.py             # VNPT AI client
    │   └── openai.py           # OpenAI client
    ├── components/             # Core components
    │   ├── __init__.py
    │   ├── classifier.py       # Question classifier
    │   ├── extractor.py        # Answer extractor
    │   ├── model.py            # Model wrapper
    │   ├── prompt_builder.py   # Basic prompt builder
    │   ├── improved_prompt_builder.py  # Improved prompts ⭐
    │   ├── retrieval.py        # Semantic context filter
    │   ├── safety_classifier.py # Safety classifier
    │   └── vector_db.py        # Vector database
    ├── configs/                # Configuration files
    │   ├── api-keys.json       # VNPT API keys (không commit)
    │   ├── api-keys.json.example
    │   ├── openai_config.json  # OpenAI config (không commit)
    │   └── openai_config.json.example
    ├── pipelines/              # Evaluation pipelines
    │   ├── __init__.py
    │   ├── standard.py         # Standard pipeline
    │   ├── hybrid.py           # Hybrid pipeline
    │   └── unified.py          # Unified pipeline ⭐
    └── utils/                  # Utility functions
        ├── __init__.py
        └── format_choices.py   # Dynamic choices formatter
```

---

## 🔍 Chi tiết các components

### Clients (`src/clients/`)

- **factory.py**: Factory pattern để tạo client phù hợp
- **vnpt.py**: Client cho VNPT AI API
- **openai.py**: Client cho OpenAI API

### Components (`src/components/`)

- **classifier.py**: Phân loại câu hỏi (MATH, CONTEXT, KNOWLEDGE)
- **safety_classifier.py**: Phát hiện câu hỏi không an toàn
- **extractor.py**: Trích xuất câu trả lời từ output của model
- **prompt_builder.py**: Xây dựng prompt cơ bản
- **improved_prompt_builder.py**: Prompt được tối ưu cho từng loại câu hỏi
- **retrieval.py**: Lọc context dựa trên semantic similarity
- **vector_db.py**: Vector database cho embeddings
- **model.py**: Wrapper thống nhất cho các model

### Pipelines (`src/pipelines/`)

- **standard.py**: Pipeline cơ bản (baseline)
- **hybrid.py**: Pipeline kết hợp model lớn và nhỏ
- **unified.py**: Pipeline thống nhất với 4 chiến lược

---

## 📝 Format dữ liệu

### Input Format (JSON)

```json
[
  {
    "id": "question_001",
    "question": "Câu hỏi của bạn?",
    "choices": ["A. Đáp án A", "B. Đáp án B", "C. Đáp án C", "D. Đáp án D"],
    "answer": "A",
    "context": "Context liên quan đến câu hỏi..."
  }
]
```

### Output Format

**Console Output:**

```
=== EVALUATION RESULTS ===
Strategy: hybrid
Provider: vnpt
Total questions: 100
Correct: 90
Accuracy: 90.00%
Total time: 45.23s
Average time: 0.45s/question
```

**CSV Output** (nếu dùng `--save-predictions`):

```csv
id,question,predicted,actual,correct
question_001,"Câu hỏi?",A,A,True
question_002,"Câu hỏi?",B,A,False
```

---

## 🐛 Troubleshooting

### Module Import Errors

```bash
# Đảm bảo bạn đang ở thư mục gốc của project
cd /path/to/vnpt_rag_system

# Chạy từ thư mục gốc
python main.py --strategy hybrid
```

### File Not Found Errors

```bash
# Sử dụng đường dẫn đúng từ thư mục gốc
python main.py --input data/val.json  # ✅ Đúng
python main.py --input ../data/val.json  # ❌ Sai (nếu đang ở thư mục gốc)
```

### API Key Errors

```bash
# Kiểm tra file config tồn tại
ls src/configs/api-keys.json
ls src/configs/openai_config.json

# Kiểm tra format JSON hợp lệ
python -m json.tool src/configs/api-keys.json
```

---

## 📚 Tài liệu tham khảo

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [VNPT AI Platform](https://ai.vnpt.vn)

---

## 👥 Contributors

- VNPT AI Team

---

## 📄 License

Internal use only - VNPT Corporation
