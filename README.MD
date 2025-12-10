# VNPT AI Model Evaluation

Dự án đánh giá hiệu năng của các mô hình AI VNPT trên bộ dữ liệu Q&A.

---

## 📋 Mục lục

- [Cài đặt môi trường](#-cài-đặt-môi-trường)
- [Cấu hình API Keys](#-cấu-hình-api-keys)
- [Cách chạy chương trình](#-cách-chạy-chương-trình)
- [Kết quả đánh giá](#-kết-quả-đánh-giá)

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
pip install requests openai

# Hoặc tạo file requirements.txt và cài đặt
pip install -r requirements.txt
```

**Các thư viện cần thiết:**
- `requests` - Gọi API HTTP
- `openai` - OpenAI Python SDK (nếu sử dụng model OpenAI)
- `numpy` - Xử lý vector cho semantic filtering

```bash
# Cài đặt đầy đủ
pip install requests openai numpy
```

---

## 🔑 Cấu hình API Keys

### File `api-keys.json` - Cấu hình cho VNPT AI API

File này chứa thông tin xác thực để kết nối với các model VNPT AI (small, large, embeddings).

**Cấu trúc file:**

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
- Index 0: Model **large** (mạnh nhất)
- Index 1: Model **small** (nhẹ hơn)
- Index 2: Model **embeddings** (tạo vector)

**Cách sử dụng:**
1. Copy file `api-keys.json.example` thành `api-keys.json`
2. Điền thông tin xác thực của bạn vào file

```bash
cp api-keys.json.example api-keys.json
# Sau đó chỉnh sửa file api-keys.json với thông tin của bạn
```

---

### File `openai_config.json` - Cấu hình cho OpenAI API

File này chứa thông tin xác thực để kết nối với OpenAI API.

**Cấu trúc file:**

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
- `api_key`: API key của bạn từ OpenAI
- `model`: Tên model sử dụng (ví dụ: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`)

**Cách sử dụng:**
1. Copy file `openai_config.json.example` thành `openai_config.json`
2. Điền API key của bạn vào file

```bash
cp openai_config.json.example openai_config.json
# Sau đó chỉnh sửa file openai_config.json với API key của bạn
```

---

## 🚀 Cách chạy chương trình

### Chạy đánh giá với `evaluate.py`

```bash
# Kích hoạt môi trường (nếu chưa)
conda activate vnpt-ai

# Chạy với model large (mặc định)
python evaluate.py

# Chạy với model small
python evaluate.py --model small

# Chạy với OpenAI
python evaluate.py --model openai
```

### Các tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--model` | Loại model: `small`, `large`, `openai` | `large` |
| `--input` | File JSON chứa câu hỏi | `val.json` |
| `--max-questions` | Số câu hỏi tối đa để đánh giá | Tất cả |
| `--start-index` | Index bắt đầu (0-based) | `0` |
| `--end-index` | Index kết thúc (exclusive) | Cuối danh sách |
| `--verbose` | In chi tiết cho mỗi câu hỏi | `False` |
| `--no-save` | Không lưu kết quả vào file | `False` |
| `--save-predictions` | Lưu predictions vào `predict.csv` | `False` |

### Ví dụ sử dụng

```bash
# Đánh giá 100 câu hỏi đầu tiên với model large
python evaluate.py --model large --input val.json --max-questions 100

# Đánh giá với chế độ verbose (chi tiết)
python evaluate.py --model small --input val.json --verbose

# Đánh giá và lưu predictions vào CSV
python evaluate.py --model openai --input test.json --save-predictions

# Đánh giá từ câu 10 đến câu 20
python evaluate.py --model large --input val.json --start-index 10 --end-index 20

# Đánh giá tất cả câu hỏi mà không lưu kết quả
python evaluate.py --model large --no-save
```

### Test API riêng lẻ

```bash
# Test VNPT AI API
python vnpt_ai_client.py

# Test OpenAI API
python openai_client.py
```

---

## 🎯 Tối ưu hóa Model (Mới!) ⭐

### Unified Pipeline - Hệ thống đánh giá thống nhất

**Tính năng mới:**
- ✅ Improved prompts cho MATH (tăng accuracy từ 75% → 85%+)
- ✅ Hỗ trợ cả VNPT AI và OpenAI models
- ✅ 4 strategies: baseline, hybrid, cost-optimized, quality-optimized
- ✅ Flexible configuration qua command line

### Quick Start

```bash
# Kích hoạt môi trường
conda activate vnpt-ai

# Hybrid strategy với VNPT models (Khuyến nghị)
python unified_pipeline.py --strategy hybrid

# Hybrid với OpenAI models
python unified_pipeline.py --strategy hybrid --provider openai

# Baseline để so sánh
python unified_pipeline.py --strategy baseline

# Test nhanh với 50 câu
python unified_pipeline.py --strategy hybrid --max-questions 50
```

### So sánh các chiến lược

| Chiến lược | Chi phí | Accuracy | MATH | CONTEXT | KNOWLEDGE |
|------------|---------|----------|------|---------|-----------|
| **Baseline** | 100% | ~92% | 85% | 100% | 90% |
| **Hybrid** ⭐ | 65% | ~90% | 85%+ | 100% | 88% |
| Cost-Optimized | 50% | ~85% | 80% | 95% | 85% |
| Quality-Optimized | 100% | ~95% | 95% | 100% | 95% |


### Ví dụ sử dụng

```bash
# Đánh giá full dataset và lưu predictions
python unified_pipeline.py --strategy hybrid --save-predictions

# Verbose mode để debug
python unified_pipeline.py --strategy hybrid --max-questions 10 --verbose

# So sánh VNPT vs OpenAI
python unified_pipeline.py --strategy hybrid --provider vnpt --max-questions 50
python unified_pipeline.py --strategy hybrid --provider openai \
  --large-model gpt-4o-mini --small-model gpt-3.5-turbo --max-questions 50

# Tắt improved prompts để so sánh
python unified_pipeline.py --strategy hybrid --no-improved-prompts
```


## 📁 Cấu trúc dự án

```
VNPT_repo/
├── api-keys.json           # Cấu hình VNPT AI API (không commit)
├── api-keys.json.example   # Template cho api-keys.json
├── openai_config.json      # Cấu hình OpenAI API (không commit)
├── openai_config.json.example # Template cho openai_config.json
├── evaluate.py             # Script đánh giá cơ bản
├── evaluate_hybrid.py      # Script đánh giá với Hybrid Pipeline
├── unified_pipeline.py     # Unified Pipeline - Hệ thống thống nhất (MỚI!) ⭐
├── vnpt_ai_client.py       # Client cho VNPT AI API
├── openai_client.py        # Client cho OpenAI API
├── val.json                # Bộ dữ liệu validation
├── test.json               # Bộ dữ liệu test
├── utils/
│   ├── classifier.py       # Phân loại câu hỏi
│   ├── context_filter.py   # Lọc context
│   ├── extractor.py        # Trích xuất câu trả lời
│   ├── format_choices.py   # Format các lựa chọn
│   ├── model.py            # Wrapper cho các model
│   ├── pipeline.py         # Pipeline đánh giá cơ bản
│   ├── hybrid_pipeline.py  # Hybrid Pipeline tối ưu
│   ├── prompt_builder.py   # Xây dựng prompt cơ bản
│   ├── improved_prompt_builder.py  # Improved prompts (MỚI!) ⭐
│   └── vector_db.py        # Vector database
├── README.MD               # File này
├── UNIFIED_PIPELINE_GUIDE.md    # Hướng dẫn Unified Pipeline (MỚI!) ⭐
├── OPENAI_SETUP.md         # Hướng dẫn OpenAI integration (MỚI!) ⭐
├── OPTIMIZATION_STRATEGY.md     # Chiến lược tối ưu chi tiết
└── test_openai_integration.py   # Test script cho OpenAI (MỚI!)
```