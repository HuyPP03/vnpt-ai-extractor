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

## 📁 Cấu trúc dự án

```
VNPT_repo/
├── api-keys.json           # Cấu hình VNPT AI API (không commit)
├── api-keys.json.example   # Template cho api-keys.json
├── openai_config.json      # Cấu hình OpenAI API (không commit)
├── openai_config.json.example # Template cho openai_config.json
├── evaluate.py             # Script đánh giá chính
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
│   ├── pipeline.py         # Pipeline đánh giá
│   ├── prompt_builder.py   # Xây dựng prompt
│   └── vector_db.py        # Vector database
└── README.MD               # File này
```

---
