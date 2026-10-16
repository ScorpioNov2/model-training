import json
import os

# 1. Cấu hình các mốc giới hạn
LIMIT_TOKENS = 1024
LIMIT_WORDS = 700
# CORPUS_FILE = "corpus.json"
CORPUS_FILE = "corpus_clean.json"

# Khởi tạo bộ đếm token (sử dụng cl100k_base của GPT-4 / GPT-3.5 hoặc o1)
try:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    print("Chưa cài đặt 'tiktoken'. Kết quả đếm token sẽ dựa trên khoảng trắng tạm thời.")
    encoder = None

# 2. Đọc file dữ liệu corpus.json
if not os.path.exists(CORPUS_FILE):
    print(f"Lỗi: Không tìm thấy file '{CORPUS_FILE}' trong thư mục hiện tại!")
    exit()

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    try:
        corpus_data = json.load(f)
    except json.JSONDecodeError:
        print("Lỗi: File corpus.json không đúng định dạng JSON hoặc bị trống!")
        exit()

total_samples = len(corpus_data)
if total_samples == 0:
    print("File corpus không chứa mẫu dữ liệu nào.")
    exit()

# 3. Khởi tạo các biến thống kê
over_token_count = 0
under_token_count = 0
over_word_count = 0
under_word_count = 0

max_tokens = 0
max_words = 0
total_tokens = 0
total_words = 0

# 4. Duyệt qua từng mẫu để phân tích
for item in corpus_data:
    text = item.get("text", "")
    
    # Đếm số từ (dựa trên khoảng trắng)
    words_list = text.split()
    word_count = len(words_list)
    
    # Đếm số token
    if encoder:
        token_count = len(encoder.encode(text))
    else:
        # Cách tính thô nếu không có tiktoken (1 từ Tiếng Việt ~ 1.2 đến 1.5 token tùy tokenizer)
        token_count = int(word_count * 1.3) 
    
    # Cập nhật giá trị cực đại và tổng số
    max_tokens = max(max_tokens, token_count)
    max_words = max(max_words, word_count)
    total_tokens += token_count
    total_words += word_count
    
    # Thống kê theo mốc Token (1024)
    if token_count > LIMIT_TOKENS:
        over_token_count += 1
    else:
        under_token_count += 1
        
    # Thống kê theo mốc Từ (700)
    if word_count > LIMIT_WORDS:
        over_word_count += 1
    else:
        under_word_count += 1

# 5. In kết quả báo cáo ra màn hình
print("=" * 50)
print(f" BÁO CÁO PHÂN TÍCH ĐỘ DÀI CORPUS (Tổng số mẫu: {total_samples})")
print("=" * 50)

print(f"\n📊 THỐNG KÊ CHUNG:")
print(f"  - Tổng số lượng Token: {total_tokens:,}")
print(f"  - Tổng số lượng Từ: {total_words:,}")
print(f"  - Độ dài trung bình: {int(total_tokens/total_samples)} tokens / {int(total_words/total_samples)} từ mỗi mẫu")
print(f"  - Mẫu dài nhất: {max_tokens:,} tokens / {max_words:,} từ")

print(f"\n🔍 KIỂM TRA MỐC GIỚI HẠN {LIMIT_TOKENS} TOKENS:")
print(f"  - Số mẫu Nhỏ hơn hoặc Bằng {LIMIT_TOKENS} tokens: {under_token_count} mẫu ({under_token_count/total_samples*100:.2f}%) ✅ (An toàn)")
print(f"  - Số mẫu Vượt quá {LIMIT_TOKENS} tokens: {over_token_count} mẫu ({over_token_count/total_samples*100:.2f}%) ⚠️ (Có nguy cơ bị cắt cụt)")

print(f"\n🔍 KIỂM TRA MỐC GIỚI HẠN {LIMIT_WORDS} TỪ:")
print(f"  - Số mẫu Nhỏ hơn hoặc Bằng {LIMIT_WORDS} từ: {under_word_count} mẫu ({under_word_count/total_samples*100:.2f}%)")
print(f"  - Số mẫu Vượt quá {LIMIT_WORDS} từ: {over_word_count} mẫu ({over_word_count/total_samples*100:.2f}%)")
print("=" * 50)
