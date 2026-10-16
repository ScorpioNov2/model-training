# import json
# import os
# import re

# # 1. Cấu hình
# CORPUS_FILE = "corpus.json"
# OUTPUT_FILE = "corpus_clean.json"
# LIMIT_TOKENS = 1024
# LIMIT_WORDS = 700
# CRITICAL_TOKENS = 5000

# # Khởi tạo bộ đếm token
# try:
#     import tiktoken
#     encoder = tiktoken.get_encoding("cl100k_base")
# except ImportError:
#     encoder = None

# def count_tokens(text):
#     if encoder:
#         return len(encoder.encode(text))
#     return int(len(text.split()) * 1.3)

# def count_words(text):
#     return len(text.split())

# def parse_turns(text):
#     """
#     Tách chuỗi văn bản thành danh sách các lượt thoại đầy đủ.
#     Mỗi lượt thoại giữ nguyên cấu trúc: <|role|>\nNội dung<|end|>\n
#     """
#     # Loại bỏ token đặc biệt ở hai đầu để bóc tách lõi hội thoại
#     core_text = text.replace("[BOS]", "").replace("[EOS]", "").strip()
    
#     # Sử dụng Regex để tìm các khối đối thoại hoàn chỉnh dạng <|user|> hoặc <|assistant|> cho tới <|end|>
#     pattern = r"(<\|(?:user|assistant)\|>[\s\S]*?<\|end\|>)"
#     turns = re.findall(pattern, core_text)
#     return [turn.strip() + "\n" for turn in turns]

# def process_medium_sample(turns):
#     """
#     Cắt thông minh cho mẫu từ 1025 -> 5000 token xuống mức dưới 700 từ.
#     Giữ tối đa các lượt thoại liên tục từ đầu, loại bỏ các lượt vượt ngưỡng.
#     """
#     valid_turns = []
#     current_words = 0
    
#     for turn in turns:
#         turn_words = count_words(turn)
#         # Nếu cộng thêm lượt này mà vượt quá 700 từ thì dừng lại
#         if current_words + turn_words > LIMIT_WORDS:
#             break
#         valid_turns.append(turn)
#         current_words += turn_words
        
#     # Trường hợp câu thoại đầu tiên đã > 700 từ (Hiếm gặp), cố giữ lại câu đầu tiên
#     if not valid_turns and turns:
#         valid_turns.append(turns[0])
        
#     # Đóng gói lại chuỗi văn bản hoàn chỉnh
#     clean_text = "[BOS]" + "".join(valid_turns).strip() + "[EOS]"
#     return clean_text

# def process_huge_sample(turns):
#     """
#     Chia đều mẫu > 5000 token thành chính xác 5 phần độc lập.
#     Mỗi phần được bao bọc đầy đủ [BOS] và [EOS].
#     """
#     n = len(turns)
#     # Tính kích thước cơ bản của mỗi phần khi chia 5
#     size = n // 5
#     rem = n % 5
    
#     chunks = []
#     start = 0
#     for i in range(5):
#         # Chia đều phần dư cho các phần đầu tiên
#         end = start + size + (1 if i < rem else 0)
#         part_turns = turns[start:end]
#         start = end
        
#         if part_turns:
#             chunk_text = "[BOS]" + "".join(part_turns).strip() + "[EOS]"
#             chunks.append(chunk_text)
            
#     return chunks

# # 2. Đọc file dữ liệu gốc
# if not os.path.exists(CORPUS_FILE):
#     print(f"Lỗi: Không tìm thấy file '{CORPUS_FILE}'!")
#     exit()

# with open(CORPUS_FILE, "r", encoding="utf-8") as f:
#     corpus_data = json.load(f)

# # 3. Tiến hành xử lý
# cleaned_corpus = []
# stats = {"normal": 0, "truncated_700w": 0, "split_5_parts": 0}

# print("Đang tiến hành phân tích và tối ưu hóa dữ liệu thông minh...")

# for item in corpus_data:
#     text = item.get("text", "")
#     tokens = count_tokens(text)
    
#     # Trường hợp 1: Dữ liệu an toàn (<= 1024 token) -> Giữ nguyên
#     if tokens <= LIMIT_TOKENS:
#         cleaned_corpus.append({"text": text})
#         stats["normal"] += 1
        
#     # Trường hợp 2: Dữ liệu cực dài (> 5000 token) -> Chia làm 5 phần độc lập
#     elif tokens > CRITICAL_TOKENS:
#         turns = parse_turns(text)
#         split_parts = process_huge_sample(turns)
#         for part in split_parts:
#             cleaned_corpus.append({"text": part})
#         stats["split_5_parts"] += 1
        
#     # Trường hợp 3: Dữ liệu vượt mốc (1025 -> 5000 token) -> Cắt thông minh xuống 700 từ
#     else:
#         turns = parse_turns(text)
#         clean_text = process_medium_sample(turns)
#         cleaned_corpus.append({"text": clean_text})
#         stats["truncated_700w"] += 1

# # 4. Ghi lại dữ liệu mới sạch ra file json
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(cleaned_corpus, f, ensure_ascii=False, indent=4)

# # 5. Xuất báo cáo sau xử lý
# print("=" * 60)
# print(f" Hoàn tất xử lý! File sạch đã được ghi nhận tại: {OUTPUT_FILE}")
# print("=" * 60)
# print(f"📊 THỐNG KÊ KẾT QUẢ XỬ LÝ:")
# print(f"  - Số mẫu an toàn giữ nguyên: {stats['normal']} mẫu")
# print(f"  - Số mẫu trung bình được cắt thông minh về mức dưới 700 từ: {stats['truncated_700w']} mẫu")
# print(f"  - Số mẫu siêu dài (>5000t) được tách nhỏ thành 5 phần: {stats['split_5_parts']} mẫu")
# print(f"  - Tổng số lượng mẫu dữ liệu sau xử lý: {len(cleaned_corpus)} mẫu")
# print("=" * 60)


import json
import os
import re

CORPUS_FILE = "corpus.json"
OUTPUT_FILE = "corpus_clean.json"
TARGET_TOKENS = 1024

try:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    encoder = None

def count_tokens(text):
    if encoder:
        return len(encoder.encode(text))
    return int(len(text.split()) * 1.8)

def split_single_huge_turn(turn_text, max_tokens):
    """
    Nếu MỘT câu thoại duy nhất vẫn dài hơn giới hạn cho phép,
    hàm này sẽ bẻ nhỏ câu thoại đó theo các dấu câu (. ! ?) hoặc đoạn văn.
    """
    # Tách phần thẻ cấu trúc <|role|> ra khỏi nội dung
    header_match = re.match(r"(<\|(?:user|assistant)\|>)", turn_text)
    header = header_match.group(1) if header_match else "<|user|>"
    
    # Lấy nội dung thuần bên trong
    content = turn_text.replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "").strip()
    
    # Tách nội dung theo dấu câu (. ! ?) để không làm gãy từ
    sentences = re.split(r"(?<=[.!?])\s+", content)
    
    sub_turns = []
    current_sentences = []
    
    # Chi phí token tối thiểu cho cấu trúc: <|role|>\n + \n<|end|>
    base_cost = count_tokens(f"{header}\n\n<|end|>")
    accumulated = base_cost
    
    for sentence in sentences:
        sent_tokens = count_tokens(sentence)
        if accumulated + sent_tokens < max_tokens:
            current_sentences.append(sentence)
            accumulated += sent_tokens
        else:
            if current_sentences:
                # Đóng gói phần hiện tại thành 1 câu thoại hợp lệ
                clean_content = " ".join(current_sentences)
                sub_turns.append(f"{header}\n{clean_content}\n<|end|>\n")
            current_sentences = [sentence]
            accumulated = base_cost + sent_tokens
            
    if current_sentences:
        clean_content = " ".join(current_sentences)
        sub_turns.append(f"{header}\n{clean_content}\n<|end|>\n")
        
    return sub_turns

def parse_turns(text):
    """
    Tách chuỗi văn bản thành danh sách các khối thoại.
    Tự động sửa lỗi nếu dữ liệu không có dấu xuống dòng \n.
    """
    core_text = text.replace("[BOS]", "").replace("[EOS]", "").strip()
    # Tìm kiếm linh hoạt không phụ thuộc vào dấu xuống dòng
    pattern = r"(<\|(?:user|assistant)\|>(?:.*?)(?:<\|end\|>|$))"
    turns = re.findall(pattern, core_text, re.DOTALL)
    return [turn.strip() + "\n" for turn in turns]

def recursive_token_cutter(turns, max_tokens=TARGET_TOKENS):
    if not turns:
        return []
    
    current_turns = []
    remaining_turns = []
    
    base_token_cost = count_tokens("[BOS][EOS]") 
    accumulated_tokens = base_token_cost
    
    for i, turn in enumerate(turns):
        turn_tokens = count_tokens(turn)
        
        # Nếu chỉ riêng câu thoại này đã tự vượt max_tokens (Điểm cốt lõi gây lỗi ở code cũ)
        if turn_tokens >= (max_tokens - base_token_cost):
            # Nếu trước đó đã gom được vài câu thoại khác, dừng lại để xử lý câu thoại khổng lồ này ở lượt sau
            if current_turns:
                remaining_turns = turns[i:]
                break
            else:
                # Tiến hành bẻ nhỏ chính câu thoại khổng lồ này ra
                broken_sub_turns = split_single_huge_turn(turn, max_tokens - base_token_cost)
                current_turns.append(broken_sub_turns[0])
                remaining_turns = broken_sub_turns[1:] + turns[i+1:]
                break
        
        # Luồng gom câu thoại bình thường như cũ
        if accumulated_tokens + turn_tokens < max_tokens:
            current_turns.append(turn)
            accumulated_tokens += turn_tokens
        else:
            remaining_turns = turns[i:]
            break
            
    clean_text = "[BOS]" + "".join(current_turns).strip() + "[EOS]"
    results = [clean_text]
    
    if remaining_turns:
        results.extend(recursive_token_cutter(remaining_turns, max_tokens))
        
    return results

# 2. Thực thi xử lý file dữ liệu
if not os.path.exists(CORPUS_FILE):
    print(f"Lỗi: Không tìm thấy file '{CORPUS_FILE}'!")
    exit()

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

cleaned_corpus = []
total_truncated_samples = 0

print("Bắt đầu xử lý đệ quy cấp độ câu thoại (Bẻ cả câu đơn siêu dài)...")

for item in corpus_data:
    text = item.get("text", "")
    tokens = count_tokens(text)
    
    if tokens < TARGET_TOKENS:
        cleaned_corpus.append({"text": text})
    else:
        turns = parse_turns(text)
        split_samples = recursive_token_cutter(turns, TARGET_TOKENS)
        
        for sample in split_samples:
            # Kiểm tra lớp phòng thủ cuối cùng chống ghi đè lỗi trống
            if count_tokens(sample) > count_tokens("[BOS][EOS]"):
                cleaned_corpus.append({"text": sample})
        total_truncated_samples += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_corpus, f, ensure_ascii=False, indent=4)

print("=" * 60)
print(f" Đã ép thành công 100% mẫu về mức an toàn! Lưu tại: {OUTPUT_FILE}")
print(f" Tổng số lượng mẫu sau khi phân tách triệt để: {len(cleaned_corpus)} mẫu")
print("=" * 60)
