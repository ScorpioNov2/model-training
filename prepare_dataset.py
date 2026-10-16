import json
import torch
from tokenizers import Tokenizer

def preprocess_and_save():
    # 1. Load tokenizer đã tạo từ bước trước
    tokenizer = Tokenizer.from_file("tokenizer.json")
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    # =========================================================================
    # CẤU HÌNH CẮT DỮ LIỆU TẠI ĐÂY:
    # Thiết lập độ dài tối đa (Context Length) ví dụ là 1024, 2048 hoặc 4096 tùy cấu hình mô hình
    CONTEXT_LENGTH = 1024 + 1 
    
    # Kích hoạt tính năng tự động cắt khi chuỗi vượt quá CONTEXT_LENGTH
    tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)
    # =========================================================================
    
    # 2. Đọc dữ liệu từ corpus_clean.json
    with open("corpus_clean.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokenized_ids = []
    
    # 3. Tokenize từng mẫu dữ liệu (lúc này các mẫu dài hơn CONTEXT_LENGTH sẽ tự động bị cắt)
    print("Đang mã hóa dữ liệu...")
    for item in data:
        text = item["text"]
        encoding = tokenizer.encode(text)
        tokenized_ids.append(encoding.ids)
    
    # 4. Padding cho các chuỗi ngắn hơn (tối đa bằng CONTEXT_LENGTH hoặc bằng mẫu dài nhất hiện tại)
    max_len = max(len(ids) for ids in tokenized_ids)
    print(f"Độ dài chuỗi lớn nhất sau khi áp dụng cắt: {max_len}")
    
    padded_ids = []
    for ids in tokenized_ids:
        padding_length = max_len - len(ids)
        padded_sequence = ids + [pad_token_id] * padding_length
        padded_ids.append(padded_sequence)
        
    # 5. Chuyển thành PyTorch Tensor và lưu
    tensor_data = torch.tensor(padded_ids, dtype=torch.long)
    print(f"Kích thước tensor dữ liệu: {tensor_data.shape}")
    
    torch.save(tensor_data, "dataset.pt")
    print("Đã lưu tensor dữ liệu thành công.")

if __name__ == "__main__":
    preprocess_and_save()
