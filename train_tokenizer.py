import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Định nghĩa hàm đọc dữ liệu từ file JSON dạng danh sách text
def batch_iterator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            yield item["text"]

def main():
    # 2. Khởi tạo một mô hình BPE trống
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 3. Sử dụng bộ tiền phân tách từ (tách theo khoảng trắng và dấu câu cơ bản)
    tokenizer.pre_tokenizer = Whitespace()

    # 4. Danh sách các token đặc biệt theo cấu trúc dữ liệu của bạn
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<|user|>", "<|assistant|>", "<|end|>"]

    # 5. Cấu hình bộ huấn luyện (Trainer)
    trainer = BpeTrainer(
        vocab_size=16000,
        special_tokens=special_tokens,
        show_progress=True
    )

    # 6. Đọc dữ liệu từ file 'corpus_clean.json' và tiến hành huấn luyện
    input_file = "corpus_clean.json"
    print(f"Đang huấn luyện tokenizer từ file {input_file}...")
    tokenizer.train_from_iterator(batch_iterator(input_file), trainer=trainer)

    # 7. Lưu kết quả ra file tokenizer.json
    output_file = "tokenizer.json"
    tokenizer.save(output_file)
    print(f"Đã tạo và lưu thành công cấu hình tokenizer vào file: {output_file}")

if __name__ == "__main__":
    main()
