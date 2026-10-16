import os
import torch
from tokenizers import Tokenizer
from model import TinyGenerativeTransformer, generate

TOKENIZER_PATH = "tokenizer.json"
CHECKPOINT_PATH = "pretrained.pt" # Đổi thành "best_finetuned.pt" sau khi chạy SFT fine-tune
# CHECKPOINT_PATH = "finetuned.pt" # Đổi thành "best_finetuned.pt" sau khi chạy SFT fine-tune
BLOCK_SIZE = 1024
N_LAYERS = 8
VOCAB_SIZE = 16000


def main():
    print("=== HỆ THỐNG SINH VĂN BẢN (INFERENCE) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Thiết bị xử lý: {device.upper()}")

    # 1. Tải bộ từ điển Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Không tìm thấy file từ điển tại {TOKENIZER_PATH}")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 2. Khởi tạo cấu trúc mô hình nền tảng giống hệt lúc train
    model = TinyGenerativeTransformer(
        vocab_size=VOCAB_SIZE, n_layers=N_LAYERS, max_seq_len=BLOCK_SIZE
    ).to(device)

    # 3. Nạp trọng số từ checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Không tìm thấy file checkpoint {CHECKPOINT_PATH}. Vui lòng chạy Train trước!")
        return

    print(f"-> Đang tải trọng số từ file {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Nạp trọng số mô hình thành công. Mô hình sẵn sàng hoạt động!")

    model.eval()

    # Lấy ID của token kết thúc để mô hình biết đường tự dừng khi hết câu
    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n--- Gợi ý: Hãy nhập câu hỏi/hội thoại thực tế (Ví dụ: 'Tôi muốn tìm hiểu về bảo vệ môi trường') ---")

    while True:
        try:
            prompt = input("\nNhập câu mồi của bạn (hoặc gõ 'exit' để thoát): ")
            if prompt.strip().lower() == "exit":
                break
            if not prompt.strip():
                continue

            # 1. Tạo cấu trúc prompt chuẩn
            full_prompt = f"[BOS]<|user|>\n{prompt.strip()}<|end|>\n<|assistant|>\n"

            # 2. Mã hóa chuỗi
            encoded = tokenizer.encode(full_prompt, add_special_tokens=True)
            input_ids = encoded.ids if hasattr(encoded, 'ids') else encoded
            prompt_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            print(prompt_ids)

            # 3. Tạo độ dài của prompt đầu vào để làm mốc cắt token
            # (Chỉ lấy phần token được mô hình sinh mới hoàn toàn)
            input_length = prompt_ids.shape[1]

            # 4. Sinh văn bản
            output_ids = generate(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=150,  
                temperature=0.2,     
                top_k=5,            
                top_p=0.5,          
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_id=eos_id,
            )

            # output_ids[0] là danh sách token, ta lấy từ vị trí input_length trở đi
            new_tokens = output_ids[0]
            print(new_tokens)
            new_tokens = output_ids[0][input_length:].tolist()
            bot_response = tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            # Dọn dẹp nốt các thẻ rác hệ thống nếu mô hình tự sinh ra ở cuối câu
            for tag in ["[BOS]", "[EOS]", "<|user|>", "<|assistant|>", "<|end|>"]:
                bot_response = bot_response.replace(tag, "")
            bot_response = bot_response.strip()

            print(f"👉 Kết quả mô hình viết tiếp:\n{bot_response}")

        except KeyboardInterrupt:
            break

    print("\n=== ĐÃ THOÁT HỆ THỐNG ===")

if __name__ == "__main__":
    main()
