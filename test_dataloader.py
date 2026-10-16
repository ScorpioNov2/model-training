import torch
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer

def test_batch_loading():
    # 1. Load lại Tensor dữ liệu từ ổ cứng
    print("Đang tải dữ liệu từ ổ cứng...")
    tensor_data = torch.load("dataset.pt", weights_only=True)
    
    # Load tokenizer để phục vụ việc giải mã xem thử
    tokenizer = Tokenizer.from_file("tokenizer.json")
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    # 2. Tạo PyTorch Dataset và DataLoader
    dataset = TensorDataset(tensor_data)
    batch_size = 1 # Lấy thử batch_size nhỏ để dễ quan sát
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Bốc thử 1 batch đầu tiên để kiểm tra
    print("\n--- TIẾN HÀNH KIỂM TRA BỐC 1 BATCH ---")
    for batch in dataloader:
        # batch[0] có kích thước: [batch_size, seq_len]
        sequences = batch[0] 
        
        # Tạo dữ liệu huấn luyện (X) và Nhãn (Y) bằng cách dịch vị trí (Shift)
        # X: bỏ token cuối cùng
        input_ids = sequences[:, :-1]
        # Y: bỏ token đầu tiên (mô hình sẽ đoán token tiếp theo)
        labels = sequences[:, 1:].clone()
        
        # Mẹo quan trọng: Thay thế các token [PAD] trong nhãn thành -100 
        # để PyTorch Loss Function tự động bỏ qua không tính loss cho các token padding này
        labels[labels == pad_token_id] = -100
        
        # In thông tin kiểm tra kích thước
        print(f"Kích thước Input IDs (X): {input_ids.shape}")
        print(f"Kích thước Labels (Y):    {labels.shape}")
        
        # Thử in và giải mã (Decode) mẫu đầu tiên trong Batch để đối chiếu
        print("\n--- Chi tiết mẫu số 1 trong batch ---")
        first_input = input_ids[0].tolist()
        first_label = labels[0].tolist()
        
        print(f"Mảng ID đầu vào (X): {first_input[:15]}...") # In 15 id đầu
        print(f"Văn bản giải mã (X): {tokenizer.decode(first_input, skip_special_tokens=False)}")
        
        print(f"\nMảng ID nhãn (Y)   : {first_label[:15]}...")
        # Lọc bỏ -100 để decode xem nhãn có khớp không
        clean_label = [idx for idx in first_label if idx != -100]
        print(f"Văn bản giải mã (Y): {tokenizer.decode(clean_label, skip_special_tokens=False)}")
        
        # Chỉ chạy thử 1 batch rồi dừng
        break

if __name__ == "__main__":
    test_batch_loading()
