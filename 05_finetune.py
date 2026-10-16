import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import TinyGenerativeTransformer, compute_loss

# --- CẤU HÌNH ---
DATA_PATH = "datasets/finetune_data.json"
TOKENIZER_PATH = "tokenizer.json"
PRETRAINED_PATH = "best_finetuned.pt"
SAVE_PATH = "finetuned.pt"

LR = 8e-5
EPOCHS = 45  # Gom batch song song nên chỉ cần 10-15 Epoch là mô hình thuộc làu làu
BATCH_SIZE = 128 # Xử lý song song 128 câu cùng lúc để tận dụng tối đa sức mạnh RTX 4060
PATIENCE = 3

class SFTDataset(Dataset):
    def __init__(self, tokenizer, eos_id):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = []
        for item in data:
            inp_ids = tokenizer.encode(item["input"]).ids
            tgt_ids = tokenizer.encode(item["target"]).ids + [eos_id]
            self.samples.append((inp_ids, tgt_ids))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_id=0):
    """Gom các câu dài ngắn khác nhau thành một ma trận vuông vắn"""
    input_ids = [torch.tensor(item[0]) for item in batch]
    target_ids = [torch.tensor(item[1]) for item in batch]
    
    # Tự động chèn token [PAD] vào cuối các câu ngắn hơn
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
    
    return {"input_ids": input_ids_padded, "target_ids": target_ids_padded}

def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== KHỞI ĐỘNG HỆ THỐNG FINE-TUNE SIÊU TỐC ===")
    print(f"-> Thiết bị sử dụng: {device.upper()}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]") if tokenizer.token_to_id("[PAD]") is not None else 0

    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    model = TinyGenerativeTransformer(vocab_size=ckpt["vocab_size"], n_layers=ckpt["n_layers"], max_seq_len=1024).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    # print(f"✅ Đã load pretrained thành công (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4f})")

    # Khởi tạo dataset và dataloader song song
    full_dataset = SFTDataset(tokenizer, eos_id)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))
    
    print(f"-> Phân chia dữ liệu: Tập Train = {len(train_dataset):,} | Tập Val = {len(val_dataset):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val, no_improve = float("inf"), 0

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, inp, tgt, sep_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(inp)
        tr_loss /= len(train_dataset)

        # Đánh giá tập Val theo cơ chế batch song song
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                i = batch["input_ids"].to(device)
                t = batch["target_ids"].to(device)
                val_loss += compute_loss(model, i, t, sep_id).item() * len(i)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
        
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": ckpt["vocab_size"], 
                "n_layers": ckpt["n_layers"]
            }, SAVE_PATH)
            print("           -> [💾] Đã cập nhật checkpoint tốt nhất vĩnh viễn!")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"-> Kích hoạt Early Stopping để bảo vệ trọng số.")
                break

if __name__ == "__main__":
    finetune()
