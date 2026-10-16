import os
import torch
from model import TinyGenerativeTransformer, pretrain_loss

# Nạp trực tiếp file dữ liệu nhị phân từ Bước 3 của bạn
DATASET_PATH = "dataset.pt"
BLOCK_SIZE = 1024
BATCH_SIZE = 32
TOTAL_STEPS = 10000        
LR = 3e-4
SAVE_PATH = "pretrained.pt"
RESUME_PATH = "resume_state.pt"   # checkpoint MOI NHAT - luu de tiep tuc khi bi ngat (Ctrl+C, mat dien...)
N_LAYERS = 8               
VOCAB_SIZE = 16000 # Khớp với cấu hình Vocab bộ từ điển của bạn


def get_batch(blocks, batch_size, device):
    """Lấy ngẫu nhiên một lô (Batch) dữ liệu từ ma trận Tensor tổng"""
    idx = torch.randint(0, blocks.size(0), (batch_size,))
    b = blocks[idx].to(device)
    return b[:, :-1], b[:, 1:]


@torch.no_grad()
def estimate_loss(model, train_blocks, val_blocks, batch_size, device, eval_iters=10):
    """Tinh loss trung binh qua nhieu batch thay vi 1 batch ngau nhien -> val_loss het bi nhieu/lac"""
    out = {}
    model.eval()
    for name, blocks in [("train", train_blocks), ("val", val_blocks)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(blocks, batch_size, device)
            losses[k] = pretrain_loss(model, x, y).item()
        out[name] = losses.mean().item()
    model.train()
    return out


def pretrain():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Thiết bị huấn luyện: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. NẠP FILE TENSOR ĐÃ ĐÓNG GÓI (Tốc độ load chưa tới 1 giây)
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file {DATASET_PATH}! Hãy chạy file Bước 3 trước.")
        
    print("-> Đang tải ma trận Tensor dữ liệu từ ổ cứng...")
    blocks = torch.load(DATASET_PATH, map_location='cpu')
    print(f"-> Nạp thành công: {blocks.size(0):,} cuộc hội thoại vuông vắn kích thước {blocks.shape}.")

    # 2. Phân tách dữ liệu: Giữ lại 2% cuộc hội thoại để làm tập Validation kiểm thử độc lập
    n_val = max(1, int(0.02 * blocks.size(0)))
    val_blocks, train_blocks = blocks[:n_val], blocks[n_val:]
    print(f"-> Chia tách: Tập Train = {train_blocks.size(0):,} | Tập Val = {val_blocks.size(0):,}")

    # 3. Khởi tạo mạng mạng Neural Generative Transformer của bạn
    model = TinyGenerativeTransformer(
        vocab_size=VOCAB_SIZE, 
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} tham số (~{n_params/1e6:.1f}M), {N_LAYERS} layer\n")

    # 4. Thiết lập bộ tối ưu hóa trọng số AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val = float("inf")
    start_step = 1

    # 5. KHÔI PHỤC nếu có checkpoint resume từ lần chạy trước bị ngắt giữa chừng
    if os.path.exists(RESUME_PATH):
        print(f"-> Tìm thấy {RESUME_PATH}, đang khôi phục tiến trình...")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"-> Tiếp tục từ step {start_step} (best_val đã có: {best_val:.4f})\n")

    def save_resume_state(step):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
        }, RESUME_PATH)

    print("=== BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (PRETRAIN) ===")
    try:
        for step in range(start_step, TOTAL_STEPS + 1):
            model.train()
            x, y = get_batch(train_blocks, BATCH_SIZE, device)
            
            optimizer.zero_grad(set_to_none=True)
            loss = pretrain_loss(model, x, y)
            loss.backward()
            
            # Cơ chế Gradient Clipping chống sập hoặc bùng nổ đạo hàm vô hạn
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Cứ sau mỗi 200 bước tính toán, tiến hành đánh giá chất lượng học trên tập Validation
            if step % 200 == 0:
                losses = estimate_loss(model, train_blocks, val_blocks, BATCH_SIZE, device)
                print(f"Step {step:5d}/{TOTAL_STEPS} | train_loss={losses['train']:.4f} | val_loss={losses['val']:.4f}")

                # Luôn lưu điểm tiếp tục (resume), dù val_loss có cải thiện hay không
                save_resume_state(step)

                # Nếu chỉ số loss trên tập Val giảm kỷ lục mới, tiến hành ghi đè checkpoint tốt nhất
                if losses['val'] < best_val:
                    best_val = losses['val']
                    torch.save({
                        "model_state_dict": model.state_dict(), 
                        "step": step,
                        "val_loss": best_val, 
                        "vocab_size": VOCAB_SIZE,
                        "n_layers": N_LAYERS
                    }, SAVE_PATH)
                    print(f"           -> lưu checkpoint TỐT NHẤT (val_loss={best_val:.4f})")

    except KeyboardInterrupt:
        print(f"\nBị ngắt (Ctrl+C) ở step {step}. Đang lưu để tiếp tục sau...")
        save_resume_state(step)
        print(f"Đã lưu {RESUME_PATH}. Chạy lại script này sẽ tự động tiếp tục từ step {step+1}.")
        return

    print(f"\n=== HOÀN THÀNH ===\nCheckpoint tốt nhất đạt mức val_loss={best_val:.4f} -> Đã lưu tại: {SAVE_PATH}")


if __name__ == "__main__":
    pretrain()


# PS C:\Users\Admin\Desktop\train_model> & D:\Python\python.exe c:/Users/Admin/Desktop/train_model/train_loop.py
# -> Thiết bị huấn luyện: CUDA
#    GPU: NVIDIA GeForce RTX 4060 Laptop GPU
# -> Đang tải ma trận Tensor dữ liệu từ ổ cứng...
# -> Nạp thành công: 138,555 cuộc hội thoại vuông vắn kích thước torch.Size([138555, 1025]).
# -> Chia tách: Tập Train = 135,784 | Tập Val = 2,771
# Model: 10,676,736 tham số (~10.7M), 8 layer

# === BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (PRETRAIN) ===
# Step   200/10000 | train_loss=3.4599 | val_loss=2.4548
#            -> lưu checkpoint TỐT NHẤT (val_loss=2.4548)
# Step   400/10000 | train_loss=1.8305 | val_loss=1.4925
#            -> lưu checkpoint TỐT NHẤT (val_loss=1.4925)
# Step   600/10000 | train_loss=1.5806 | val_loss=1.1537
#            -> lưu checkpoint TỐT NHẤT (val_loss=1.1537)
# Step   800/10000 | train_loss=1.2887 | val_loss=1.1568
# Step  1000/10000 | train_loss=1.2529 | val_loss=1.0123
#            -> lưu checkpoint TỐT NHẤT (val_loss=1.0123)
# Step  1200/10000 | train_loss=1.2997 | val_loss=0.9636
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.9636)
# Step  1400/10000 | train_loss=1.2064 | val_loss=0.9157
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.9157)
# Step  1600/10000 | train_loss=1.1621 | val_loss=0.8800
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.8800)
# Step  1800/10000 | train_loss=1.3279 | val_loss=0.8883
# Step  2000/10000 | train_loss=1.1651 | val_loss=0.7970
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.7970)
# Step  2200/10000 | train_loss=1.1919 | val_loss=0.8377
# Step  2400/10000 | train_loss=1.1526 | val_loss=0.8002
# Step  2600/10000 | train_loss=1.1190 | val_loss=0.9153
# Step  2800/10000 | train_loss=1.0757 | val_loss=0.7647
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.7647)
# Step  3000/10000 | train_loss=0.9939 | val_loss=0.7850
# Step  3200/10000 | train_loss=1.0592 | val_loss=0.7525
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.7525)
# Step  3400/10000 | train_loss=1.0325 | val_loss=0.8357
# Step  3600/10000 | train_loss=0.9881 | val_loss=0.8055
# Step  3800/10000 | train_loss=1.0574 | val_loss=0.7975
# Step  4000/10000 | train_loss=1.0185 | val_loss=0.7385
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.7385)
# Step  4200/10000 | train_loss=0.9817 | val_loss=0.7616
# Step  4400/10000 | train_loss=1.0071 | val_loss=0.7787
# Step  4600/10000 | train_loss=0.9468 | val_loss=0.7482
# Step  4800/10000 | train_loss=0.9864 | val_loss=0.6799
#            -> lưu checkpoint TỐT NHẤT (val_loss=0.6799)
# Step  5000/10000 | train_loss=0.9831 | val_loss=0.7793

# Bị ngắt (Ctrl+C) ở step 5013. Đang lưu để tiếp tục sau...
# Đã lưu resume_state.pt. Chạy lại script này sẽ tự động tiếp tục từ step 5014.