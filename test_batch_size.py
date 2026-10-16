import time
import torch
from model import TinyGenerativeTransformer, pretrain_loss

DATASET_PATH = "dataset.pt"
BLOCK_SIZE = 1024
VOCAB_SIZE = 16000
N_LAYERS = 8
BATCH_SIZES_TO_TEST = [32, 64, 128, 256]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Thiet bi: {device}")

blocks = torch.load(DATASET_PATH, map_location="cpu")
model = TinyGenerativeTransformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
                                   max_seq_len=BLOCK_SIZE).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def get_batch(batch_size):
    idx = torch.randint(0, blocks.size(0), (batch_size,))
    b = blocks[idx].to(device)
    return b[:, :-1], b[:, 1:]


def bench(batch_size, n_steps=20):
    # warm-up: loai bo thoi gian khoi tao CUDA context / cache lan dau, khong tinh vao ket qua
    for _ in range(3):
        x, y = get_batch(batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = pretrain_loss(model, x, y)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(n_steps):
        x, y = get_batch(batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = pretrain_loss(model, x, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()  # doi GPU chay xong that su, khong do nham luc con "thu viec"
    elapsed = time.time() - t0

    step_time = elapsed / n_steps
    tokens_per_sec = batch_size * BLOCK_SIZE / step_time
    line = f"batch={batch_size:4d} | {step_time*1000:7.1f} ms/step | {tokens_per_sec:10,.0f} token/s"
    if device == "cuda":
        line += f" | VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


for bs in BATCH_SIZES_TO_TEST:
    try:
        bench(bs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"batch={bs:4d} -> HET VRAM, dung lai o day. Chon batch nho hon muc nay.")
            if device == "cuda":
                torch.cuda.empty_cache()
            break
        raise


# PS C:\Users\Admin\Desktop\train_model> & D:\Python\python.exe c:/Users/Admin/Desktop/train_model/test_batch_size.py
# Thiet bi: cuda
# batch=  32 |  2340.1 ms/step |     14,003 token/s | VRAM: 11.49 GB
# batch=  64 -> HET VRAM, dung lai o day. Chon batch nho hon muc nay.