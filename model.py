import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGenerativeTransformer(nn.Module):
    def __init__(self, vocab_size=16000, d_model=256, n_heads=4, n_layers=8,
                 d_ff=1024, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        # h = self.blocks(h, mask=causal_mask)
        h = self.blocks(h, mask=causal_mask, is_causal=True)
        return self.head(self.ln_f(h))

def pretrain_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

def compute_loss(model, input_ids, target_ids, sep_id, pad_id=-100):
    # Lấy ra Batch Size thực tế từ dữ liệu đầu vào (ví dụ: 64 hoặc 128)
    batch_size = input_ids.shape[0]
    
    # SỬA LỖI TẠI ĐÂY: Tạo token [SEP] và tự động co giãn chiều Batch khớp với dữ liệu nạp vào
    # .expand(batch_size, 1) sẽ đổi shape từ [1, 1] thành [batch_size, 1] cực kỳ nhanh trên GPU
    sep_tensor = torch.tensor([[sep_id]], device=input_ids.device).expand(batch_size, 1)
    
    # Giờ đây tất cả các ma trận đều có chung Batch Size, lệnh cat chạy mượt mà 100%
    full = torch.cat([input_ids, sep_tensor, target_ids], dim=1)

    x, y = full[:, :-1], full[:, 1:].clone()
    input_len = input_ids.shape[1] + 1
    y[:, :input_len - 1] = pad_id
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cum_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        indices_to_remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        logits[indices_to_remove] = filter_value
    return logits


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for token_id in set(generated_ids[0].tolist()):
        val = logits[0, token_id]
        logits[0, token_id] = val / penalty if val > 0 else val * penalty
    return logits


def block_repeated_ngrams(logits, generated_ids, n=3):
    seq = generated_ids[0].tolist()
    if len(seq) < n - 1:
        return logits
    seen = {}
    for i in range(len(seq) - n + 1):
        seen.setdefault(tuple(seq[i:i + n - 1]), set()).add(seq[i + n - 1])
    for banned in seen.get(tuple(seq[-(n - 1):]), set()):
        logits[0, banned] = -float("inf")
    return logits


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=150, temperature=0.8,
             top_k=50, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3,
             eos_id=None, min_confidence=None):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].clone()
        logits = apply_repetition_penalty(logits, ids, repetition_penalty)
        logits = logits / temperature

        if min_confidence is not None:
            raw_probs = F.softmax(logits, dim=-1)
            if raw_probs.max().item() < min_confidence:
                break   # model khong chac ve token tiep theo -> dung thay vi "bia"

        logits = block_repeated_ngrams(logits, ids, no_repeat_ngram_size)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break   # model tu "biet" ket thuc, khong can chay het max_new_tokens
    return ids
