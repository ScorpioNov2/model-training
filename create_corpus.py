# [BOS],[PAD],[EOS], thinking <thinking>,</thinking>,<sys_prompt></sys_prompt>, <|im_start|>, <|im_end|>, <|goodbye|>, <|start|>, <|end|>
# <instruction>Nghĩ ra một ý tưởng sáng tạo để cải thiện chủ đề.</instruction>
# [BOS], <|end|>, <|user|>, <|assistant|>, [PAD], [EOS]

import json
import os

# Giả lập dữ liệu đầu vào của bạn (Thay bằng đường dẫn file thật nếu cần)
# data_input = [
#     {
#         "instruction": " Đưa ra ba lời khuyên để duy trì sức khoẻ.",
#         "input": " ",
#         "output": " 1. Áp dụng chế độ ăn cân bằng...",
#         "": ""
#     },
#     {
#         "instruction": " Ba màu cơ bản là gì?",
#         "input": " ",
#         "output": " Ba màu cơ bản là đỏ...",
#         "": ""
#     }
# ]

# Nếu bạn đọc từ file json có sẵn, hãy bỏ comment 2 dòng dưới đây:
with open("datasets/dataset_02.json", "r", encoding="utf-8") as f:
    new_raw_data = json.load(f)

# 1. Đường dẫn file corpus hiện tại của bạn
corpus_file = "corpus.json"


# 2. Đọc mảng dữ liệu cũ từ file corpus
if os.path.exists(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        try:
            corpus_data = json.load(f)
        except json.JSONDecodeError:
            corpus_data = []
else:
    corpus_data = []


# 3. Xử lý dữ liệu mới và CHỈ giữ lại key "text"
new_formatted_records = []

for item in new_raw_data:
    full_text = "[BOS]"
    dialog = item.get("dialog", [])
    
    for turn in dialog:
        # Xóa khoảng trắng thừa ở đầu/cuối câu thoại
        turn = turn.strip()
        
        # Kiểm tra người nói, xóa tiền tố và gán token tương ứng
        if turn.startswith("A:"):
            # Lọc bỏ "A:" ở đầu câu và xóa khoảng trắng thừa còn lại
            clean_value = turn.replace("A:", "", 1).strip()
            full_text += f"<|user|>\n{clean_value}<|end|>\n"
            
        elif turn.startswith("B:"):
            # Lọc bỏ "B:" ở đầu câu và xóa khoảng trắng thừa còn lại
            clean_value = turn.replace("B:", "", 1).strip()
            full_text += f"<|assistant|>\n{clean_value}<|end|>\n"
            
    # Gộp token kết thúc chuỗi [EOS]
    full_text = full_text.strip() + "[EOS]"
    
    # CHỈ tạo duy nhất key "text" cho đồng nhất với file corpus cũ của bạn
    new_formatted_records.append({
        "text": full_text
    })

# 4. Tiến hành gộp mảng dữ liệu mới vào mảng cũ
corpus_data.extend(new_formatted_records)

# 5. Ghi đè lại file corpus.json với đầy đủ tiếng Việt có dấu
with open(corpus_file, "w", encoding="utf-8") as f:
    json.dump(corpus_data, f, ensure_ascii=False, indent=4)

print(f"Đã append thành công {len(new_formatted_records)} bản ghi mới vào '{corpus_file}'.")
print(f"Tổng số lượng bản ghi trong corpus hiện tại: {len(corpus_data)}")


# {
#     "text": "[BOS]<|user|>\nĐưa ra ba lời khuyên để duy trì sức khoẻ.<|end|>\n<|assistant|>\n1. Áp dụng chế độ ăn cân bằng và bổ dưỡng: Đảm bảo bữa ăn của bạn bao gồm nhiều loại rau quả, protein nạc, ngũ cốc nguyên hạt và chất béo lành mạnh. Điều này giúp cung cấp cho cơ thể các chất dinh dưỡng thiết yếu để hoạt động tốt nhất và có thể giúp ngăn ngừa các bệnh mãn tính. 2. Hoạt động thể chất thường xuyên: Tập thể dục rất quan trọng trong việc duy trì xương, cơ bắp khỏe mạnh và sức khỏe tim mạch. Hãy đặt mục tiêu tập ít nhất 150 phút bài tập aerobic vừa phải hoặc 75 phút bài tập cường độ cao mỗi tuần. 3. Ngủ đủ giấc: Ngủ đủ giấc có chất lượng rất quan trọng đối với sức khỏe thể chất và tinh thần. Nó giúp điều chỉnh tâm trạng, cải thiện chức năng nhận thức, hỗ trợ sự phát triển lành mạnh và chức năng miễn dịch. Hãy đặt mục tiêu ngủ 7-9 giờ mỗi đêm.<|end|>[EOS]"
# },
# {
#     "text": "[BOS]<|user|>\nBa màu cơ bản là gì?<|end|>\n<|assistant|>\nBa màu cơ bản là đỏ, xanh dương và vàng.Những màu này được gọi là màu cơ bản bởi vì chúng không thể được tạo ra bằng cách trộn các màu khác và tất cả các màu khác có thể được tạo ra bằng cách kết hợp chúng theo các tỷ lệ khác nhau.Trong hệ thống màu phụ gia, được sử dụng cho ánh sáng, các màu cơ bản là đỏ, xanh lá cây và xanh dương (RGB).<|end|>[EOS]"
# },
# {
#     "text": "[BOS]<|user|>\nTạo ra một câu hỏi trắc nghiệm về năng lượng tái tạo.<|end|>\n<|assistant|>\nNguồn năng lượng tái tạo chính được sử dụng trên toàn cầu là gì? A. Gió B. Mặt Trời C. Sinh khối D. Thủy điện<|end|>[EOS]"
# }