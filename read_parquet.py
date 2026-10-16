import pandas as pd

# Đọc file parquet
df = pd.read_parquet("datasets/dataset_02.parquet")

# Hiển thị 5 dòng đầu tiên
print(df.head())

df.to_json("dataset_02.json", orient="records", force_ascii=False, indent=4)