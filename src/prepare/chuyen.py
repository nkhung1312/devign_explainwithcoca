import pandas as pd
import torch
from torch_geometric.data import Data
import pickle

FILENAME = "sample_trace_dev.csv"
OUTFILE = "sample_trace_dev.pkl"

df = pd.read_csv(FILENAME)
df.columns = df.columns.str.replace("'", "").str.strip()
print("Tên cột:", [repr(col) for col in df.columns])
print(df.head(1))

# Ưu tiên chọn code_col theo thứ tự: code → bug_function → function → functions
priority = ['code', 'bug_function', 'function', 'functions']
code_col = None
label_col = None

for name in priority:
    for c in df.columns:
        if name == c.lower():
            code_col = c
            break
    if code_col is not None:
        break

for c in df.columns:
    if 'label' in c.lower():
        label_col = c
        break

if code_col is None or label_col is None:
    raise Exception("Không tìm thấy cột code/label!")

print(f"code_col: {code_col}, label_col: {label_col}")
print("Ví dụ giá trị code_col:", code_col, "→", df.iloc[0][code_col][:100])

data_list = []
for idx, row in df.iterrows():
    code = row[code_col]
    label = row[label_col]

    # Mock CPG, embedding, mask (giữ nguyên như trên)
    lines = code.split('\n')
    nodes = {i: type('Node', (), {'line': i+1})() for i in range(len(lines)) if lines[i].strip() != ''}
    edge_index = torch.combinations(torch.arange(len(nodes)), r=2).t() if len(nodes) > 1 else torch.zeros((2, 0), dtype=torch.long)
    x = torch.ones(len(nodes), 10)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )
    data_list.append(data)

with open(OUTFILE, "wb") as f:
    pickle.dump(data_list, f)

print(f"Đã chuẩn hóa xong, lưu ra {OUTFILE}")
