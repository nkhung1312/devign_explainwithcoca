import json
import torch
import pickle
import glob
import numpy as np
import pandas as pd

# --- Đọc file mẫu .pkl để lấy format
with open("0_cpg_input.pkl", "rb") as f:
    sample_list = pickle.load(f)

if isinstance(sample_list, pd.DataFrame):
    print("File mẫu là DataFrame, chuyển sang list...")
    sample_list = sample_list.to_dict('records')

sample = sample_list[0]
print("Các trường/keys mẫu:", sample.keys())

# --- Đọc tất cả file json cần chuẩn hóa
input_files = sorted(glob.glob("auto_labeler_*.json")) + sorted(glob.glob("after_fix_*.json"))
print("Tìm thấy file:", input_files)

data_list = []

for fname in input_files:
    with open(fname, encoding='utf-8') as f:
        obj = json.load(f)
    
    bug_info = obj.get("bug_info") or {}
    main_func_name = bug_info.get("function")
    functions = obj.get("functions", {})
    if not functions:
        print(f"Bỏ qua {fname} vì không có function code.")
        continue
    if main_func_name and main_func_name in functions:
        func_code = functions[main_func_name].get("code", "")
    else:
        func_code = next(iter(functions.values())).get("code", "")
    if not func_code:
        print(f"Bỏ qua {fname} vì không có function code.")
        continue

    bug_line = None
    try:
        bug_info = obj.get("bug_info") or {}
        bug_line = int(bug_info.get("line", None))
    except Exception:
        bug_line = None

    lines = func_code.split('\n')
    nodes = [line for line in lines if line.strip() != ""]
    num_nodes = len(nodes)
    if num_nodes == 0:
        print(f"Bỏ qua {fname} vì function code rỗng.")
        continue

    node_mask = [1 if bug_line and (i+1) == bug_line else 0 for i in range(num_nodes)]

    # === SỬA ĐÚNG FORMAT THEO FILE MẪU ===
    feat = sample['input']
    if hasattr(feat, 'shape') and len(feat.shape) > 1:
        feat_dim = feat.shape[1]
    elif hasattr(feat, 'shape') and len(feat.shape) == 1:
        feat_dim = 1
    elif isinstance(feat, (list, tuple)) and len(feat) > 0 and hasattr(feat[0], '__len__'):
        feat_dim = len(feat[0])
    else:
        feat_dim = 10
    feat_dtype = getattr(feat, 'dtype', np.float32)
    x = np.ones((num_nodes, feat_dim), dtype=feat_dtype)
    edge_index = (
        torch.combinations(torch.arange(num_nodes), r=2).t().numpy()
        if num_nodes > 1 else np.zeros((2, 0), dtype=np.int64)
    )
    label = obj.get('label', 0)

    d = {
        'input': x,
        'edge_index': edge_index,
        'target': label,
        'node_mask': np.array(node_mask, dtype=np.int64)
    }
    for extra_key in sample.keys():
        if extra_key not in d:
            d[extra_key] = None
    data_list.append(d)

# ---- Lưu ra file .pkl ----
with open('cpg_input_with_mask.pkl', 'wb') as f:
    pickle.dump(data_list, f)

print(f"Đã chuẩn hóa xong {len(data_list)} samples, lưu ra cpg_input_with_mask.pkl")
