import pickle
import os

for fname in os.listdir("."):
    if not fname.endswith(".pkl"): continue
    print(f"--- File: {fname} ---")
    df = pickle.load(open(fname, "rb"))
    print("Columns:", df.columns)
    for idx, row in df.iterrows():
        data = row['input']
        # Nếu là list thì lấy phần tử đầu tiên
        if isinstance(data, (list, tuple)):
            data = data[0][1][0] if isinstance(data[0], tuple) else data[0]
        # Kiểm tra các thuộc tính
        if hasattr(data, 'edge_mask_gt'):
            print("edge_mask_gt found:", data.edge_mask_gt)
        if hasattr(data, 'node_mask_gt'):
            print("node_mask_gt found:", data.node_mask_gt)
        # Nếu không, in ra các thuộc tính sẵn có
        print("Available attributes:", dir(data))
        break   # Check first row only
    print("="*40)
