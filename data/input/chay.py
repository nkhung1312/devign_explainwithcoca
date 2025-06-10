import pickle
import pandas as pd
import torch

# Đường dẫn đến file pkl
pkl_file = '0_cpg_input.pkl'
df = pickle.load(open(pkl_file, 'rb'))

for idx, row in df.iterrows():
    data = row['input']
    code = getattr(data, 'code', 'NO_CODE_IN_OBJECT')  # hoặc row['code'] nếu DataFrame có cột này
    out = []
    out.append(f'--- Graph index: {idx} ---')
    out.append('### Source code:')
    out.append(str(code))
    out.append('')

    # Node list
    out.append('### Nodes:')
    x = data.x
    for i, feat in enumerate(x):
        if hasattr(feat, 'tolist'):
            node_content = feat.tolist()
        else:
            node_content = feat
        out.append(f'Node {i}: {node_content}')
    out.append('')

    # Edge list
    edge_index = data.edge_index
    for j in range(edge_index.shape[1]):
        from_node, to_node = edge_index[0, j].item(), edge_index[1, j].item()
        out.append(f'Edge {j}: {from_node} -> {to_node}')
    out.append('\n---\n')

    # Lưu ra file cho mỗi function hoặc append vào một file tổng
    fname = f'annotate_graph_{idx}.txt'
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
