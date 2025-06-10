
import torch
import os
import pickle
import pandas as pd
from src.process.model import Net
from src.process.explainer_models import GraphExplainerEdge

import torch
from torch_geometric.data import Data

def dataframe_to_data_objects(df):
    data_list = []
    for idx, row in df.iterrows():
        input_tuple = row['input']
        label = row['target']
        # TH1: Nếu input_tuple là list có phần tử, lấy phần tử đầu
        if isinstance(input_tuple, (list, tuple)):
            tup = input_tuple[0]
            x_name, tensor_list = tup
            x_tensor = torch.stack(tensor_list)
        # TH2: Nếu input_tuple là Data (đã đúng kiểu PyG Data)
        elif isinstance(input_tuple, Data):
            data_list.append(input_tuple)
            continue
        else:
            print(f"Lỗi không xác định input_tuple (index {idx}): {type(input_tuple)}")
            continue

        # Dummy edge_index (bạn cần thay bằng edge_index thật nếu có)
        num_nodes = x_tensor.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        data_obj = Data(x=x_tensor, edge_index=edge_index, y=torch.tensor([label]))
        data_list.append(data_obj)
    return data_list
class Args:
    def __init__(self):
        self.lr = 0.01
        self.num_epochs = 200
        self.mask_thresh = 0.5
        self.gam = 0.1
        self.lam = 1.0
        self.alp = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gated_graph_conv_args = {
            "out_channels": 200,
            "num_layers": 6
        }
        self.conv_args = {
            "conv1d_1": {
                "in_channels": 205,
                "out_channels": 50,
                "kernel_size": 3,
                "padding": 1
            },
            "conv1d_2": {
                "in_channels": 50,
                "out_channels": 20,
                "kernel_size": 1,
                "padding": 1
            },
            "maxpool1d_1": {
                "kernel_size": 3,
                "stride": 2
            },
            "maxpool1d_2": {
                "kernel_size": 2,
                "stride": 2
            }
        }
        self.emb_size = 101

def load_model(args, checkpoint_path):
    model = Net(
        args.gated_graph_conv_args,
        args.conv_args,
        args.emb_size,
        args.device
    )
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    return model

def load_testset_from_dataframe(pkl_path):
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    data_list = dataframe_to_data_objects(df)
    return data_list

if __name__ == '__main__':
    args = Args()
    checkpoint_path = 'data/model/checkpoint.pt'
    model = load_model(args, checkpoint_path)
    
    # GỘP nhiều file testset nếu muốn
    testset = []
    for fname in os.listdir("data/input"):
        if fname.endswith(".pkl"):
            print("Loading:", fname)
            testset.extend(load_testset_from_dataframe(os.path.join("data/input", fname)))
    test_indices = list(range(len(testset)))

    explainer = GraphExplainerEdge(
        base_model=model,
        G_dataset=testset,
        test_indices=test_indices,
        args=args
    )

    PN, PS, FNS, ave_exp, acc, pre, rec, f1 = explainer.explain_nodes_gnn_stats()
    print(f'=== EXPLAINER EVALUATION RESULTS ===')
    print(f'PN: {PN:.3f}, PS: {PS:.3f}, FNS: {FNS:.3f}, Ave#Exp: {ave_exp:.2f}')
    print(f'ACC: {acc:.3f}, PRE: {pre:.3f}, REC: {rec:.3f}, F1: {f1:.3f}')
