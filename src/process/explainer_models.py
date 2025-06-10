import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

class GraphExplainerEdge(nn.Module):
    def __init__(self, base_model, G_dataset, test_indices, args, fix_exp=None):
        super(GraphExplainerEdge, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.G_dataset = G_dataset
        self.test_indices = test_indices
        self.args = args
        self.device = args.device if hasattr(args, "device") else torch.device('cpu')
        self.fix_exp = fix_exp

    def explain_nodes_gnn_stats(self):
        exp_dict = {}  # {gid: edge_mask}
        num_dict = {}  # {gid: num_exp_edges}
        PN_total, PS_total, FNS_total = 0, 0, 0
        accs, pres, recs, f1s = [], [], [], []
        expl_count = 0
        for gid in tqdm.tqdm(self.test_indices):
            data = self.G_dataset[gid].to(self.device)
            ori_pred = self.base_model(data)
            pred_label = torch.round(ori_pred).long().item()
            ori_label = data.y.item() if hasattr(data.y, 'item') else int(data.y)
            # Debug nhãn và dự đoán
            print(f"[GID {gid}] ori_label={ori_label}, pred_label={pred_label}, ori_pred={ori_pred.item()}")
            # Chỉ giải thích khi predict đúng (tuỳ bạn muốn filter gì)
            if pred_label == 1 and ori_label == 1:
                edge_mask, exp_num = self.explain(data, ori_pred)
                exp_dict[gid] = edge_mask.detach().cpu()
                num_dict[gid] = exp_num
                # --- Đánh giá PN/PS ---
                pn = self._compute_pn(data, edge_mask, ori_label)
                ps = self._compute_ps(data, edge_mask, ori_label)
                print(f"[GID {gid}] PN: {pn}, PS: {ps}, Num_exp_edges: {exp_num}")
                PN_total += pn
                PS_total += ps
                if pn + ps > 0:
                    FNS_total += 2 * pn * ps / (pn + ps)
                expl_count += 1
        ave_exp = sum(num_dict.values()) / expl_count if expl_count else 0
        PN = PN_total / expl_count if expl_count else 0
        PS = PS_total / expl_count if expl_count else 0
        FNS = FNS_total / expl_count if expl_count else 0
        # Nếu chưa có groundtruth edge thì acc/pre/rec/f1 để 0
        return PN, PS, FNS, ave_exp, 0, 0, 0, 0

    def explain(self, data, ori_pred):
        # Khởi tạo mask cho từng edge
        edge_mask = nn.Parameter(torch.randn(data.edge_index.size(1), device=self.device))
        optimizer = optim.Adam([edge_mask], lr=self.args.lr, weight_decay=0)
        mask_thresh = getattr(self.args, 'mask_thresh', 0.5)
        num_epochs = getattr(self.args, 'num_epochs', 100)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            mask_sigmoid = torch.sigmoid(edge_mask)
            masked_edge_index = self._mask_edge_index(data.edge_index, mask_sigmoid, threshold=mask_thresh)
            data_masked = self._clone_data_with_new_edge(data, masked_edge_index)
            pred1 = self.base_model(data_masked)
            # Counterfactual: loại bỏ các edge explainer giữ lại
            masked_edge_index_cf = self._mask_edge_index(data.edge_index, mask_sigmoid, threshold=mask_thresh, invert=True)
            data_cf = self._clone_data_with_new_edge(data, masked_edge_index_cf)
            pred2 = self.base_model(data_cf)
            # Loss function như CoCa
            relu = nn.ReLU()
            gam = getattr(self.args, 'gam', 0.1)
            lam = getattr(self.args, 'lam', 1.0)
            alp = getattr(self.args, 'alp', 0.5)
            bpr1 = relu(gam + 0.5 - pred1)
            bpr2 = relu(gam + pred2 - 0.5)
            l1 = torch.norm(mask_sigmoid, p=1)
            loss = l1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
            # Debug loss/mask
            if epoch % 20 == 0 or epoch == num_epochs-1:
                print(f"[Epoch {epoch}] Loss={loss.item():.4f}, bpr1={bpr1.item():.4f}, bpr2={bpr2.item():.4f}, l1={l1.item():.4f}")
            loss.backward()
            optimizer.step()
        # Kết thúc training mask, chọn các edge có sigmoid(mask) > threshold
        final_mask = torch.sigmoid(edge_mask)
        exp_num = (final_mask > mask_thresh).sum().item()
        return final_mask.detach(), exp_num

    def _mask_edge_index(self, edge_index, mask_sigmoid, threshold=0.5, invert=False):
        if invert:
            keep_idx = (mask_sigmoid <= threshold).nonzero(as_tuple=True)[0]
        else:
            keep_idx = (mask_sigmoid > threshold).nonzero(as_tuple=True)[0]
        if keep_idx.numel() == 0:
            keep_idx = torch.arange(0, min(2, mask_sigmoid.size(0)), device=mask_sigmoid.device)
        return edge_index[:, keep_idx]

    def _clone_data_with_new_edge(self, data, new_edge_index):
        data_new = data.clone()
        data_new.edge_index = new_edge_index
        return data_new

    def _compute_pn(self, data, edge_mask, ori_label):
        mask_sigmoid = torch.sigmoid(edge_mask)
        edge_index_pn = self._mask_edge_index(data.edge_index, mask_sigmoid, threshold=self.args.mask_thresh, invert=True)
        data_pn = self._clone_data_with_new_edge(data, edge_index_pn)
        pred = self.base_model(data_pn)
        pred_label = torch.round(pred).long().item()
        print(f"  [PN-check] ori_label={ori_label}, pred_label={pred_label}")
        return 1 if pred_label != ori_label else 0

    def _compute_ps(self, data, edge_mask, ori_label):
        mask_sigmoid = torch.sigmoid(edge_mask)
        edge_index_ps = self._mask_edge_index(data.edge_index, mask_sigmoid, threshold=self.args.mask_thresh, invert=False)
        data_ps = self._clone_data_with_new_edge(data, edge_index_ps)
        pred = self.base_model(data_ps)
        pred_label = torch.round(pred).long().item()
        print(f"  [PS-check] ori_label={ori_label}, pred_label={pred_label}")
        return 1 if pred_label == ori_label else 0
