import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===================== Dataset Processing =====================
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, max_seq_len, pad_item=0):
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len
        self.pad_item = pad_item

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tgt = self.targets[idx]
        # pad or truncate
        if len(seq) < self.max_seq_len:
            padded = [self.pad_item] * (self.max_seq_len - len(seq)) + seq
        else:
            padded = seq[-self.max_seq_len:]
        return torch.LongTensor(padded), torch.LongTensor([tgt])

# ===================== Model Definition =====================
class IBSelectorModel(nn.Module):
    def __init__(self, num_items, embed_dim=128, max_seq_len=50,
                 nhead=2, num_layers=2, hidden_dim=256,
                 k=5, beta=0.1, tau=0.5):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.k = k
        self.beta = beta
        self.tau = tau

        self.item_embed = nn.Embedding(num_items+1, embed_dim, padding_idx=0)
        self.pos_embed  = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.selector = nn.Linear(embed_dim, 1)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_items+1)
        )

    def forward(self, input_ids, target_ids=None):
        B, T = input_ids.size()
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_embed(input_ids) + self.pos_embed(pos)
        H = self.transformer(x.permute(1,0,2)).permute(1,0,2)  # (B,T,emb)
        logits = self.selector(H).squeeze(-1)
        # Gumbel-TopK selection (Straight-Through)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
        noisy = (logits + gumbel) / self.tau
        soft = F.softmax(noisy, dim=-1)
        topk = torch.topk(soft, self.k, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(1, topk, 1.0)
        mask = (hard - soft).detach() + soft
        # IB loss
        p = soft
        kl = (p * (torch.log(p+1e-8) + np.log(T))).sum(dim=-1).mean()
        # aggregate selected embeddings
        H_sel = (H * mask.unsqueeze(-1)).sum(dim=1) / self.k
        logits_pred = self.predictor(H_sel)
        out = {'logits': logits_pred, 'kl': kl}
        if target_ids is not None:
            ce = F.cross_entropy(logits_pred, target_ids.squeeze())
            out['loss'] = ce + self.beta * kl
        return out

# ===================== Utility Metrics =====================
def hit_rate_at_k(preds, target, k):
    topk = preds.topk(k, dim=-1).indices
    return (topk == target.unsqueeze(1)).any(dim=1).float().mean().item()

def ndcg_at_k(preds, target, k):
    topk = preds.topk(k, dim=-1).indices
    target = target.unsqueeze(1)
    hits = (topk == target).float()
    if hits.sum() == 0:
        return 0.0
    ranks = torch.arange(1, k+1, device=preds.device)
    dcg = (hits / torch.log2(ranks+1)).sum().item()
    return dcg

# ===================== Main Training & Evaluation =====================
if __name__ == '__main__':
    # 1. Load data
    df = pd.read_csv('data/moviestv/processed_interactions.csv')  # columns: user_id,item_id,rating,timestamp
    # ignore rating
    # map item ids to continuous indices
    item2idx = {item: i+1 for i, item in enumerate(df['item_id'].unique())}
    df['item_idx'] = df['item_id'].map(item2idx)
    num_items = len(item2idx)

    # 2. Prepare train/val/test via leave-one-out + validation
    df = df.sort_values(['user_id', 'timestamp'])
    train_seqs, train_tgts = [], []
    val_seqs, val_tgts = [], []
    test_seqs, test_tgts = [], []
    for _, group in df.groupby('user_id'):
        items = group['item_idx'].tolist()
        if len(items) < 3:
            continue  # need at least one train, one val, one test
        # test: last
        test_seqs.append(items[:-1])
        test_tgts.append(items[-1])
        # val: second last
        val_seqs.append(items[:-2])
        val_tgts.append(items[-2])
        # train: sliding window on items[0:-2]
        for i in range(1, len(items)-2):
            train_seqs.append(items[:i])
            train_tgts.append(items[i])

    # 3. DataLoader
    max_seq_len = 50
    train_ds = SequenceDataset(train_seqs, train_tgts, max_seq_len)
    val_ds   = SequenceDataset(val_seqs, val_tgts, max_seq_len)
    test_ds  = SequenceDataset(test_seqs, test_tgts, max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512)
    test_loader  = DataLoader(test_ds, batch_size=512)

    # 4. Model & Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IBSelectorModel(num_items=num_items, max_seq_len=max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    epochs = 20
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for seq_batch, tgt_batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            out = model(seq_batch, tgt_batch)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq_batch.size(0)
        print(f'Epoch {epoch} Train Loss: {total_loss/len(train_ds):.4f}')

        # 6. Validation
        model.eval()
        hr1, hr5, hr10 = 0, 0, 0
        ndcg1, ndcg5, ndcg10 = 0, 0, 0
        with torch.no_grad():
            for seq_batch, tgt_batch in val_loader:
                seq_batch = seq_batch.to(device)
                tgt_batch = tgt_batch.to(device).squeeze()
                preds = model(seq_batch)['logits']
                bs = seq_batch.size(0)
                hr1 += hit_rate_at_k(preds, tgt_batch, 1) * bs
                hr5 += hit_rate_at_k(preds, tgt_batch, 5) * bs
                hr10 += hit_rate_at_k(preds, tgt_batch, 10) * bs
                ndcg1 += ndcg_at_k(preds, tgt_batch, 1) * bs
                ndcg5 += ndcg_at_k(preds, tgt_batch, 5) * bs
                ndcg10 += ndcg_at_k(preds, tgt_batch, 10) * bs
        n_val = len(val_ds)
        print(f"Epoch {epoch} Val   HR@1: {hr1/n_val:.4f}, HR@5: {hr5/n_val:.4f}, HR@10: {hr10/n_val:.4f}")
        print(f"           NDCG@1: {ndcg1/n_val:.4f}, NDCG@5: {ndcg5/n_val:.4f}, NDCG@10: {ndcg10/n_val:.4f}")

    # 7. Final Test Evaluation
    model.eval()
    hr1, hr5, hr10 = 0, 0, 0
    ndcg1, ndcg5, ndcg10 = 0, 0, 0
    with torch.no_grad():
        for seq_batch, tgt_batch in test_loader:
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device).squeeze()
            preds = model(seq_batch)['logits']
            bs = seq_batch.size(0)
            hr1 += hit_rate_at_k(preds, tgt_batch, 1) * bs
            hr5 += hit_rate_at_k(preds, tgt_batch, 5) * bs
            hr10 += hit_rate_at_k(preds, tgt_batch, 10) * bs
            ndcg1 += ndcg_at_k(preds, tgt_batch, 1) * bs
            ndcg5 += ndcg_at_k(preds, tgt_batch, 5) * bs
            ndcg10 += ndcg_at_k(preds, tgt_batch, 10) * bs
    n_test = len(test_ds)
    print(f"Test  HR@1: {hr1/n_test:.4f}, HR@5: {hr5/n_test:.4f}, HR@10: {hr10/n_test:.4f}")
    print(f"      NDCG@1: {ndcg1/n_test:.4f}, NDCG@5: {ndcg5/n_test:.4f}, NDCG@10: {ndcg10/n_test:.4f}")
