import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset

URI = "bolt://localhost:7687"
AUTH = ("", "")

class AE(nn.Module):
    def __init__(self, d, code=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, code))
        self.dec = nn.Sequential(nn.Linear(code, 16), nn.ReLU(), nn.Linear(16, d))
    def forward(self, x): 
        return self.dec(self.enc(x))

def extract_and_convert():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        node_result = session.run("MATCH (a:Account) RETURN a.id AS account_id")
        account_ids = [record["account_id"] for record in node_result]
        node_mapping = {acc_id: i for i, acc_id in enumerate(account_ids)}
        num_nodes = len(account_ids)

        # Pull raw 1-hop transactions. We handle multi-hop and time bounding dynamically in PyG.
        edge_query = """
        MATCH (s:Account)-[r:TRANSFERRED]->(rec:Account)
        RETURN s.id AS source, rec.id AS target, r.amount AS amount, r.timestamp AS timestamp, r.is_laundering AS label
        """
        edge_result = session.run(edge_query)

        sources, targets, amounts, raw_timestamps, labels = [], [], [], [], []
        for record in edge_result:
            sources.append(node_mapping[record["source"]])
            targets.append(node_mapping[record["target"]])
            amounts.append(record["amount"] if record["amount"] is not None else 0.0)
            raw_timestamps.append(record["timestamp"])
            labels.append(record["label"] if record["label"] is not None else 0)
            
    driver.close()

    src = torch.tensor(sources, dtype=torch.long)
    tgt = torch.tensor(targets, dtype=torch.long)
    amount = torch.tensor(amounts, dtype=torch.float)
    y_edge = torch.tensor(labels, dtype=torch.long)
    
    # 1. CHRONOLOGICAL SORTING
    timestamps_pd = pd.to_datetime(raw_timestamps)
    ts_sec = torch.tensor(timestamps_pd.astype('int64').values // 10**9, dtype=torch.long)
    
    order = torch.argsort(ts_sec)
    src, tgt, y_edge, ts_sec = src[order], tgt[order], y_edge[order], ts_sec[order]
    amount = amount[order]
    
    n = len(order)
    tr_end = int(0.6 * n)
    train_mask = torch.arange(n) < tr_end

    # 2. FEATURE ENGINEERING
    hour = (ts_sec % 86400) / 86400 * 2 * torch.pi
    out_deg_train = torch.bincount(src[train_mask], minlength=num_nodes).float()
    in_deg_train = torch.bincount(tgt[train_mask], minlength=num_nodes).float()
    
    feats = torch.stack([
        torch.log1p(amount),
        torch.sin(hour), 
        torch.cos(hour),
        torch.log1p(out_deg_train[src]),
        torch.log1p(in_deg_train[tgt])
    ], dim=1)

    mu, sd = feats[train_mask].mean(0), feats[train_mask].std(0) + 1e-6
    feats = (feats - mu) / sd

    # 3. AUTOENCODER ANOMALY SCORING
    ae = AE(d=feats.shape[1], code=2)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(feats[train_mask]), batch_size=8192, shuffle=True)
    
    for epoch in range(5):
        ae.train()
        for (xb,) in loader:
            opt.zero_grad()
            loss = F.mse_loss(ae(xb), xb)
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        err = ((feats - ae(feats)) ** 2).mean(1)
    
    # Standardized score for the neural network input
    score_log = torch.log1p(err)
    score_feature = (score_log - score_log[train_mask].mean()) / (score_log[train_mask].std() + 1e-6)

    # Strictly positive probability distribution for the PyG anomaly-biased sampler
    sampling_weight = (err - err.min()) / (err.max() - err.min()) + 1e-4

    edge_index = torch.stack([src, tgt], dim=0)
    final_edge_attr = torch.cat([feats, score_feature.unsqueeze(1)], dim=1)
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=final_edge_attr, 
        y_edge=y_edge, 
        time=ts_sec,
        edge_weight=sampling_weight, # Active hunting distribution
        num_nodes=num_nodes
    )
    
    return data, tr_end