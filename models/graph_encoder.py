import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention

# Assumes node_encoder.py is in the same directory or package
from .node_encoder import TrafficNodeModel


class TimeEncoder(nn.Module):
    """
    Encodes continuous time intervals (dt) into vectors using Sinusoidal positional encoding.
    Handles large variances in time gaps (milliseconds to seconds).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Frequency term: 1 / 10000^(2i/d_model)
        self.div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        self.register_buffer('div_term_tensor', self.div_term)

    def forward(self, dt):
        """
        Args:
            dt: [Num_Edges, 1]
        """
        pe = dt * self.div_term_tensor
        encoding = torch.zeros(dt.shape[0], self.embedding_dim, device=dt.device)
        encoding[:, 0::2] = torch.sin(pe)
        encoding[:, 1::2] = torch.cos(pe)
        return encoding


class TrafficGraphModel(nn.Module):
    """
    SM-GAT Main Model Architecture.
    Integrates Node Encoding (Micro) and Graph Reasoning (Macro).
    """

    def __init__(self, num_classes,
                 num_kernels=64, embed_dim=128, hidden_dim=256,
                 gat_heads=4, gat_layers=2):
        super().__init__()

        # === Part 1: Node Encoder ===
        self.node_processor = TrafficNodeModel(
            num_kernels=num_kernels,
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            stat_dim=5
        )

        # === Part 2: Edge Encoder (Type + Time) ===
        # Edge Types: 0=Intra-Burst, 1=Inter-Burst(Seq), 2=Inter-Burst(Anchor)
        self.edge_type_emb = nn.Embedding(3, hidden_dim)
        self.time_encoder = TimeEncoder(hidden_dim)
        self.edge_fusion = nn.Linear(hidden_dim, hidden_dim)

        # === Part 3: Graph Encoder (Relation-Aware GAT) ===
        self.gat_layers = nn.ModuleList()
        # First GAT Layer
        self.gat_layers.append(
            GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                      heads=gat_heads,
                      edge_dim=hidden_dim,
                      concat=True)
        )
        # Subsequent GAT Layers
        for _ in range(gat_layers - 1):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // gat_heads,
                          heads=gat_heads,
                          edge_dim=hidden_dim,
                          concat=True)
            )

        # === Part 4: Readout & Classifier ===
        # Global Attention Pooling with Gate NN
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.pool = GlobalAttention(gate_nn=self.readout_gate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Batch object containing:
                - x_seq: Packet size sequences
                - x_stats: Statistical features
                - edge_index, edge_attr: Graph topology and edge features
                - batch: Batch assignment indices
        """
        # 1. Node Encoding
        node_embeddings = self.node_processor(data.x_seq, data.x_stats)

        # 2. Edge Encoding
        edge_types = data.edge_attr[:, 0].long()
        time_diffs = data.edge_attr[:, 1].unsqueeze(1)

        e_type_emb = self.edge_type_emb(edge_types)
        e_time_emb = self.time_encoder(time_diffs)

        # Fuse semantics and time context
        edge_embeddings = self.edge_fusion(e_type_emb + e_time_emb)

        # 3. Graph Reasoning (R-GAT)
        x = node_embeddings
        for layer in self.gat_layers:
            x = layer(x, data.edge_index, edge_attr=edge_embeddings)
            x = F.elu(x)

        # 4. Global Readout
        graph_emb = self.pool(x, data.batch)

        # 5. Classification
        logits = self.classifier(graph_emb)

        return logits
