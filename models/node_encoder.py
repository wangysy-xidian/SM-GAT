import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class TrafficNumericalEmbedding(nn.Module):
    """
    Step 1: Numerical Alignment Module (The Translator)
    Goal: Handle numerical drift (e.g., packet size 1448 vs 1450) via soft-quantization.
    """
    def __init__(self, num_kernels=64, embedding_dim=128, min_sigma=1.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.embedding_dim = embedding_dim
        self.min_sigma = min_sigma

        # 1. Learnable Anchors (Mu) and Widths (Sigma)
        self.mu = nn.Parameter(torch.randn(num_kernels), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.zeros(num_kernels), requires_grad=True)

        # 2. Embeddings for Magnitude and Direction
        self.kernel_embeddings = nn.Embedding(num_kernels, embedding_dim)
        # 0: Padding, 1: Downlink (Negative), 2: Uplink (Positive)
        self.sign_embeddings = nn.Embedding(3, embedding_dim)

        # 3. Fusion Layer
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def init_anchors_with_kmeans(self, raw_values):
        """
        Initialize anchors using K-Means to prevent cold-start issues.
        Args:
            raw_values: List of floats/ints from the training set.
        """
        print(f"[*] Initializing anchors with K-Means on {len(raw_values)} samples...")
        abs_values = np.abs(np.array(raw_values)).reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.num_kernels, n_init=10, random_state=42)
        kmeans.fit(abs_values)

        centers = np.sort(kmeans.cluster_centers_.flatten())

        with torch.no_grad():
            self.mu.copy_(torch.from_numpy(centers).float())
            # Initialize Sigma based on half the distance between adjacent anchors
            if len(centers) > 1:
                avg_dist = np.mean(centers[1:] - centers[:-1])
                self.log_sigma.fill_(np.log(avg_dist + 1e-5))
        print("[*] Anchor initialization complete.")

    def get_sigma(self):
        return F.softplus(self.log_sigma) + self.min_sigma

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len] (Raw numerical values, e.g., 1448, -32)
        Returns:
            [Batch, Seq_Len, Embedding_Dim]
        """
        # A. Decouple Magnitude and Direction
        x_abs = torch.abs(x)
        x_sign = torch.sign(x)  # -1, 0, 1

        # Map Sign: 0(Pad)->0, -1(Down)->1, 1(Up)->2
        sign_indices = torch.zeros_like(x_sign, dtype=torch.long)
        sign_indices[x_sign < 0] = 1
        sign_indices[x_sign > 0] = 2

        # B. Gaussian Soft-Quantization
        x_expanded = x_abs.unsqueeze(-1)      # [B, S, 1]
        mu_expanded = self.mu.view(1, 1, -1)  # [1, 1, K]
        sigma = self.get_sigma().view(1, 1, -1)

        dist_sq = (x_expanded - mu_expanded) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # [B, S, K]
        # Normalize weights
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

        # C. Lookup and Aggregation
        mag_emb = torch.matmul(weights, self.kernel_embeddings.weight)  # [B, S, Dim]
        sign_emb = self.sign_embeddings(sign_indices)                   # [B, S, Dim]

        # D. Fusion
        return self.output_proj(mag_emb + sign_emb)


class IntraNodeEncoder(nn.Module):
    """
    Step 2: Intra-Node Feature Encoder (The Processor)
    Goal: Handle structural drift by fusing sequence patterns with global statistics.
    """
    def __init__(self, input_dim=128, hidden_dim=256, stat_dim=5):
        super().__init__()

        # 1. Multi-Scale 1D-CNN (Captures local patterns)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=5, padding=2)
        rem = hidden_dim - 2 * (hidden_dim // 3)
        self.conv7 = nn.Conv1d(input_dim, rem, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # 2. FiLM Generator (Modulates features using statistical context)
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2)  # Outputs Gamma and Beta
        )

        # 3. Attention Pooling
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, seq_emb, stat_features):
        """
        Args:
            seq_emb: [Batch, Seq_Len, Input_Dim]
            stat_features: [Batch, Stat_Dim] (Normalized statistics)
        """
        # A. CNN Feature Extraction
        x = seq_emb.permute(0, 2, 1)  # [B, Dim, Seq]
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        h_seq = torch.cat([x3, x5, x7], dim=1)  # [B, Hidden, Seq]
        h_seq = self.bn(h_seq)

        # B. FiLM Modulation
        film_params = self.stat_mlp(stat_features)  # [B, 2*H]
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2)  # [B, H, 1]
        beta = beta.unsqueeze(2)

        h_modulated = h_seq * (1 + gamma) + beta

        # C. Attention Pooling
        h_modulated = h_modulated.permute(0, 2, 1)  # [B, Seq, Hidden]
        scores = self.attn_score(h_modulated)       # [B, Seq, 1]
        weights = F.softmax(scores, dim=1)

        h_node = torch.sum(h_modulated * weights, dim=1)  # [B, Hidden]
        return h_node


class TrafficNodeModel(nn.Module):
    """
    Main Wrapper: Interface for node-level feature extraction.
    """
    def __init__(self, num_kernels=64, embedding_dim=128, hidden_dim=256, stat_dim=5):
        super().__init__()
        self.embedder = TrafficNumericalEmbedding(num_kernels, embedding_dim)
        self.encoder = IntraNodeEncoder(embedding_dim, hidden_dim, stat_dim)

    def init_anchors(self, raw_values):
        self.embedder.init_anchors_with_kmeans(raw_values)

    def forward(self, raw_seq, raw_stats):
        seq_vectors = self.embedder(raw_seq)
        node_vector = self.encoder(seq_vectors, raw_stats)
        return node_vector
