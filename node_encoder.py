import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


# ==========================================
# Step 1: 数值对齐模块 (The Translator)
# 目标：抗数值漂移 (e.g., 1448 vs 1450)
# ==========================================
class TrafficNumericalEmbedding(nn.Module):
    def __init__(self, num_kernels=64, embedding_dim=128, min_sigma=1.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.embedding_dim = embedding_dim
        self.min_sigma = min_sigma

        # 1. 可学习的锚点 (Mu) 和 宽度 (Sigma)
        self.mu = nn.Parameter(torch.randn(num_kernels), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.zeros(num_kernels), requires_grad=True)

        # 2. 幅度嵌入 (Magnitude) 和 方向嵌入 (Sign/Direction)
        self.kernel_embeddings = nn.Embedding(num_kernels, embedding_dim)
        # 0: Padding, 1: Downlink (负数), 2: Uplink (正数)
        self.sign_embeddings = nn.Embedding(3, embedding_dim)

        # 3. 融合层
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def init_anchors_with_kmeans(self, raw_values):
        """
        [重要] 使用 K-Means 初始化锚点，防止冷启动问题。
        raw_values: 训练集中所有包大小的列表 (list of floats/ints)
        """
        print(f"[*] Initializing anchors with K-Means on {len(raw_values)} samples...")
        abs_values = np.abs(np.array(raw_values)).reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.num_kernels, n_init=10, random_state=42)
        kmeans.fit(abs_values)

        centers = np.sort(kmeans.cluster_centers_.flatten())

        with torch.no_grad():
            self.mu.copy_(torch.from_numpy(centers).float())
            # 初始化 Sigma 为相邻锚点间距的一半，保证覆盖
            if len(centers) > 1:
                avg_dist = np.mean(centers[1:] - centers[:-1])
                self.log_sigma.fill_(np.log(avg_dist + 1e-5))
        print("[*] Anchor initialization complete.")

    def get_sigma(self):
        return F.softplus(self.log_sigma) + self.min_sigma

    def forward(self, x):
        """
        x: [Batch, Seq_Len] (原始数值，如 1448, -32)
        Returns: [Batch, Seq_Len, Embedding_Dim]
        """
        # A. 解耦方向与大小
        x_abs = torch.abs(x)
        x_sign = torch.sign(x)  # -1, 0, 1

        # 映射 Sign: 0(Pad)->0, -1(Down)->1, 1(Up)->2
        sign_indices = torch.zeros_like(x_sign, dtype=torch.long)
        sign_indices[x_sign < 0] = 1
        sign_indices[x_sign > 0] = 2

        # B. Soft-Quantization (计算高斯相似度)
        x_expanded = x_abs.unsqueeze(-1)  # [B, S, 1]
        mu_expanded = self.mu.view(1, 1, -1)  # [1, 1, K]
        sigma = self.get_sigma().view(1, 1, -1)

        dist_sq = (x_expanded - mu_expanded) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # [B, S, K]
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)  # 归一化

        # C. 查表与聚合
        mag_emb = torch.matmul(weights, self.kernel_embeddings.weight)  # [B, S, Dim]
        sign_emb = self.sign_embeddings(sign_indices)  # [B, S, Dim]

        # D. 融合
        return self.output_proj(mag_emb + sign_emb)


# ==========================================
# Step 2: 节点内特征编码 (The Processor)
# 目标：抗结构漂移 (Sequence + Statistics Fusion)
# ==========================================
class IntraNodeEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, stat_dim=5):
        super().__init__()

        # 1. Multi-Scale 1D-CNN (捕捉局部模式)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=5, padding=2)
        rem = hidden_dim - 2 * (hidden_dim // 3)
        self.conv7 = nn.Conv1d(input_dim, rem, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # 2. FiLM Generator (利用统计特征生成调节参数)
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2)  # 输出 Gamma 和 Beta
        )

        # 3. Attention Pooling (自动忽略 Padding)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, seq_emb, stat_features):
        """
        seq_emb: [Batch, Seq_Len, Input_Dim] (来自 Step 1)
        stat_features: [Batch, Stat_Dim] (归一化后的统计值)
        """
        # A. CNN 提取特征
        x = seq_emb.permute(0, 2, 1)  # [B, Dim, Seq]
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        h_seq = torch.cat([x3, x5, x7], dim=1)  # [B, Hidden, Seq]
        h_seq = self.bn(h_seq)

        # B. FiLM 融合 (统计特征调制序列特征)
        film_params = self.stat_mlp(stat_features)  # [B, 2*H]
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2)  # [B, H, 1]
        beta = beta.unsqueeze(2)

        h_modulated = h_seq * (1 + gamma) + beta

        # C. Attention Pooling
        h_modulated = h_modulated.permute(0, 2, 1)  # [B, Seq, Hidden]
        scores = self.attn_score(h_modulated)  # [B, Seq, 1]

        # Masking (可选): 如果输入全是0(padding)，这里可以加mask，
        # 但由于Attention会自动给不重要的部分低分，且CNN有padding，通常直接softmax即可
        weights = F.softmax(scores, dim=1)

        h_node = torch.sum(h_modulated * weights, dim=1)  # [B, Hidden]
        return h_node


# ==========================================
# Main Wrapper: 对外接口类
# ==========================================
class TrafficNodeModel(nn.Module):
    def __init__(self, num_kernels=64, embedding_dim=128, hidden_dim=256, stat_dim=5):
        super().__init__()
        # 实例化 Step 1
        self.embedder = TrafficNumericalEmbedding(num_kernels, embedding_dim)
        # 实例化 Step 2
        self.encoder = IntraNodeEncoder(embedding_dim, hidden_dim, stat_dim)

    def init_anchors(self, raw_values):
        """ 暴露给外部的初始化接口 """
        self.embedder.init_anchors_with_kmeans(raw_values)

    def forward(self, raw_seq, raw_stats):
        """
        raw_seq: [Batch, Seq_Len] (Float Tensor) - 包大小序列
        raw_stats: [Batch, Stat_Dim] (Float Tensor) - 统计特征
        """
        # Step 1: 数值 -> 向量序列
        seq_vectors = self.embedder(raw_seq)

        # Step 2: 向量序列 + 统计特征 -> 节点向量
        node_vector = self.encoder(seq_vectors, raw_stats)

        return node_vector