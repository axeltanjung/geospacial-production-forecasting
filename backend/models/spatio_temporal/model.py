"""
Spatio-Temporal Graph Neural Network (ST-GNN).

Combines:
- Temporal encoder (LSTM) for time-series patterns
- Spatial encoder (GNN) for well-to-well dependencies
- Fusion layer for joint prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x)
        return self.norm(hidden[-1])


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.W1(x)))
        N = h.size(0)
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        attn_input = torch.cat([h_i, h_j], dim=-1)
        attn_scores = self.attention(attn_input).squeeze(-1)
        attn_scores = attn_scores.masked_fill(adj == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        aggregated = torch.mm(attn_weights, h)
        output = F.relu(self.norm2(self.W2(aggregated + h)))
        return self.dropout(output)


class SpatioTemporalGNN(nn.Module):
    def __init__(
        self,
        temporal_input_dim: int,
        spatial_input_dim: int,
        temporal_hidden: int = 64,
        spatial_hidden: int = 64,
        fusion_hidden: int = 128,
        output_dim: int = 3,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            temporal_input_dim, temporal_hidden, num_lstm_layers, dropout
        )
        self.spatial_encoder = SpatialEncoder(
            spatial_input_dim, spatial_hidden, dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(temporal_hidden + spatial_hidden, fusion_hidden),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mean_head = nn.Linear(fusion_hidden // 2, output_dim)
        self.var_head = nn.Linear(fusion_hidden // 2, output_dim)

    def forward(
        self,
        temporal_input: torch.Tensor,
        spatial_input: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_features = self.temporal_encoder(temporal_input)
        spatial_features = self.spatial_encoder(spatial_input, adj)
        fused = torch.cat([temporal_features, spatial_features], dim=-1)
        hidden = self.fusion(fused)
        mean = self.mean_head(hidden)
        log_var = self.var_head(hidden)
        std = F.softplus(log_var)
        return mean, std


class STGNNTrainer:
    def __init__(self, model: SpatioTemporalGNN, lr: float = 0.001, weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def gaussian_nll_loss(self, mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = std ** 2
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
        return nll.mean()

    def train_step(
        self,
        temporal_input: torch.Tensor,
        spatial_input: torch.Tensor,
        adj: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        mean, std = self.model(temporal_input, spatial_input, adj)
        loss = self.gaussian_nll_loss(mean, std, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def evaluate(
        self,
        temporal_input: torch.Tensor,
        spatial_input: torch.Tensor,
        adj: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(temporal_input, spatial_input, adj)
            loss = self.gaussian_nll_loss(mean, std, target)
            mae = F.l1_loss(mean, target)
            mse = F.mse_loss(mean, target)

        self.scheduler.step(loss)
        return {
            "loss": loss.item(),
            "mae": mae.item(),
            "rmse": torch.sqrt(mse).item()
        }
