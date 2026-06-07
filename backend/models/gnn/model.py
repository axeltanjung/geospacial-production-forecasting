"""
Graph Neural Network Models for Well Production Forecasting.

Implements:
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- Combined GCN-GAT Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree
        output = torch.mm(norm_adj, support)
        if self.bias is not None:
            output += self.bias
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim, 1))
        self.a_tgt = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_tgt)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)

        attn_src = torch.einsum("nhd,hdk->nh", h, self.a_src)
        attn_tgt = torch.einsum("nhd,hdk->nh", h, self.a_tgt)

        attn = attn_src.unsqueeze(2) + attn_tgt.unsqueeze(1)
        attn = self.leaky_relu(attn)

        mask = (adj == 0).unsqueeze(-1).expand_as(attn.permute(2, 0, 1)).permute(1, 2, 0)
        attn = attn.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        h_prime = torch.einsum("nmh,nhd->nhd", attn, h)
        return h_prime.reshape(N, -1)


class WellGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(GraphConvLayer(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            h = layer(x, adj)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            if h.shape == x.shape:
                h = h + x
            x = h
        return self.output_head(x)


class WellGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        h = self.gat1(x, adj)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout(h) + x
        x = h
        h = self.gat2(x, adj)
        h = self.norm2(h)
        h = F.elu(h)
        h = self.dropout(h) + x
        return self.output_head(h)


class WellProductionGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gcn_branch = nn.Sequential(
            GraphConvLayer(input_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GraphConvLayer(input_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim * 2)  # mean + variance
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.norm1(self.gcn1(x, adj)))
        h = self.dropout(h)
        h = F.relu(self.norm2(self.gcn2(h, adj)))
        h = self.dropout(h)
        out = self.output_head(h)
        mean, log_var = torch.chunk(out, 2, dim=-1)
        std = torch.exp(0.5 * log_var).clamp(min=1e-6)
        return mean, std
