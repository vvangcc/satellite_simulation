"""DGR resource encoder (paper section 3.4)."""
from __future__ import annotations

import torch
from torch import nn


class DGRResourceEncoder(nn.Module):
    """Dynamic graph reconfiguration encoder with 2-layer attentive message passing."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        top_k: int = 2,
        lambda_mix: float = 0.6,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.top_k = top_k
        self.lambda_mix = lambda_mix
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, 1) for _ in range(num_layers)]
        )
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.residual_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.w_a_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    @staticmethod
    def affinity_score(neighbor_feats: torch.Tensor) -> torch.Tensor:
        m = neighbor_feats[..., 0]
        q_t = neighbor_feats[..., 1]
        q_c = neighbor_feats[..., 2]
        return 0.2 * m - 0.4 * q_t - 0.4 * q_c

    def _build_reconnected_adj(
        self,
        physical_adj: torch.Tensor,
        node_mask: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Build weighted A_e = lambda*A_p + (1-lambda)*A_r with Top-K affinity weights."""
        batch_size, num_nodes, _ = physical_adj.shape
        reconnected = torch.zeros_like(physical_adj)
        affinity = self.affinity_score(node_feats)

        for b in range(batch_size):
            valid = node_mask[b] > 0.5
            valid_indices = torch.where(valid)[0]
            if len(valid_indices) <= 1:
                continue
            center_idx = 0
            neighbor_indices = valid_indices[valid_indices != center_idx]
            if len(neighbor_indices) == 0:
                continue
            scores = affinity[b, neighbor_indices]
            k = min(self.top_k, len(neighbor_indices))
            topk = torch.topk(scores, k=k)
            topk_nodes = neighbor_indices[topk.indices]
            weights = torch.softmax(topk.values, dim=0)
            for node_j, weight in zip(topk_nodes, weights):
                reconnected[b, center_idx, node_j] = weight
                reconnected[b, node_j, center_idx] = weight

        fused = self.lambda_mix * physical_adj + (1.0 - self.lambda_mix) * reconnected
        return fused

    def forward(
        self,
        node_feats: torch.Tensor,
        physical_adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = node_feats.shape
        adj = self._build_reconnected_adj(physical_adj, node_mask, node_feats)

        h = self.leaky_relu(self.input_proj(node_feats))
        mask_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        edge_bias = torch.log(adj + 1e-6)

        for layer_idx in range(self.num_layers):
            wa_h = self.w_a_layers[layer_idx](h)
            wa_i = wa_h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            wa_j = wa_h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            attn_input = torch.cat([wa_i, wa_j], dim=-1)
            attn_logits = self.attn_layers[layer_idx](attn_input).squeeze(-1)
            attn_logits = attn_logits + edge_bias
            attn_logits = attn_logits.masked_fill(adj <= 0, float("-inf"))
            attn_logits = attn_logits.masked_fill(mask_pair <= 0, float("-inf"))
            alpha = torch.softmax(attn_logits, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            aggregated = torch.matmul(alpha, h)
            m_l = node_mask.unsqueeze(-1)
            h_new = self.leaky_relu(self.self_layers[layer_idx](h) + aggregated)
            h = h_new * m_l + self.residual_layers[layer_idx](h) * m_l

        return h[:, 0, :]
