"""mixed_tasks DGR-D3QN network with extended task/queue state."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.d3qn import DuelingQHead
from models.dgr_encoder import DGRResourceEncoder
from models.mixed_tasks_state_layout import (
    ACTION_DIM,
    DEST_STATE_DIM,
    MAX_GRAPH_NODES,
    MIXED_TASKS_QUEUE_STATE_DIM,
    MIXED_TASKS_TASK_STATE_DIM,
    RESOURCE_FEAT_DIM,
)


class MixedTasksD3QNNetwork(nn.Module):
    """DGR encoder + dueling Q head for mixed_tasks packed state."""

    def __init__(
        self,
        dgr_hidden_dim: int = 64,
        q_hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        activation: str = "LeakyRelu",
        q_hidden_layers: int = 2,
        gnn_layers: int = 2,
        top_k: int = 2,
        lambda_mix: float = 0.6,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.dgr_hidden_dim = dgr_hidden_dim
        self.encoder = DGRResourceEncoder(
            input_dim=RESOURCE_FEAT_DIM,
            hidden_dim=dgr_hidden_dim,
            num_layers=gnn_layers,
            top_k=top_k,
            lambda_mix=lambda_mix,
            negative_slope=negative_slope,
        )
        q_input_dim = MIXED_TASKS_TASK_STATE_DIM + MIXED_TASKS_QUEUE_STATE_DIM + dgr_hidden_dim + DEST_STATE_DIM
        self.q_head = DuelingQHead(
            q_input_dim,
            q_hidden_dim,
            action_dim,
            activation=activation,
            hidden_layers=q_hidden_layers,
            negative_slope=negative_slope,
        )

    def encode_state(
        self,
        task_state: torch.Tensor,
        queue_state: torch.Tensor,
        dest_state: torch.Tensor,
        node_feats: torch.Tensor,
        phys_adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        s_r = self.encoder(node_feats, phys_adj, node_mask)
        return torch.cat([task_state, queue_state, s_r, dest_state], dim=-1)

    def forward_from_components(
        self,
        task_state: torch.Tensor,
        queue_state: torch.Tensor,
        dest_state: torch.Tensor,
        node_feats: torch.Tensor,
        phys_adj: torch.Tensor,
        node_mask: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.encode_state(task_state, queue_state, dest_state, node_feats, phys_adj, node_mask)
        q_values = self.q_head(state)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask <= 0, float("-inf"))
        return q_values

    def forward(self, packed_state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        task_state, queue_state, dest_state, node_feats, phys_adj, node_mask = self.unpack_batch(packed_state)
        return self.forward_from_components(
            task_state, queue_state, dest_state, node_feats, phys_adj, node_mask, action_mask
        )

    @staticmethod
    def unpack_batch(packed_state: torch.Tensor):
        batch_size = packed_state.shape[0]
        offset = 0
        task_state = packed_state[:, offset: offset + MIXED_TASKS_TASK_STATE_DIM]
        offset += MIXED_TASKS_TASK_STATE_DIM
        queue_state = packed_state[:, offset: offset + MIXED_TASKS_QUEUE_STATE_DIM]
        offset += MIXED_TASKS_QUEUE_STATE_DIM
        dest_state = packed_state[:, offset: offset + DEST_STATE_DIM]
        offset += DEST_STATE_DIM
        node_feat_size = MAX_GRAPH_NODES * RESOURCE_FEAT_DIM
        node_feats = packed_state[:, offset: offset + node_feat_size].reshape(
            batch_size, MAX_GRAPH_NODES, RESOURCE_FEAT_DIM
        )
        offset += node_feat_size
        adj_size = MAX_GRAPH_NODES * MAX_GRAPH_NODES
        phys_adj = packed_state[:, offset: offset + adj_size].reshape(
            batch_size, MAX_GRAPH_NODES, MAX_GRAPH_NODES
        )
        offset += adj_size
        node_mask = packed_state[:, offset: offset + MAX_GRAPH_NODES]
        return task_state, queue_state, dest_state, node_feats, phys_adj, node_mask


def compute_mixed_tasks_q_input_dim(dgr_hidden_dim: int = 64) -> int:
    return MIXED_TASKS_TASK_STATE_DIM + MIXED_TASKS_QUEUE_STATE_DIM + dgr_hidden_dim + DEST_STATE_DIM
