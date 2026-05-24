"""Dueling Double DQN with action masking (paper section 3.5)."""
from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.action_mask import parse_experience_batch
from models.dgr_encoder import DGRResourceEncoder
from models.state_layout import (
    ACTION_DIM,
    DEST_STATE_DIM,
    MAX_GRAPH_NODES,
    RESOURCE_FEAT_DIM,
    TASK_STATE_DIM,
    build_action_mask_from_layout,
    compute_dgr_packed_dim,
    task_is_completed_from_state,
    unpack_dgr_state,
)


def compute_ddqn_bootstrap_target(
    online_net,
    target_net,
    next_state: torch.Tensor,
    next_action_mask: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Double-DQN bootstrap with safe handling for terminal and empty action masks."""
    with torch.no_grad():
        target_term = torch.zeros_like(reward)
        not_done = done < 0.5
        if not_done.any():
            nd_indices = not_done.nonzero(as_tuple=True)[0]
            nd_masks = next_action_mask[nd_indices]
            has_valid = nd_masks.sum(dim=1) > 0.5
            if has_valid.any():
                valid_indices = nd_indices[has_valid]
                ns = next_state[valid_indices]
                nm = next_action_mask[valid_indices]
                next_q_online = online_net(ns, nm)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = target_net(ns, nm).gather(1, next_actions).squeeze(1)
                target_term[valid_indices] = next_q_target
        return reward + gamma * target_term


def get_activation(act_type: str, negative_slope: float = 0.01):
    if act_type == "LeakyRelu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    if act_type == "Relu":
        return nn.ReLU()
    return nn.Identity()


class DuelingQHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        activation: str = "LeakyRelu",
        hidden_layers: int = 2,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation, negative_slope)
        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.mid_acts = nn.ModuleList(
            [get_activation(activation, negative_slope) for _ in range(hidden_layers)]
        )
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_layer(x))
        for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
            x = mid_act(mid_layer(x))
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        return value + advantages - advantages.mean(dim=1, keepdim=True)


class DGRD3QNNetwork(nn.Module):
    """End-to-end DGR encoder + dueling Q head."""

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
        q_input_dim = TASK_STATE_DIM + dgr_hidden_dim + DEST_STATE_DIM
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
        dest_state: torch.Tensor,
        node_feats: torch.Tensor,
        phys_adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        s_r = self.encoder(node_feats, phys_adj, node_mask)
        return torch.cat([task_state, s_r, dest_state], dim=-1)

    def forward_from_components(
        self,
        task_state: torch.Tensor,
        dest_state: torch.Tensor,
        node_feats: torch.Tensor,
        phys_adj: torch.Tensor,
        node_mask: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.encode_state(task_state, dest_state, node_feats, phys_adj, node_mask)
        q_values = self.q_head(state)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask <= 0, float("-inf"))
        return q_values

    def forward(self, packed_state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        task_state, dest_state, node_feats, phys_adj, node_mask = self.unpack_batch(packed_state)
        return self.forward_from_components(
            task_state, dest_state, node_feats, phys_adj, node_mask, action_mask
        )

    @staticmethod
    def unpack_batch(packed_state: torch.Tensor):
        batch_size = packed_state.shape[0]
        task_state = packed_state[:, :TASK_STATE_DIM]
        offset = TASK_STATE_DIM
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
        return task_state, dest_state, node_feats, phys_adj, node_mask


class D3QN_Agent:
    def __init__(
        self,
        buffer_length: int,
        batch_size: int,
        gamma: float,
        device,
        learning_rate: float = 2e-4,
        repeat: int = 1,
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
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.repeat = repeat
        self.action_dim = action_dim
        self.packed_state_dim = compute_dgr_packed_dim()

        net_kwargs = dict(
            dgr_hidden_dim=dgr_hidden_dim,
            q_hidden_dim=q_hidden_dim,
            action_dim=action_dim,
            activation=activation,
            q_hidden_layers=q_hidden_layers,
            gnn_layers=gnn_layers,
            top_k=top_k,
            lambda_mix=lambda_mix,
            negative_slope=negative_slope,
        )
        self.online_net = DGRD3QNNetwork(**net_kwargs).to(device)
        self.target_net = DGRD3QNNetwork(**net_kwargs).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.replay_buffer = deque(maxlen=buffer_length)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.last_loss = None

    @staticmethod
    def _batch_action_masks(states: torch.Tensor) -> torch.Tensor:
        masks = []
        for idx in range(states.shape[0]):
            task_state, _, _, _, node_mask = unpack_dgr_state(states[idx].detach().cpu().numpy())
            is_completed = task_is_completed_from_state(task_state)
            masks.append(build_action_mask_from_layout(node_mask, is_completed))
        return torch.tensor(np.asarray(masks, dtype=np.float32), device=states.device)

    def update(self, experiences):
        self.replay_buffer.extend(experiences)
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]
            state, action_mask, action, reward, next_state, next_action_mask, done = parse_experience_batch(batch)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

            curr_q = self.online_net(state, action_mask)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                expected_q = compute_ddqn_bootstrap_target(
                    self.online_net,
                    self.target_net,
                    next_state,
                    next_action_mask,
                    reward,
                    done,
                    self.gamma,
                )

            loss = F.mse_loss(curr_q, expected_q)
            self.last_loss = float(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        print("Target network updated")

    def save_model(self, file_path):
        if file_path:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            state_dict = torch.load(file_path, map_location=self.device)
            self.online_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)


class FlatD3QNNetwork(nn.Module):
    """Dueling Double DQN on packed DGR state without GNN encoder (ablation baseline)."""

    def __init__(
        self,
        state_dim: int,
        q_hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        activation: str = "LeakyRelu",
        q_hidden_layers: int = 2,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.q_head = DuelingQHead(
            state_dim,
            q_hidden_dim,
            action_dim,
            activation=activation,
            hidden_layers=q_hidden_layers,
            negative_slope=negative_slope,
        )

    def forward(self, packed_state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_values = self.q_head(packed_state)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask <= 0, float("-inf"))
        return q_values


class FlatD3QN_Agent:
    """Flat D3QN: dueling + double DQN + action mask, no DGR encoder."""

    def __init__(
        self,
        buffer_length: int,
        batch_size: int,
        gamma: float,
        device,
        learning_rate: float = 2e-4,
        repeat: int = 1,
        q_hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        activation: str = "LeakyRelu",
        q_hidden_layers: int = 2,
        negative_slope: float = 0.01,
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.repeat = repeat
        self.action_dim = action_dim
        self.packed_state_dim = compute_dgr_packed_dim()

        net_kwargs = dict(
            state_dim=self.packed_state_dim,
            q_hidden_dim=q_hidden_dim,
            action_dim=action_dim,
            activation=activation,
            q_hidden_layers=q_hidden_layers,
            negative_slope=negative_slope,
        )
        self.online_net = FlatD3QNNetwork(**net_kwargs).to(device)
        self.target_net = FlatD3QNNetwork(**net_kwargs).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.replay_buffer = deque(maxlen=buffer_length)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.last_loss = None

    def update(self, experiences):
        self.replay_buffer.extend(experiences)
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]
            state, action_mask, action, reward, next_state, next_action_mask, done = parse_experience_batch(batch)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

            curr_q = self.online_net(state, action_mask).gather(1, action.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                expected_q = compute_ddqn_bootstrap_target(
                    self.online_net,
                    self.target_net,
                    next_state,
                    next_action_mask,
                    reward,
                    done,
                    self.gamma,
                )

            loss = F.mse_loss(curr_q, expected_q)
            self.last_loss = float(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        print("Target network updated")

    def save_model(self, file_path):
        if file_path:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            state_dict = torch.load(file_path, map_location=self.device)
            self.online_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
