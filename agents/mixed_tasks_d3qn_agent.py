"""mixed_tasks D3QN agent (Dueling + Double, uniform replay)."""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from models.action_mask import parse_experience_batch
from models.mixed_tasks_d3qn import MixedTasksD3QNNetwork
from models.mixed_tasks_state_layout import (
    build_mixed_tasks_action_mask_from_layout,
    compute_mixed_tasks_packed_dim,
    task_is_completed_from_mixed_tasks_state,
    unpack_mixed_tasks_state,
)


class MixedTasksD3QN_Agent:
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
        action_dim: int = 5,
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
        self.packed_state_dim = compute_mixed_tasks_packed_dim()

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
        self.online_net = MixedTasksD3QNNetwork(**net_kwargs).to(device)
        self.target_net = MixedTasksD3QNNetwork(**net_kwargs).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.replay_buffer = deque(maxlen=buffer_length)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

    @staticmethod
    def _batch_next_masks(states: torch.Tensor) -> torch.Tensor:
        masks = []
        for idx in range(states.shape[0]):
            task_state, _, _, _, node_mask = unpack_mixed_tasks_state(states[idx].detach().cpu().numpy())
            is_completed = task_is_completed_from_mixed_tasks_state(task_state)

            masks.append(build_mixed_tasks_action_mask_from_layout(node_mask, is_completed))
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

            curr_q = self.online_net(state, action_mask).gather(1, action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_online = self.online_net(next_state, next_action_mask)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_state, next_action_mask).gather(1, next_actions).squeeze(1)
                expected_q = reward + (1.0 - done) * self.gamma * next_q_target

            loss = F.mse_loss(curr_q, expected_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

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
