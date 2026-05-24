import random
import unittest

import numpy as np
import torch

from models.action_mask import build_action_mask
from models.dgr_encoder import DGRResourceEncoder
from models.action_mask import parse_experience_batch
from models.d3qn import DGRD3QNNetwork, D3QN_Agent, FlatD3QN_Agent, compute_ddqn_bootstrap_target
from models.state_layout import (
    TASK_IDX_IS_COMPLETED,
    TASK_STATE_DIM,
    build_action_mask_from_layout,
    compute_dgr_packed_dim,
    pack_dgr_state,
    unpack_dgr_state,
)


class DGREncoderTests(unittest.TestCase):
    def test_output_shape_and_affinity_reconnect(self):
        encoder = DGRResourceEncoder(hidden_dim=16, num_layers=2, top_k=2, lambda_mix=0.6)
        node_feats = torch.tensor(
            [
                [
                    [0.8, 0.1, 0.2],
                    [0.7, 0.2, 0.1],
                    [0.6, 0.3, 0.4],
                    [0.5, 0.4, 0.5],
                    [0.4, 0.5, 0.6],
                ]
            ],
            dtype=torch.float32,
        )
        phys_adj = torch.zeros(1, 5, 5)
        for j in range(1, 5):
            phys_adj[0, 0, j] = 1.0
            phys_adj[0, j, 0] = 1.0
        node_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
        out = encoder(node_feats, phys_adj, node_mask)
        self.assertEqual(tuple(out.shape), (1, 16))

        scores = encoder.affinity_score(node_feats[0, 1:])
        self.assertEqual(scores.shape, (4,))


class D3QNNetworkTests(unittest.TestCase):
    def _sample_packed(self, is_completed=False):
        task_state = np.zeros(TASK_STATE_DIM, dtype=np.float32)
        task_state[1] = 0.5
        task_state[2] = 0.3
        task_state[TASK_IDX_IS_COMPLETED] = float(is_completed)
        dest_state = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        node_feats = np.random.rand(5, 3).astype(np.float32)
        phys_adj = np.eye(5, dtype=np.float32)
        phys_adj[0, 1] = phys_adj[1, 0] = 1.0
        node_mask = np.array([1, 1, 0, 0, 0], dtype=np.float32)
        return pack_dgr_state(task_state, dest_state, node_feats, phys_adj, node_mask)

    def test_forward_with_action_mask(self):
        net = DGRD3QNNetwork(dgr_hidden_dim=16, q_hidden_dim=32, action_dim=5)
        packed = self._sample_packed(is_completed=False)
        task_state, dest_state, node_feats, phys_adj, node_mask = unpack_dgr_state(packed)
        action_mask = build_action_mask_from_layout(node_mask, is_completed=False)
        q = net.forward_from_components(
            torch.tensor(task_state).unsqueeze(0),
            torch.tensor(dest_state).unsqueeze(0),
            torch.tensor(node_feats).unsqueeze(0),
            torch.tensor(phys_adj).unsqueeze(0),
            torch.tensor(node_mask).unsqueeze(0),
            torch.tensor(action_mask).unsqueeze(0),
        )
        self.assertEqual(tuple(q.shape), (1, 5))
        masked = q[0].detach().numpy()
        self.assertTrue(np.isneginf(masked[2]))
        self.assertTrue(np.isfinite(masked[4]))

    def test_auto_packed_dim(self):
        self.assertEqual(compute_dgr_packed_dim(), len(self._sample_packed()))


class D3QNAgentTests(unittest.TestCase):
    def test_double_update_runs(self):
        agent = D3QN_Agent(buffer_length=1000, batch_size=4, gamma=0.99, device=torch.device('cpu'))
        task_state = np.zeros(TASK_STATE_DIM, dtype=np.float32)
        task_state[1] = 0.5
        task_state[2] = 0.3
        task_state[11] = 0.25
        task_state[14] = 0.5
        dest_state = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        node_feats = np.full((5, 3), 0.5, dtype=np.float32)
        phys_adj = np.eye(5, dtype=np.float32)
        phys_adj[0, 1] = phys_adj[1, 0] = 1.0
        node_mask = np.array([1, 1, 0, 0, 0], dtype=np.float32)
        packed = pack_dgr_state(task_state, dest_state, node_feats, phys_adj, node_mask)
        action_mask = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        next_action_mask = np.array([1, 0, 0, 0, 0], dtype=np.float32)
        experiences = [[packed, action_mask, 0, 1.0, packed, next_action_mask, 0] for _ in range(8)]
        agent.update(experiences)
        agent.target_update()


class D3QNTargetNaNTests(unittest.TestCase):
    def test_terminal_and_empty_mask_no_nan(self):
        dim = compute_dgr_packed_dim()
        packed = np.zeros(dim, dtype=np.float32)
        zero_mask = np.zeros(5, dtype=np.float32)
        batch = [
            [packed, zero_mask, 0, -1.0, packed, zero_mask, 1.0],
            [packed, zero_mask, 0, 0.5, packed, zero_mask, 0.0],
        ] * 8

        for agent in (
            D3QN_Agent(buffer_length=64, batch_size=16, gamma=0.99, device=torch.device("cpu")),
            FlatD3QN_Agent(buffer_length=64, batch_size=16, gamma=0.99, device=torch.device("cpu")),
        ):
            agent.update(batch)
            _, _, _, reward, next_state, next_action_mask, done = parse_experience_batch(batch)
            expected = compute_ddqn_bootstrap_target(
                agent.online_net,
                agent.target_net,
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(next_action_mask, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32),
                agent.gamma,
            )
            self.assertFalse(torch.isnan(expected).any())


if __name__ == '__main__':
    unittest.main()
