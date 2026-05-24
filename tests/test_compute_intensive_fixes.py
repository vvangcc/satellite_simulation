import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.dgr_encoder import DGRResourceEncoder
from models.state_layout import (
    TASK_STATE_DIM,
    build_task_state_vector,
    compute_dgr_packed_dim,
    compute_dqn_flat_state_dim,
)
from task_model import TaskInfo


class TaskStateLayoutTests(unittest.TestCase):
    def test_task_state_dim(self):
        self.assertEqual(TASK_STATE_DIM, 15)

    def test_build_task_state_vector_shape(self):
        task = TaskInfo(
            task_type=0,
            current_size_bytes=1000,
            demand_seq_flops=[100, 200, 300],
            output_size_seq_bytes=[800, 600, 400],
            total_stages=3,
            stage_idx=1,
        )
        vec = build_task_state_vector(task, max_size=10000, computing_ability=1000, max_hop=20, hops=3)
        self.assertEqual(vec.shape[0], TASK_STATE_DIM)
        self.assertAlmostEqual(vec[11], 1 / 4)
        self.assertAlmostEqual(vec[12], 3 / 4)

    def test_auto_state_dims(self):
        self.assertEqual(compute_dqn_flat_state_dim(), 42)
        self.assertEqual(compute_dgr_packed_dim(), 64)


class DGREncoderWeightTests(unittest.TestCase):
    def _forward(self, top_k, lambda_mix):
        encoder = DGRResourceEncoder(hidden_dim=16, num_layers=2, top_k=top_k, lambda_mix=lambda_mix)
        node_feats = torch.tensor(
            [
                [
                    [0.9, 0.1, 0.1],
                    [0.8, 0.2, 0.1],
                    [0.2, 0.8, 0.7],
                    [0.3, 0.7, 0.6],
                    [0.4, 0.6, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        phys_adj = torch.zeros(1, 5, 5)
        for j in range(1, 5):
            phys_adj[0, 0, j] = 1.0
            phys_adj[0, j, 0] = 1.0
        node_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
        return encoder(node_feats, phys_adj, node_mask)

    def test_top_k_changes_output(self):
        out_k1 = self._forward(top_k=1, lambda_mix=0.6)
        out_k3 = self._forward(top_k=3, lambda_mix=0.6)
        self.assertFalse(torch.allclose(out_k1, out_k3))

    def test_lambda_mix_changes_output(self):
        out_low = self._forward(top_k=2, lambda_mix=0.95)
        out_high = self._forward(top_k=2, lambda_mix=0.05)
        self.assertFalse(torch.allclose(out_low, out_high))


class RLExperienceLogicTests(unittest.TestCase):
    def test_pending_then_flush_produces_different_states(self):
        from SatelliteNetworkSimulator_Computing import flush_rl_transition, finalize_rl_transition

        task = TaskInfo(
            task_type=0,
            current_size_bytes=100,
            demand_seq_flops=[100],
            output_size_seq_bytes=[50],
            total_stages=1,
        )
        task.last_state = np.array([1.0, 2.0], dtype=np.float32)
        task.last_action = 0
        task.last_action_mask = np.ones(5, dtype=np.float32)
        task.set_pending_transition(0.5, 0)

        packet = SimpleNamespace(task=task, size=100)
        propagator = SimpleNamespace(experiences=[], graph=SimpleNamespace(has_edge=lambda *_: True), satellites={})
        satellite = SimpleNamespace(
            mode="DGR_D3QN", name="A", neighbors=[], memory=1000, current_memory_occupy=100,
            computing_remain=0, computing_ability=1, is_computing=False, active=True,
            env=SimpleNamespace(now=0), last_computing_time=0, propagator=propagator,
            heartbeat_timeout=9999, last_heartbeat={},
        )

        next_state = np.array([3.0, 4.0], dtype=np.float32)
        flush_rl_transition(propagator, satellite, packet, next_state)
        self.assertEqual(len(propagator.experiences), 1)
        s, _, _, r, ns, _, done = propagator.experiences[0]
        self.assertFalse(np.allclose(s, ns))
        self.assertEqual(r, 0.5)
        self.assertEqual(done, 0)

        task.set_pending_transition(-1.0, 1)
        finalize_rl_transition(propagator, satellite, packet, np.array([5.0, 6.0]), -2.0, 1)
        self.assertEqual(len(propagator.experiences), 2)
        self.assertEqual(propagator.experiences[1][3], -2.0)


if __name__ == "__main__":
    unittest.main()
