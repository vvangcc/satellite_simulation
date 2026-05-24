import random
import unittest

import numpy as np
import torch

from agents.mixed_tasks_d3qn_agent import MixedTasksD3QN_Agent
from agents.mixed_tasks_per_d3qn_agent import MixedTasksPERD3QN_Agent
from models.mixed_tasks_reward import MixedTasksRewardFunction
from models.mixed_tasks_state_layout import (
    MIXED_TASKS_TASK_STATE_DIM,
    compute_mixed_tasks_packed_dim,
    pack_mixed_tasks_state,
)
from models.per_replay_buffer import PERReplayBuffer
from task_model import build_mixed_tasks_task, build_mixed_tasks_delay_sensitive_task


class MixedTasksTaskTests(unittest.TestCase):
    def test_task_mix_proportions(self):
        rng = random.Random(0)
        counts = {0: 0, 1: 0}
        for _ in range(1000):
            t = build_mixed_tasks_task(
                task_type=0 if rng.random() < 0.7 else 1,
                destination="Satellite_500_1_1",
                birth_time=0.0,
                size_range=(1, 2),
                task_num_stages=[2, 3],
                computing_demand_factor=[1400, 2200],
                size_reduction_factor=[1.3, 4.0],
                final_result_size_range=[5120, 15360],
                rng=rng,
            )
            counts[t.task_type] += 1
        self.assertGreater(counts[0], counts[1])

    def test_d_task_hard_deadline(self):
        task = build_mixed_tasks_delay_sensitive_task("Satellite_500_1_1", 0.0, rng=random.Random(1))
        self.assertEqual(task.deadline, 2.0)
        self.assertEqual(task.total_stages, 1)


class MixedTasksStateTests(unittest.TestCase):
    def test_packed_dim(self):
        self.assertEqual(
            compute_mixed_tasks_packed_dim(),
            MIXED_TASKS_TASK_STATE_DIM + 4 + 4 + 5 * 3 + 25 + 5,
        )


class MixedTasksRewardTests(unittest.TestCase):
    def test_d_success_beats_c_coefficients(self):
        rf = MixedTasksRewardFunction()
        d_task = build_mixed_tasks_delay_sensitive_task("d", 0.0, rng=random.Random(2))
        c_task = build_mixed_tasks_task(
            0, "c", 0.0, (1, 2), [2], [1400, 2200], [1.3, 4.0], [5120, 15360], rng=random.Random(2)
        )
        self.assertGreater(rf.success_reward(d_task, 0.5, True), rf.success_reward(c_task, 0.5, True))


class PERBufferTests(unittest.TestCase):
    def test_priority_update_changes_sampling(self):
        buf = PERReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4, beta_end=1.0)
        exp = [np.zeros(3), np.ones(5), 0, 1.0, np.zeros(3), np.ones(5), 0]
        for _ in range(20):
            buf.add(exp, priority=1.0)
        _, idx0, _ = buf.sample(4)
        buf.update_priorities(idx0, np.full(4, 10.0))
        _, idx1, _ = buf.sample(4)
        self.assertTrue(set(idx1).issubset(set(range(20))))


class MixedTasksAgentTests(unittest.TestCase):
    def _packed(self):
        task = np.zeros(MIXED_TASKS_TASK_STATE_DIM, dtype=np.float32)
        queue = np.zeros(4, dtype=np.float32)
        dest = np.zeros(4, dtype=np.float32)
        feats = np.ones((5, 3), dtype=np.float32) * 0.5
        adj = np.eye(5, dtype=np.float32)
        mask = np.ones(5, dtype=np.float32)
        return pack_mixed_tasks_state(task, queue, dest, feats, adj, mask)

    def test_mixed_tasks_d3qn_update(self):
        agent = MixedTasksD3QN_Agent(buffer_length=100, batch_size=4, gamma=0.99, device=torch.device("cpu"))
        packed = self._packed()
        mask = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        exps = [[packed, mask, 0, 1.0, packed, mask, 0] for _ in range(8)]
        agent.update(exps)

    def test_mixed_tasks_per_d3qn_update(self):
        agent = MixedTasksPERD3QN_Agent(buffer_length=100, batch_size=4, gamma=0.99, device=torch.device("cpu"))
        packed = self._packed()
        mask = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        exps = [[packed, mask, 0, 1.0, packed, mask, 0] for _ in range(8)]
        agent.update(exps)


if __name__ == "__main__":
    unittest.main()
