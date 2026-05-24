import unittest
from types import SimpleNamespace

import numpy as np

from models.action_mask import (
    ACTION_DIM,
    COMPUTE_ACTION,
    build_action_mask,
    masked_argmax,
    random_valid_action,
)
from task_model import TaskInfo


class MockGraph:
    def __init__(self, edges):
        self._edges = set(edges)

    def has_edge(self, u, v):
        return (u, v) in self._edges


class ActionMaskTests(unittest.TestCase):
    def _make_context(self, *, neighbors, completed=False, neighbor_memory_ok=True, local_memory_ok=True):
        task = TaskInfo(
            task_type=0,
            current_size_bytes=100,
            demand_seq_flops=[1000],
            output_size_seq_bytes=[50],
            total_stages=1,
            is_completed=completed,
        )
        packet = SimpleNamespace(task=task, size=100)
        neighbor_sats = {}
        for name in neighbors:
            neighbor_sats[name] = SimpleNamespace(
                active=True,
                current_memory_occupy=0 if neighbor_memory_ok else 999,
                memory=1000,
            )
        satellite = SimpleNamespace(
            name="sat_a",
            active=True,
            neighbors=list(neighbors),
            memory=1000,
            current_memory_occupy=0 if local_memory_ok else 999,
            computing_remain=0.0,
            computing_ability=1.0,
            is_computing=False,
            env=SimpleNamespace(now=0.0),
            last_computing_time=0.0,
            last_heartbeat={},
            heartbeat_timeout=0.25,
            propagator=SimpleNamespace(satellites=neighbor_sats),
        )
        edges = {(satellite.name, n) for n in neighbors}
        graph = MockGraph(edges)
        return packet, satellite, graph

    def test_padding_neighbors_are_masked(self):
        packet, satellite, graph = self._make_context(neighbors=["n1"])
        mask = build_action_mask(packet, satellite, graph)
        self.assertEqual(mask.shape, (ACTION_DIM,))
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)
        self.assertEqual(mask[2], 0.0)
        self.assertEqual(mask[3], 0.0)

    def test_completed_task_masks_compute_action(self):
        packet, satellite, graph = self._make_context(neighbors=["n1"], completed=True)
        mask = build_action_mask(packet, satellite, graph)
        self.assertEqual(mask[COMPUTE_ACTION], 0.0)
        self.assertEqual(mask[0], 1.0)

    def test_insufficient_neighbor_memory_masks_forward(self):
        packet, satellite, graph = self._make_context(neighbors=["n1"], neighbor_memory_ok=False)
        mask = build_action_mask(packet, satellite, graph)
        self.assertEqual(mask[0], 0.0)
        self.assertEqual(mask[COMPUTE_ACTION], 1.0)

    def test_random_valid_action_never_picks_masked_slot(self):
        mask = np.array([1, 0, 0, 0, 1], dtype=np.float32)
        rng = np.random.default_rng(0)
        for _ in range(100):
            action = random_valid_action(mask, rng=rng)
            self.assertIn(action, (0, 4))

    def test_masked_argmax_respects_mask(self):
        q_values = np.array([10, 9, 8, 7, 6], dtype=np.float32)
        mask = np.array([0, 1, 0, 0, 0], dtype=np.float32)
        self.assertEqual(masked_argmax(q_values, mask), 1)


if __name__ == "__main__":
    unittest.main()
