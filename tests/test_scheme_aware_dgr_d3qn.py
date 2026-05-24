import unittest
from types import SimpleNamespace

import numpy as np

from models.action_mask import COMPUTE_ACTION
from policies.scheme_aware_dgr_d3qn import (
    bind_sso_compute_node,
    build_forward_to_target_mask,
    build_scheme_action_mask,
    resolve_decision_target,
)
from task_model import TaskInfo


class MockGraph:
    def __init__(self, edges):
        self._edges = set(edges)

    def has_edge(self, u, v):
        return (u, v) in self._edges


def _satellite(name, neighbors, *, routing_tables=None, neighbor_hops=None):
    neighbor_sats = {
        n: SimpleNamespace(active=True, current_memory_occupy=0, memory=1000)
        for n in neighbors
    }
    return SimpleNamespace(
        name=name,
        active=True,
        neighbors=list(neighbors),
        memory=1000,
        current_memory_occupy=0,
        computing_remain=0.0,
        computing_ability=1.0,
        is_computing=False,
        env=SimpleNamespace(now=0.0),
        last_computing_time=0.0,
        last_heartbeat={},
        heartbeat_timeout=0.25,
        max_hop=10,
        routing_tables=routing_tables or {},
        neighbor_hops=neighbor_hops or {n: {} for n in neighbors},
        propagator=SimpleNamespace(satellites=neighbor_sats),
    )


class SchemeActionMaskTests(unittest.TestCase):
    def _task(self, *, stages=1, fixed_compute_node=None, completed=False, scheme="SSO"):
        demand = [1000] * stages
        outputs = [50] * stages
        task = TaskInfo(
            task_type=0,
            current_size_bytes=100,
            demand_seq_flops=demand,
            output_size_seq_bytes=outputs,
            total_stages=stages,
            stage_idx=0 if not completed else stages,
            is_completed=completed,
            fixed_compute_node=fixed_compute_node,
            scheme=scheme,
            destination="ground_sat",
        )
        return task

    def _ctx(self, task, satellite, graph):
        return SimpleNamespace(task=task, size=100, destination=task.destination), satellite, graph

    def test_go_masks_compute(self):
        task = self._task(scheme="GO")
        sat = _satellite("sat_a", ["n1"])
        packet, satellite, graph = self._ctx(task, sat, MockGraph({("sat_a", "n1")}))
        mask = build_scheme_action_mask(packet, satellite, graph, "GO")
        self.assertEqual(mask[COMPUTE_ACTION], 0.0)
        self.assertEqual(mask[0], 1.0)

    def test_sso_unbound_allows_compute_and_forward(self):
        task = self._task(fixed_compute_node=None)
        sat = _satellite("sat_a", ["n1"])
        packet, satellite, graph = self._ctx(task, sat, MockGraph({("sat_a", "n1")}))
        mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
        self.assertEqual(mask[COMPUTE_ACTION], 1.0)
        self.assertEqual(mask[0], 1.0)

    def test_sso_on_fixed_node_only_compute(self):
        task = self._task(fixed_compute_node="sat_a")
        sat = _satellite("sat_a", ["n1"])
        packet, satellite, graph = self._ctx(task, sat, MockGraph({("sat_a", "n1")}))
        mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
        self.assertEqual(mask[COMPUTE_ACTION], 1.0)
        self.assertEqual(mask[0], 0.0)

    def test_sso_off_fixed_node_no_compute_forward_toward_fixed(self):
        task = self._task(fixed_compute_node="sat_b")
        sat = _satellite(
            "sat_a",
            ["n1", "n2"],
            routing_tables={"sat_b": (None, 3)},
            neighbor_hops={
                "n1": {"sat_b": 1},
                "n2": {"sat_b": 4},
            },
        )
        graph = MockGraph({("sat_a", "n1"), ("sat_a", "n2")})
        packet, satellite, graph = self._ctx(task, sat, graph)
        mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
        self.assertEqual(mask[COMPUTE_ACTION], 0.0)
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)

    def test_sso_completed_masks_compute_allows_forward_to_destination(self):
        task = self._task(fixed_compute_node="sat_a", completed=True)
        sat = _satellite(
            "sat_a",
            ["n1"],
            routing_tables={"ground_sat": (None, 2)},
            neighbor_hops={"n1": {"ground_sat": 1}},
        )
        packet, satellite, graph = self._ctx(task, sat, MockGraph({("sat_a", "n1")}))
        mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
        self.assertEqual(mask[COMPUTE_ACTION], 0.0)
        self.assertEqual(mask[0], 1.0)

    def test_three_stage_sso_mask_progression(self):
        task = self._task(stages=3, fixed_compute_node="sat_a")
        bind_sso_compute_node(task, "sat_a", "SSO")
        sat = _satellite("sat_a", ["n1"], routing_tables={"ground_sat": (None, 5)})
        graph = MockGraph({("sat_a", "n1")})

        for stage in range(3):
            self.assertFalse(task.is_completed)
            packet, satellite, graph = self._ctx(task, sat, graph)
            mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
            self.assertEqual(mask[COMPUTE_ACTION], 1.0, msg=f"stage {stage} should allow compute")
            self.assertEqual(mask[0], 0.0, msg=f"stage {stage} should block forward")
            task.complete_current_stage()

        self.assertTrue(task.is_completed)
        packet, satellite, graph = self._ctx(task, sat, graph)
        mask = build_scheme_action_mask(packet, satellite, graph, "SSO")
        self.assertEqual(mask[COMPUTE_ACTION], 0.0)
        self.assertEqual(mask[0], 1.0)

    def test_resolve_decision_target_sso(self):
        task = self._task(fixed_compute_node="sat_b")
        packet = SimpleNamespace(destination="ground_sat", task=task)
        self.assertEqual(resolve_decision_target(task, packet, "SSO"), "sat_b")
        task.is_completed = True
        self.assertEqual(resolve_decision_target(task, packet, "SSO"), "ground_sat")

    def test_bind_sso_compute_node(self):
        task = self._task(stages=1)
        bind_sso_compute_node(task, "sat_x", "SSO")
        self.assertEqual(task.fixed_compute_node, "sat_x")
        bind_sso_compute_node(task, "sat_y", "SSO")
        self.assertEqual(task.fixed_compute_node, "sat_x")

    def test_forward_to_target_prefers_closer_neighbor(self):
        task = self._task()
        sat = _satellite(
            "sat_a",
            ["n1", "n2"],
            routing_tables={"sat_b": (None, 3)},
            neighbor_hops={"n1": {"sat_b": 1}, "n2": {"sat_b": 4}},
        )
        packet = SimpleNamespace(task=task, size=100)
        graph = MockGraph({("sat_a", "n1"), ("sat_a", "n2")})
        mask = build_forward_to_target_mask(packet, sat, graph, "sat_b")
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)


if __name__ == "__main__":
    unittest.main()
