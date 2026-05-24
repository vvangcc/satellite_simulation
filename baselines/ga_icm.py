"""
GA baseline (Genetic-Algorithm-style centralized optimizer).

GA baseline is implemented based on the original ICM centralized algorithm.
The legacy ICM logic lives in ``tradition_routing()`` inside
``SatelliteNetworkSimulator_Computing.py``: it uses the global network view
(``neighbor_graph`` subgraph of the propagator graph) to jointly pick a compute
node and forwarding path while accounting for propagation, transmission,
compute, queue, and congestion penalties.
"""
from __future__ import annotations

from baselines.modes import is_ga_mode


class GABaseline:
    """
    Wrapper documenting the GA / ICM centralized routing interface.

    The actual per-packet decision is still executed by ``tradition_routing`` on
    the satellite instance; this class provides multi-stage-aware helpers.
    """

    @staticmethod
    def stage_demand(task) -> int:
        return task.current_stage_demand

    @staticmethod
    def stage_output_size(task) -> int:
        return task.current_stage_output_size

    @staticmethod
    def sso_total_demand(task) -> int:
        """SSO: all remaining compute demand must finish on one satellite."""
        return task.remaining_demand

    @staticmethod
    def sso_final_output_size(task) -> int:
        return task.final_output_size


__all__ = ["GABaseline", "is_ga_mode"]
