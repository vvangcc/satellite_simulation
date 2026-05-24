"""Chapter 3 baseline algorithms."""

from baselines.ga_icm import GABaseline
from baselines.modes import (
    allows_onboard_compute,
    is_ga_mode,
    is_heuristic_mode,
    resolve_simulator_mode,
)

__all__ = [
    "GABaseline",
    "allows_onboard_compute",
    "is_ga_mode",
    "is_heuristic_mode",
    "resolve_simulator_mode",
]
