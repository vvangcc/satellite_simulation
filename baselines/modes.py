"""Mode aliases and helpers for Chapter 3 baselines (no simulator imports)."""
from __future__ import annotations

HEURISTIC_MODES = frozenset({"Tradition", "Ground", "GA", "SSO", "GO"})


def is_heuristic_mode(mode: str) -> bool:
    return mode in HEURISTIC_MODES


def allows_onboard_compute(mode: str) -> bool:
    return mode not in {"Ground", "GO"}


def is_ga_mode(mode: str) -> bool:
    """GA is the paper-facing name for the ICM-based centralized baseline."""
    return mode in {"GA", "Tradition"}


def resolve_simulator_mode(mode: str) -> str:
    """Map paper-facing mode aliases to internal simulator mode strings."""
    if mode == "GO":
        return "Ground"
    if mode == "GA":
        return "Tradition"
    return mode
