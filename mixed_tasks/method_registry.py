"""Chapter 4 method registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MethodSpec:
    method_name: str
    raw_mode_name: str
    config_file: str
    train_config: str
    description: str = ""


METHODS: Dict[str, MethodSpec] = {
    "PPO": MethodSpec(
        method_name="PPO",
        raw_mode_name="CH4_PPO",
        config_file="algo_ppo.yaml",
        train_config="train_ppo.yaml",
        description="PPO baseline for multi-type tasks",
    ),
    "D3QN": MethodSpec(
        method_name="D3QN",
        raw_mode_name="CH4_D3QN",
        config_file="algo_d3qn.yaml",
        train_config="train_d3qn.yaml",
        description="Dueling Double DQN without PER",
    ),
    "PER-D3QN": MethodSpec(
        method_name="PER-D3QN",
        raw_mode_name="CH4_PER_D3QN",
        config_file="algo_per_d3qn.yaml",
        train_config="train_per_d3qn.yaml",
        description="PER + Dueling Double DQN (main algorithm)",
    ),
}


def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f"Unknown mixed_tasks method: {name}. Choose from {list(METHODS)}")
    return METHODS[name]


def all_methods() -> List[MethodSpec]:
    return list(METHODS.values())
