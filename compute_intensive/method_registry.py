"""Paper-facing method names mapped to simulator configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    raw_mode_name: str
    experiment_group: str
    config_file: str
    scheme: Optional[str] = None
    algorithm: Optional[str] = None
    description: str = ""


OFFLOADING_METHODS: Dict[str, MethodSpec] = {
    "GO": MethodSpec(
        method_name="GO",
        raw_mode_name="DGR_D3QN",
        experiment_group="offloading",
        config_file="offloading_go.yaml",
        scheme="GO",
        algorithm="DGR-D3QN",
        description="Ground offloading via DGR-D3QN with compute action masked.",
    ),
    "SSO": MethodSpec(
        method_name="SSO",
        raw_mode_name="DGR_D3QN",
        experiment_group="offloading",
        config_file="offloading_sso.yaml",
        scheme="SSO",
        algorithm="DGR-D3QN",
        description="Single-satellite offloading via DGR-D3QN with fixed_compute_node binding.",
    ),
    "MSCO": MethodSpec(
        method_name="MSCO",
        raw_mode_name="DGR_D3QN",
        experiment_group="offloading",
        config_file="offloading_msco.yaml",
        scheme="MSCO",
        algorithm="DGR-D3QN",
        description="Multi-satellite collaborative offloading via unconstrained DGR-D3QN.",
    ),
}

ALGORITHM_METHODS: Dict[str, MethodSpec] = {
    "GA": MethodSpec(
        method_name="GA",
        raw_mode_name="ICM",
        experiment_group="algorithm",
        config_file="algo_ga.yaml",
        scheme="MSCO",
        algorithm="GA_from_ICM",
        description="GA baseline is implemented based on the original ICM centralized algorithm.",
    ),
    "DQN": MethodSpec(
        method_name="DQN",
        raw_mode_name="DQN",
        experiment_group="algorithm",
        config_file="algo_dqn.yaml",
        scheme="MSCO",
        algorithm="DQN",
        description="Flat DQN baseline without DGR / dueling / double / shuffle.",
    ),
    "DGR-D3QN": MethodSpec(
        method_name="DGR-D3QN",
        raw_mode_name="DGR_D3QN",
        experiment_group="algorithm",
        config_file="algo_dgr_d3qn.yaml",
        scheme="MSCO",
        algorithm="DGR-D3QN",
        description="DGR encoder + Dueling + Double DQN + action mask.",
    ),
}

ABLATION_METHODS: Dict[str, MethodSpec] = {
    "D3QN": MethodSpec(
        method_name="D3QN",
        raw_mode_name="D3QN",
        experiment_group="ablation",
        config_file="algo_d3qn.yaml",
        scheme="MSCO",
        algorithm="D3QN",
        description="Dueling + Double DQN on flattened task/graph state without DGR encoder.",
    ),
    "DGR-D3QN": MethodSpec(
        method_name="DGR-D3QN",
        raw_mode_name="DGR_D3QN",
        experiment_group="ablation",
        config_file="algo_dgr_d3qn_ablation.yaml",
        scheme="MSCO",
        algorithm="DGR-D3QN",
        description="DGR encoder + Dueling + Double DQN for ablation comparison.",
    ),
}

TRAIN_METHODS: Dict[str, MethodSpec] = {
    "DQN": MethodSpec(
        method_name="DQN",
        raw_mode_name="DQN",
        experiment_group="algorithm",
        config_file="train_dqn.yaml",
        description="Train flat DQN.",
    ),
    "D3QN": MethodSpec(
        method_name="D3QN",
        raw_mode_name="D3QN",
        experiment_group="ablation",
        config_file="train_d3qn.yaml",
        description="Train flat D3QN (dueling + double, no DGR).",
    ),
    "DGR-D3QN": MethodSpec(
        method_name="DGR-D3QN",
        raw_mode_name="DGR_D3QN",
        experiment_group="algorithm",
        config_file="train_dgr_d3qn.yaml",
        description="Train DGR-D3QN.",
    ),
}

ALL_METHODS = {**OFFLOADING_METHODS, **ALGORITHM_METHODS, **ABLATION_METHODS}


def get_methods_for_group(group: str):
    if group == "offloading":
        return list(OFFLOADING_METHODS.values())
    if group == "algorithm":
        return list(ALGORITHM_METHODS.values())
    if group == "ablation":
        return list(ABLATION_METHODS.values())
    if group == "all":
        return list({**OFFLOADING_METHODS, **ALGORITHM_METHODS, **ABLATION_METHODS}.values())
    raise ValueError(f"Unknown experiment group: {group}")


def get_train_method(method_name: str) -> MethodSpec:
    if method_name not in TRAIN_METHODS:
        raise ValueError(
            f"Unknown train method '{method_name}'. Choose from: {', '.join(TRAIN_METHODS)}"
        )
    return TRAIN_METHODS[method_name]
