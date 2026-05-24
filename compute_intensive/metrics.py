"""Collect and persist compute_intensive experiment metrics."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from baselines.modes import is_heuristic_mode

# failure_rate in legacy JSON was an alias of random_edges_del (edge count), NOT a percentage.


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _track_episode_rewards(raw_mode_name: str) -> bool:
    rl_modes = {"DQN", "D3QN", "DGR_D3QN", "Pure_DQN", "New_DQN", "Weak_DQN"}
    return raw_mode_name in rl_modes


def aggregate_statics(statics_datas: Dict[str, float]) -> Dict[str, Optional[float]]:
    total = statics_datas.get("Total", 0)
    reached = statics_datas.get("Reached_0", 0) + statics_datas.get("Reached_1", 0)
    lost = (
        statics_datas.get("Lost_relay_0", 0)
        + statics_datas.get("Lost_relay_1", 0)
        + statics_datas.get("Lost_upload", 0)
    )
    finished = lost + reached
    total_delay = statics_datas.get("Total_delay_0", 0) + statics_datas.get("Total_delay_1", 0)
    total_hops = statics_datas.get("Total_hops_0", 0) + statics_datas.get("Total_hops_1", 0)
    return {
        "average_delay": _safe_div(total_delay, reached),
        "packet_loss_rate": _safe_div(lost, finished),
        "average_hops": _safe_div(total_hops, reached),
        "completed_task_count": int(reached),
        "dropped_task_count": int(lost),
        "total_task_count": int(total),
    }


SCENARIO_NAME = "compute_intensive"


def build_result_record(
    *,
    method_name: str,
    raw_mode_name: str,
    experiment_group: str,
    load_level: Optional[float],
    seed: int,
    statics_datas: Dict[str, float],
    hop_count_list: List[float],
    episode_reward_list: List[float],
    scenario: str = SCENARIO_NAME,
    scheme: Optional[str] = None,
    algorithm: Optional[str] = None,
    sweep_type: str = "single",
    task_rate_target: Optional[float] = None,
    actual_task_rate: Optional[float] = None,
    computing_ability: Optional[float] = None,
    computing_ability_gflops: Optional[float] = None,
    random_edges_del: Optional[int] = None,
    removed_edges_count: Optional[int] = None,
    packet_frequency: Optional[float] = None,
    simulation_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    metrics = aggregate_statics(statics_datas)
    track_rewards = _track_episode_rewards(raw_mode_name)
    rewards = [r for r in episode_reward_list if r is not None] if track_rewards else []
    return {
        "scenario": scenario,
        "method_name": method_name,
        "scheme": scheme,
        "algorithm": algorithm,
        "raw_mode_name": raw_mode_name,
        "experiment_group": experiment_group,
        "sweep_type": sweep_type,
        "load_level": load_level,
        "task_rate_target": task_rate_target,
        "actual_task_rate": actual_task_rate,
        "computing_ability": computing_ability,
        "computing_ability_gflops": computing_ability_gflops,
        "random_edges_del": random_edges_del,
        "removed_edges_count": removed_edges_count,
        "failure_rate": None,
        "packet_frequency": packet_frequency,
        "simulation_duration_sec": simulation_duration_sec,
        "seed": seed,
        "status": "success",
        "average_delay": metrics["average_delay"],
        "packet_loss_rate": metrics["packet_loss_rate"],
        "average_hops": metrics["average_hops"],
        "average_episode_reward": _safe_div(sum(rewards), len(rewards)) if rewards else None,
        "completed_task_count": metrics["completed_task_count"],
        "dropped_task_count": metrics["dropped_task_count"],
        "total_task_count": metrics["total_task_count"],
        "reached_after_computed_0": statics_datas.get("Reached_after_computed_0", 0),
        "hop_count_list": hop_count_list,
        "episode_reward_list": episode_reward_list if track_rewards else [],
    }


def build_failed_record(
    *,
    method_name: str,
    raw_mode_name: str,
    experiment_group: str,
    load_level: Optional[float],
    seed: int,
    error_message: str,
    scenario: str = SCENARIO_NAME,
    scheme: Optional[str] = None,
    algorithm: Optional[str] = None,
    sweep_type: str = "single",
    task_rate_target: Optional[float] = None,
    computing_ability: Optional[float] = None,
    computing_ability_gflops: Optional[float] = None,
    random_edges_del: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "method_name": method_name,
        "scheme": scheme,
        "algorithm": algorithm,
        "raw_mode_name": raw_mode_name,
        "experiment_group": experiment_group,
        "sweep_type": sweep_type,
        "load_level": load_level,
        "task_rate_target": task_rate_target,
        "actual_task_rate": None,
        "computing_ability": computing_ability,
        "computing_ability_gflops": computing_ability_gflops,
        "random_edges_del": random_edges_del,
        "removed_edges_count": None,
        "failure_rate": None,
        "seed": seed,
        "status": "failed",
        "error_message": error_message,
        "average_delay": None,
        "packet_loss_rate": None,
        "average_hops": None,
        "average_episode_reward": None,
        "completed_task_count": 0,
        "dropped_task_count": 0,
        "total_task_count": 0,
        "reached_after_computed_0": 0,
        "hop_count_list": [],
        "episode_reward_list": [],
    }


def save_result(record: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    parts = [record["method_name"]]
    if record.get("task_rate_target") is not None:
        parts.append(f"rate{record['task_rate_target']}")
    elif record.get("load_level") is not None:
        parts.append(f"load{record['load_level']}")
    if record.get("computing_ability_gflops") is not None:
        parts.append(f"gflops{record['computing_ability_gflops']}")
    if record.get("random_edges_del") is not None:
        parts.append(f"edges{record['random_edges_del']}")
    filename = f"{'_'.join(str(p) for p in parts)}_seed{record['seed']}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, ensure_ascii=False)
    return path
