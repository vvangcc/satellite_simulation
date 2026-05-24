"""mixed_tasks experiment metrics aggregation."""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def aggregate_mixed_tasks_metrics(statics: Dict[str, Any], episode_reward_list) -> Dict[str, float]:
    d_total = statics.get("D_total", 0)
    d_on_time = statics.get("D_on_time", 0)
    c_total = statics.get("C_total", 0)
    c_soft = statics.get("C_soft_deadline", 0)
    c_completed = statics.get("C_completed", 0)
    c_delay_sum = statics.get("C_total_delay", 0.0)
    return {
        "on_time_rate_D": d_on_time / d_total if d_total else 0.0,
        "soft_deadline_rate_C": c_soft / c_total if c_total else 0.0,
        "average_delay_C": c_delay_sum / c_completed if c_completed else 0.0,
        "average_episode_reward": (
            sum(episode_reward_list) / len(episode_reward_list) if episode_reward_list else 0.0
        ),
    }


def build_result_record(
    method_name: str,
    raw_mode_name: str,
    seed: int,
    load_level: float,
    statics: Dict[str, Any],
    episode_reward_list,
    scenario: str = "mixed_tasks",
) -> Dict[str, Any]:
    agg = aggregate_mixed_tasks_metrics(statics, episode_reward_list)
    return {
        "scenario": scenario,
        "method_name": method_name,
        "raw_mode_name": raw_mode_name,
        "seed": seed,
        "load_level": load_level,
        "delay_sensitive_total_count": statics.get("D_total", 0),
        "delay_sensitive_completed_count": statics.get("D_completed", 0),
        "delay_sensitive_on_time_count": statics.get("D_on_time", 0),
        "compute_intensive_total_count": statics.get("C_total", 0),
        "compute_intensive_completed_count": statics.get("C_completed", 0),
        "compute_intensive_soft_deadline_count": statics.get("C_soft_deadline", 0),
        "on_time_rate_D": agg["on_time_rate_D"],
        "soft_deadline_rate_C": agg["soft_deadline_rate_C"],
        "average_delay_C": agg["average_delay_C"],
        "average_episode_reward": agg["average_episode_reward"],
        "episode_reward_list": list(episode_reward_list),
    }


def save_result(record: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{record['method_name']}_load{record['load_level']}_seed{record['seed']}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)
    return path
