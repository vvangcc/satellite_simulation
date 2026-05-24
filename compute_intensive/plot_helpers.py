"""Shared helpers for compute_intensive result plotting."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

DEFAULT_COMPUTE_SWEEP_TASK_RATE = 60
DEFAULT_TRAIN_TASK_RATE = 40
DEFAULT_COMPUTE_LEVELS_GFLOPS = [40, 60, 80, 100, 120]

TRAINING_LOG_FILES = {
    "DQN": "train_dqn.txt",
    "D3QN": "train_d3qn.txt",
    "DGR-D3QN": "train_dgr_d3qn.txt",
}

REWARD_LINE = re.compile(r"Average ending reward:\s*([-\d.eE+]+|None)")


def x_key_task_rate(rec: Dict) -> float | None:
    if rec.get("task_rate_target") is not None:
        return float(rec["task_rate_target"])
    if rec.get("load_level") is not None:
        return float(rec["load_level"])
    return None


def is_task_rate_sweep_record(rec: Dict) -> bool:
    return rec.get("sweep_type") == "task_rate"


def is_compute_sweep_record(rec: Dict) -> bool:
    return rec.get("sweep_type") == "compute"


def compute_sweep_task_rate_label(records: List[Dict]) -> float:
    for rec in records:
        if rec.get("sweep_type") == "compute" and rec.get("task_rate_target") is not None:
            return float(rec["task_rate_target"])
    return float(DEFAULT_COMPUTE_SWEEP_TASK_RATE)


def is_offloading_plot_record(rec: Dict) -> bool:
    return (
        rec.get("experiment_group") == "offloading"
        and rec.get("algorithm") == "DGR-D3QN"
    )


def is_algorithm_plot_record(rec: Dict) -> bool:
    return (
        rec.get("experiment_group") == "algorithm"
        and rec.get("scheme") == "MSCO"
    )


def training_log_relative_path(method: str, *, quick: bool = False) -> str:
    filename = TRAINING_LOG_FILES.get(method)
    if not filename:
        raise ValueError(f"Unknown training method for log path: {method}")
    if quick:
        return os.path.join("training_process_data", "compute_intensive", "debug", filename)
    return os.path.join("training_process_data", "compute_intensive", filename)


def resolve_training_log_path(log_dir: str, method: str) -> Optional[str]:
    filename = TRAINING_LOG_FILES.get(method)
    if not filename:
        return None
    path = os.path.join(log_dir, filename)
    if os.path.exists(path):
        return path
    debug_path = os.path.join(log_dir, "debug", filename)
    if os.path.exists(debug_path):
        return debug_path
    return None


def _latest_run_id(records: List[Dict]) -> Optional[str]:
    run_ids = [str(record["run_id"]) for record in records if record.get("run_id")]
    if not run_ids:
        return None
    return max(run_ids)


def _reward_from_jsonl_record(record: Dict) -> float | None:
    for key in ("episode_reward", "Average ending reward", "moving_avg_reward_50", "moving_avg_reward_100"):
        value = record.get(key)
        if value is None:
            continue
        return float(value)
    return None


def load_training_rewards(log_dir: str, method: str) -> List[float]:
    path = resolve_training_log_path(log_dir, method)
    if not path:
        return []

    json_records: List[Dict] = []
    legacy_rewards: List[float] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    json_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                continue
            match = REWARD_LINE.search(line)
            if not match:
                continue
            token = match.group(1)
            if token == "None":
                continue
            legacy_rewards.append(float(token))

    if not json_records:
        return legacy_rewards

    latest_run_id = _latest_run_id(json_records)
    rewards: List[float] = []
    for record in json_records:
        if latest_run_id is not None and record.get("run_id") != latest_run_id:
            continue
        reward = _reward_from_jsonl_record(record)
        if reward is not None:
            rewards.append(reward)
    return rewards
