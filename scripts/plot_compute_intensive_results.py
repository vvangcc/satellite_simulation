#!/usr/bin/env python3
"""Plot compute_intensive paper figures from results/compute_intensive JSON outputs."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


OFFLOADING_ORDER = ["GO", "SSO", "MSCO"]
ALGORITHM_ORDER = ["GA", "DQN", "DGR-D3QN"]
ABLATION_ORDER = ["D3QN", "DGR-D3QN"]

TASK_RATE_XLABEL = "Average tasks per second\n平均每秒任务数"
COMPUTE_XLABEL = "Satellite compute capacity / GFLOPS\n卫星计算能力 / GFLOPS"

from compute_intensive.plot_helpers import (
    compute_sweep_task_rate_label,
    is_algorithm_plot_record,
    is_compute_sweep_record,
    is_offloading_plot_record,
    is_task_rate_sweep_record,
    load_training_rewards,
    x_key_task_rate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot compute_intensive experiment figures")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(ROOT, "results", "compute_intensive"),
        help="Directory containing experiment JSON files",
    )
    parser.add_argument(
        "--training-log-dir",
        default=os.path.join(ROOT, "training_process_data", "compute_intensive"),
        help="Directory containing compute_intensive training reward txt logs",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "results", "compute_intensive", "figures"),
        help="Directory for saved figures",
    )
    return parser.parse_args()


def load_records(input_dir: str) -> List[Dict]:
    records = []
    for path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as fp:
            records.append(json.load(fp))
    return records


def _mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _x_key(rec: Dict) -> Optional[float]:
    return x_key_task_rate(rec)


def _compute_x_key(rec: Dict) -> Optional[float]:
    if rec.get("computing_ability_gflops") is not None:
        return float(rec["computing_ability_gflops"])
    return None


def group_by_method_and_x(records: List[Dict], experiment_group: str, x_fn, record_filter=None):
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("experiment_group") != experiment_group:
            continue
        if rec.get("status") == "failed":
            continue
        if record_filter is not None and not record_filter(rec):
            continue
        x = x_fn(rec)
        if x is None:
            continue
        grouped[rec["method_name"]][x].append(rec)
    return grouped


def _series_for_metric(grouped, methods, metric):
    xs = sorted({x for method in grouped.values() for x in method.keys() if x is not None})
    series = {}
    for method in methods:
        ys = []
        for x in xs:
            items = grouped.get(method, {}).get(x, [])
            ys.append(_mean([item.get(metric) for item in items]))
        series[method] = ys
    return xs, series


def _line_plot(xs, series, xlabel, ylabel, title, path, methods):
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    for idx, method in enumerate(methods):
        ys = series.get(method, [])
        if not ys or all(y is None for y in ys):
            continue
        ax.plot(xs, ys, marker="o", label=method, color=colors[idx % len(colors)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reward_curve(records, methods: Tuple[str, str], path, title, training_log_dir: str):
    rewards_by_method = {m: load_training_rewards(training_log_dir, m) for m in methods}
    for method in methods:
        if not rewards_by_method[method]:
            for rec in records:
                if rec.get("method_name") == method and rec.get("episode_reward_list"):
                    rewards_by_method[method] = rec["episode_reward_list"]
                    break
    if not any(rewards_by_method.values()):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, rewards in rewards_by_method.items():
        if rewards:
            ax.plot(range(1, len(rewards) + 1), rewards, marker="o", label=method)
    ax.set_xlabel("Training interval")
    ax.set_ylabel("Average episode ending reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _offloading_task_rate_filter(rec: Dict) -> bool:
    return is_offloading_plot_record(rec) and is_task_rate_sweep_record(rec)


def _offloading_compute_filter(rec: Dict) -> bool:
    return is_offloading_plot_record(rec) and is_compute_sweep_record(rec)


def _algorithm_task_rate_filter(rec: Dict) -> bool:
    return is_algorithm_plot_record(rec) and is_task_rate_sweep_record(rec)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.input_dir)
    if not records:
        raise SystemExit(f"No JSON results found in {args.input_dir}")

    off_rate = group_by_method_and_x(records, "offloading", _x_key, _offloading_task_rate_filter)
    if off_rate:
        xs, series = _series_for_metric(off_rate, OFFLOADING_ORDER, "average_delay")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Average end-to-end delay (s)",
            "Offloading: average delay vs task rate",
            os.path.join(args.output_dir, "offloading_average_delay_vs_task_rate.png"),
            OFFLOADING_ORDER,
        )
        xs, series = _series_for_metric(off_rate, OFFLOADING_ORDER, "packet_loss_rate")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Packet loss rate",
            "Offloading: packet loss rate vs task rate",
            os.path.join(args.output_dir, "offloading_packet_loss_rate_vs_task_rate.png"),
            OFFLOADING_ORDER,
        )

    off_compute = group_by_method_and_x(records, "offloading", _compute_x_key, _offloading_compute_filter)
    if off_compute:
        compute_task_rate = compute_sweep_task_rate_label(records)
        xs, series = _series_for_metric(off_compute, OFFLOADING_ORDER, "average_delay")
        _line_plot(
            xs, series, COMPUTE_XLABEL, "Average end-to-end delay (s)",
            f"Offloading: average delay vs compute capacity (task rate = {compute_task_rate:g} tasks/s)",
            os.path.join(args.output_dir, "offloading_average_delay_vs_compute_capacity.png"),
            OFFLOADING_ORDER,
        )

    algo_rate = group_by_method_and_x(records, "algorithm", _x_key, _algorithm_task_rate_filter)
    if algo_rate:
        xs, series = _series_for_metric(algo_rate, ALGORITHM_ORDER, "average_delay")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Average end-to-end delay (s)",
            "Algorithm: average delay vs task rate",
            os.path.join(args.output_dir, "algorithm_average_delay_vs_task_rate.png"),
            ALGORITHM_ORDER,
        )
        xs, series = _series_for_metric(algo_rate, ALGORITHM_ORDER, "packet_loss_rate")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Packet loss rate",
            "Algorithm: packet loss rate vs task rate",
            os.path.join(args.output_dir, "algorithm_packet_loss_rate_vs_task_rate.png"),
            ALGORITHM_ORDER,
        )
        xs, series = _series_for_metric(algo_rate, ALGORITHM_ORDER, "average_hops")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Average hop count",
            "Algorithm: average hops vs task rate",
            os.path.join(args.output_dir, "algorithm_average_hops_vs_task_rate.png"),
            ALGORITHM_ORDER,
        )

    ablation_rate = group_by_method_and_x(records, "ablation", _x_key, is_task_rate_sweep_record)
    if ablation_rate:
        xs, series = _series_for_metric(ablation_rate, ABLATION_ORDER, "average_delay")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Average end-to-end delay (s)",
            "Ablation: average delay vs task rate",
            os.path.join(args.output_dir, "ablation_average_delay_vs_task_rate.png"),
            ABLATION_ORDER,
        )
        xs, series = _series_for_metric(ablation_rate, ABLATION_ORDER, "packet_loss_rate")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Packet loss rate",
            "Ablation: packet loss rate vs task rate",
            os.path.join(args.output_dir, "ablation_packet_loss_rate_vs_task_rate.png"),
            ABLATION_ORDER,
        )
        xs, series = _series_for_metric(ablation_rate, ABLATION_ORDER, "average_hops")
        _line_plot(
            xs, series, TASK_RATE_XLABEL, "Average hop count",
            "Ablation: average hops vs task rate",
            os.path.join(args.output_dir, "ablation_average_hops_vs_task_rate.png"),
            ABLATION_ORDER,
        )

    _reward_curve(
        records,
        ("DQN", "DGR-D3QN"),
        os.path.join(args.output_dir, "training_reward_dqn_vs_dgr_d3qn.png"),
        "Training reward: DQN vs DGR-D3QN",
        args.training_log_dir,
    )
    _reward_curve(
        records,
        ("D3QN", "DGR-D3QN"),
        os.path.join(args.output_dir, "training_reward_d3qn_vs_dgr_d3qn.png"),
        "Training reward: D3QN vs DGR-D3QN",
        args.training_log_dir,
    )

    print(f"Figures saved to {args.output_dir}")
    print("Note: task processing time CDF is intentionally not generated (not a compute_intensive metric).")


if __name__ == "__main__":
    main()
