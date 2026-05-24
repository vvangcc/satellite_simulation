#!/usr/bin/env python3
"""Plot mixed_tasks experiment results."""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot mixed_tasks results")
    parser.add_argument(
        "--input-dir",
        default=os.path.join("results", "mixed_tasks"),
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "mixed_tasks", "figures"),
        help="Directory for figure output",
    )
    return parser.parse_args()


def load_records(input_dir: str):
    records = []
    for path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as fp:
            records.append(json.load(fp))
    return records


def _bar_plot(records, metric, ylabel, title, output_path, order=("PPO", "D3QN", "PER-D3QN")):
    by_method = {}
    for rec in records:
        name = rec.get("method_name")
        by_method.setdefault(name, []).append(rec.get(metric, 0.0))
    methods = [m for m in order if m in by_method]
    values = [sum(by_method[m]) / len(by_method[m]) for m in methods]
    plt.figure(figsize=(7, 4))
    plt.bar(methods, values, color=["#4C72B0", "#55A868", "#C44E52"][: len(methods)])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _reward_curve(records, output_path, order=("PPO", "D3QN", "PER-D3QN")):
    plt.figure(figsize=(8, 4))
    for name in order:
        series = []
        for rec in records:
            if rec.get("method_name") == name:
                series.extend(rec.get("episode_reward_list", []))
        if series:
            plt.plot(range(1, len(series) + 1), series, label=name)
    plt.xlabel("Logging step")
    plt.ylabel("Average ending reward")
    plt.title("Training average ending reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.input_dir)
    if not records:
        print(f"No JSON results found in {args.input_dir}")
        return

    _bar_plot(
        records,
        "on_time_rate_D",
        "On-time completion rate",
        "Delay-sensitive task on-time rate (D)",
        os.path.join(args.output_dir, "on_time_rate_D_comparison.png"),
    )
    _bar_plot(
        records,
        "soft_deadline_rate_C",
        "Soft-deadline completion rate",
        "Compute-intensive soft-deadline rate (C)",
        os.path.join(args.output_dir, "soft_deadline_rate_C_comparison.png"),
    )
    _bar_plot(
        records,
        "average_delay_C",
        "Average end-to-end delay (s)",
        "Compute-intensive average delay (C)",
        os.path.join(args.output_dir, "average_delay_C_comparison.png"),
    )
    _reward_curve(
        records,
        os.path.join(args.output_dir, "training_reward_comparison.png"),
    )
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
