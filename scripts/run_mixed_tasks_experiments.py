#!/usr/bin/env python3
"""Run mixed_tasks (multi-type task) experiments."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mixed_tasks.method_registry import all_methods, get_method
from mixed_tasks.runner import run_experiment_from_config_path


def parse_args():
    parser = argparse.ArgumentParser(description="mixed_tasks multi-type task experiments")
    parser.add_argument("--phase", choices=["train", "test", "all"], required=True)
    parser.add_argument("--method", choices=["PPO", "D3QN", "PER-D3QN"], default=None)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "mixed_tasks"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Short smoke run (duration=30)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = os.path.join(ROOT, "configs", "mixed_tasks")
    methods = [get_method(args.method)] if args.method else all_methods()
    phases = ["train", "test"] if args.phase == "all" else [args.phase]
    duration = 30 if args.quick else args.duration

    for spec in methods:
        for phase in phases:
            cfg_name = spec.train_config if phase == "train" else spec.config_file
            config_path = os.path.join(config_dir, cfg_name)
            overrides = {}
            if args.seed is not None:
                overrides.setdefault("general", {})["random_seed"] = args.seed
            if duration is not None:
                overrides.setdefault("general", {})["duration"] = duration
            print(f"Running mixed_tasks {spec.method_name} phase={phase} via {cfg_name}")
            path = run_experiment_from_config_path(
                config_path,
                output_dir=args.output_dir,
                overrides=overrides or None,
            )
            print(f"  saved: {path}")


if __name__ == "__main__":
    main()
