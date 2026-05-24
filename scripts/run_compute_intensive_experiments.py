#!/usr/bin/env python3
"""Run compute_intensive (section 3.6) offloading, algorithm, and ablation experiments."""
from __future__ import annotations

import argparse
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compute_intensive.metrics import build_failed_record, save_result, SCENARIO_NAME
from compute_intensive.method_registry import get_methods_for_group, get_train_method
from compute_intensive.plot_helpers import (
    DEFAULT_COMPUTE_SWEEP_TASK_RATE,
    training_log_relative_path,
)
from compute_intensive.runner import run_experiment_from_config_path

DEFAULT_TASK_RATE_LEVELS = [30, 40, 50, 60, 70, 80]
DEFAULT_COMPUTE_LEVELS_GFLOPS = [40, 60, 80, 100, 120]


def parse_args():
    parser = argparse.ArgumentParser(description="compute_intensive experiment runner")
    parser.add_argument(
        "--group",
        choices=["offloading", "algorithm", "ablation", "all"],
        help="Experiment group (not used with --phase train)",
    )
    parser.add_argument(
        "--phase",
        choices=["train", "test"],
        default="test",
        help="Train RL weights or run evaluation",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Paper-facing method name (required for --phase train)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "results", "compute_intensive"),
        help="Directory for JSON result files",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override simulation duration in seconds",
    )
    parser.add_argument(
        "--task-rate",
        type=float,
        default=None,
        help="Override environment.target_task_rate_per_sec for a single run",
    )
    parser.add_argument(
        "--sweep",
        choices=["task_rate", "compute"],
        default=None,
        help="Sweep task arrival rate or satellite compute capacity",
    )
    parser.add_argument(
        "--compute-gflops",
        type=float,
        default=None,
        help="Override computing_ability as GFLOPS for a single run",
    )
    parser.add_argument(
        "--random-edges-del",
        type=int,
        default=None,
        help="Override environment.random_edges_del (count of edges removed, not a percentage)",
    )
    parser.add_argument(
        "--failure-rate",
        type=int,
        default=None,
        help="Alias of --random-edges-del; this is an edge count, not a percentage.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run; uses --duration if set, otherwise 5 seconds",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Override training duration (simulation steps) for --phase train",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint path (overrides resume_from_checkpoint)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-satellite computing time dictionary after each run",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing .json result files in --output-dir before running",
    )
    parser.add_argument(
        "--clean-figures",
        action="store_true",
        help="Delete existing figure files under --output-dir/figures before running",
    )
    parser.add_argument(
        "--clean-training-log",
        action="store_true",
        help="Delete the training log for --method before a fresh train run",
    )
    return parser.parse_args()


def _resolve_random_edges_del(args) -> int | None:
    if args.random_edges_del is not None:
        return args.random_edges_del
    return args.failure_rate


def _resolve_duration(args) -> int | None:
    if args.duration is not None:
        return args.duration
    if args.quick:
        return 5
    return None


def _resolve_compute_sweep_task_rate(args) -> float:
    if args.task_rate is not None:
        return args.task_rate
    return float(DEFAULT_COMPUTE_SWEEP_TASK_RATE)


def _resolve_sweep_type(args) -> str:
    if args.sweep == "task_rate":
        return "task_rate"
    if args.sweep == "compute":
        return "compute"
    return "single"


def _computing_ability_from_gflops(compute_gflops: float | None) -> float | None:
    if compute_gflops is not None:
        return float(compute_gflops) * 1e9
    return None


def _clean_output_dir(output_dir: str, *, clean_json: bool, clean_figures: bool) -> None:
    if clean_json and os.path.isdir(output_dir):
        removed = 0
        for name in os.listdir(output_dir):
            if name.endswith(".json") and os.path.isfile(os.path.join(output_dir, name)):
                os.remove(os.path.join(output_dir, name))
                removed += 1
        if removed:
            print(f"Removed {removed} existing JSON file(s) from {output_dir}")
    if clean_figures:
        figures_dir = os.path.join(output_dir, "figures")
        if os.path.isdir(figures_dir):
            removed = 0
            for name in os.listdir(figures_dir):
                path = os.path.join(figures_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            if removed:
                print(f"Removed {removed} existing figure file(s) from {figures_dir}")


def _build_env_overrides(args, task_rate=None, compute_gflops=None):
    env = {"packet_frequency": 1.0}
    if task_rate is not None:
        env["target_task_rate_per_sec"] = task_rate
    if compute_gflops is not None:
        env["computing_ability"] = float(compute_gflops) * 1e9
    edges_del = _resolve_random_edges_del(args)
    if edges_del is not None:
        env["random_edges_del"] = edges_del
    if args.quick:
        env.update(
            {
                "packet_size_range": [5120, 51200],
                "computing_demand_factor": [200, 400],
                "task_num_stages": [2, 2],
                "final_result_size_range": [5120, 15360],
            }
        )
    return env


def _run_one(spec, args, overrides, task_rate=None, compute_gflops=None):
    config_dir = os.path.join(ROOT, "configs", "compute_intensive")
    config_path = os.path.join(config_dir, spec.config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")

    env_overrides = _build_env_overrides(args, task_rate=task_rate, compute_gflops=compute_gflops)
    if env_overrides:
        overrides = {**overrides, "environment": {**overrides.get("environment", {}), **env_overrides}}

    label_parts = [spec.method_name]
    if task_rate is not None:
        label_parts.append(f"task_rate={task_rate}")
    if compute_gflops is not None:
        label_parts.append(f"compute={compute_gflops}GFLOPS")
    print(f"  -> {' '.join(label_parts)} (raw_mode={spec.raw_mode_name}) via {spec.config_file}")

    path = run_experiment_from_config_path(
        config_path,
        output_dir=args.output_dir,
        overrides=overrides or None,
    )
    print(f"     saved: {path}")
    return path


def _train_model_stem(method_name: str) -> str:
    mapping = {
        "DQN": "train_dqn",
        "D3QN": "train_d3qn",
        "DGR-D3QN": "train_dgr_d3qn",
    }
    return mapping[method_name]


def _clean_training_log(method_name: str, *, quick: bool) -> None:
    rel_path = training_log_relative_path(method_name, quick=quick)
    log_path = os.path.join(ROOT, rel_path)
    if os.path.isfile(log_path):
        os.remove(log_path)
        print(f"Removed training log: {log_path}")


def _run_train(args):
    if not args.method:
        raise SystemExit("--method is required for --phase train (DQN, D3QN, or DGR-D3QN)")
    spec = get_train_method(args.method)

    if args.clean_training_log and args.resume:
        print("clean-training-log is ignored when resume is used.")
    elif args.clean_training_log:
        _clean_training_log(spec.method_name, quick=args.quick)

    from datetime import datetime

    config_dir = os.path.join(ROOT, "configs", "compute_intensive")
    config_path = os.path.join(config_dir, spec.config_file)
    overrides = {"general": {"phase": "train"}}
    overrides.setdefault("general", {})["training_run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    overrides.setdefault("experiment", {})["method_name"] = spec.method_name
    if args.seed is not None:
        overrides.setdefault("general", {})["random_seed"] = args.seed

    duration = _resolve_duration(args) if args.duration is not None or args.quick else None
    if args.train_steps is not None:
        duration = args.train_steps
    elif duration is not None:
        pass
    if duration is not None:
        overrides.setdefault("general", {})["duration"] = duration

    if args.task_rate is not None:
        overrides.setdefault("environment", {})["target_task_rate_per_sec"] = args.task_rate
    edges_del = _resolve_random_edges_del(args)
    if edges_del is not None:
        overrides.setdefault("environment", {})["random_edges_del"] = edges_del

    model_stem = _train_model_stem(spec.method_name)
    if args.quick:
        debug_dir = os.path.join(ROOT, "model_weights", "compute_intensive", "debug")
        overrides.setdefault("agent", {})["model_path"] = os.path.join(debug_dir, f"{model_stem}.pth")
        overrides.setdefault("general", {})["checkpoint_dir"] = os.path.join(debug_dir, "checkpoints")
        overrides.setdefault("environment", {})["save_training_data"] = f"compute_intensive/debug/{model_stem}.txt"

    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
        overrides.setdefault("general", {})["resume_from_checkpoint"] = resume_path

    print(f"Training {spec.method_name} via {spec.config_file}")
    if args.quick:
        print("  Quick mode: models -> model_weights/compute_intensive/debug/")
        print("  Training logs -> training_process_data/compute_intensive/debug/")
    else:
        print("  Models -> model_weights/compute_intensive/")
        print("  Training logs -> training_process_data/compute_intensive/")
    from compute_intensive.runner import load_config

    cfg = load_config(config_path)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    train_rate = cfg.get("environment", {}).get("target_task_rate_per_sec")
    if train_rate is not None:
        print(f"  target_task_rate_per_sec: {train_rate}")
    if args.train_steps is not None:
        print(f"  train steps (duration): {args.train_steps}")
    if args.resume:
        print(f"  resume_from_checkpoint: {overrides.get('general', {}).get('resume_from_checkpoint')}")

    try:
        run_experiment_from_config_path(config_path, output_dir=args.output_dir, overrides=overrides, save_results=False)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return
    print("Training run finished.")


def main():
    args = parse_args()
    if args.clean_output or args.clean_figures:
        _clean_output_dir(
            args.output_dir,
            clean_json=args.clean_output,
            clean_figures=args.clean_figures,
        )
    if args.phase == "train":
        _run_train(args)
        return

    if not args.group:
        raise SystemExit("--group is required for test runs")

    methods = get_methods_for_group(args.group)
    duration = _resolve_duration(args)
    sweep_type = _resolve_sweep_type(args)

    if args.sweep == "task_rate":
        sweep_values = [(rate, None) for rate in DEFAULT_TASK_RATE_LEVELS]
    elif args.sweep == "compute":
        if args.group != "offloading":
            raise SystemExit("--sweep compute is only supported for --group offloading")
        fixed_rate = _resolve_compute_sweep_task_rate(args)
        sweep_values = [(fixed_rate, gflops) for gflops in DEFAULT_COMPUTE_LEVELS_GFLOPS]
    elif args.task_rate is not None:
        sweep_values = [(args.task_rate, None)]
    elif args.compute_gflops is not None:
        rate = args.task_rate if args.task_rate is not None else None
        sweep_values = [(rate, args.compute_gflops)]
    else:
        sweep_values = [(None, None)]

    print(f"Running compute_intensive experiments: group={args.group}, phase=test, runs={len(methods) * len(sweep_values)}")
    if duration is not None:
        print(f"Simulation duration override: {duration}s")
    if args.sweep == "compute":
        print(f"Compute sweep fixed task rate: {_resolve_compute_sweep_task_rate(args)} tasks/s")

    total_runs = 0
    success_runs = 0
    failed_runs = 0
    saved_paths = []

    for spec in methods:
        for task_rate, compute_gflops in sweep_values:
            total_runs += 1
            overrides = {}
            overrides.setdefault("experiment", {})["sweep_type"] = sweep_type
            if args.seed is not None:
                overrides.setdefault("general", {})["random_seed"] = args.seed
            if duration is not None:
                general = overrides.setdefault("general", {})
                general["duration"] = duration
                general["print_cycle"] = max(1, min(duration, 10))
            if args.verbose:
                overrides.setdefault("general", {})["verbose"] = True

            env_snapshot = _build_env_overrides(args, task_rate=task_rate, compute_gflops=compute_gflops)

            try:
                path = _run_one(spec, args, overrides, task_rate=task_rate, compute_gflops=compute_gflops)
                saved_paths.append(path)
                success_runs += 1
            except FileNotFoundError as exc:
                failed_runs += 1
                record = build_failed_record(
                    scenario=SCENARIO_NAME,
                    method_name=spec.method_name,
                    scheme=spec.scheme,
                    algorithm=spec.algorithm,
                    raw_mode_name=spec.raw_mode_name,
                    experiment_group=spec.experiment_group,
                    load_level=task_rate,
                    seed=overrides.get("general", {}).get("random_seed", 42),
                    error_message=str(exc),
                    sweep_type=sweep_type,
                    task_rate_target=task_rate,
                    computing_ability=_computing_ability_from_gflops(compute_gflops),
                    computing_ability_gflops=compute_gflops,
                    random_edges_del=env_snapshot.get("random_edges_del"),
                )
                path = save_result(record, args.output_dir)
                saved_paths.append(path)
                print(f"     failed (saved): {path}")
                print(f"     error: {exc}")
            except Exception as exc:
                failed_runs += 1
                record = build_failed_record(
                    scenario=SCENARIO_NAME,
                    method_name=spec.method_name,
                    scheme=spec.scheme,
                    algorithm=spec.algorithm,
                    raw_mode_name=spec.raw_mode_name,
                    experiment_group=spec.experiment_group,
                    load_level=task_rate,
                    seed=overrides.get("general", {}).get("random_seed", 42),
                    error_message=f"{type(exc).__name__}: {exc}",
                    sweep_type=sweep_type,
                    task_rate_target=task_rate,
                    computing_ability=_computing_ability_from_gflops(compute_gflops),
                    computing_ability_gflops=compute_gflops,
                    random_edges_del=env_snapshot.get("random_edges_del"),
                )
                path = save_result(record, args.output_dir)
                saved_paths.append(path)
                print(f"     failed (saved): {path}")
                traceback.print_exc()

    print("\n=== Sweep summary ===")
    print(f"total_runs:   {total_runs}")
    print(f"success_runs: {success_runs}")
    print(f"failed_runs:  {failed_runs}")
    print(f"result files: {len(saved_paths)} in {args.output_dir}")


if __name__ == "__main__":
    main()
