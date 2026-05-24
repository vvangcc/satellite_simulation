"""Run a single compute_intensive experiment from a YAML configuration."""
from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from baselines.modes import resolve_simulator_mode
from compute_intensive.checkpoint import (
    best_model_path_from_model_path,
    build_checkpoint_path,
    checkpoint_stem_from_model_path,
    load_agent_checkpoint,
    save_agent_checkpoint,
    save_inference_model,
)
from compute_intensive.metrics import build_result_record, save_result, SCENARIO_NAME
from compute_intensive.training_logger import TrainingLogger
from PRC import build_agent, cfg_get, normalize_model_path

RL_AGENT_MODES = {
    "Pure_DQN", "New_DQN", "Pure_PPO", "New_PPO", "Weak_DQN",
    "Pure_SAC", "New_SAC", "DGR_D3QN", "D3QN", "DQN",
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def _build_environment(config: Dict[str, Any], agent, policy_net, device):
    from RL_environment_for_computing import SatelliteEnv

    sim_mode = resolve_simulator_mode(config["agent"]["mode"])
    phase = config["general"]["phase"]
    epsilon = cfg_get(config["general"], "epsilon", "epsilon_start", default=0.5)
    if phase != "train":
        epsilon = 0
    env_cfg = config["environment"]

    experiment_meta = config.get("experiment", {})
    return SatelliteEnv(
        mode=sim_mode,
        select_mode=config["general"]["select_mode"],
        q_net=policy_net,
        epsilon=epsilon,
        reward_factors=config["general"]["reward_factors"],
        device=device,
        mission_possibility=env_cfg["mission_possibility"],
        poisson_rate=env_cfg["poisson_rate"],
        packet_frequency=env_cfg["packet_frequency"],
        computing_demand_factor=env_cfg["computing_demand_factor"],
        computing_demand_factor_2=env_cfg["computing_demand_factor_2"],
        size_after_computing_factor=env_cfg["size_after_computing_factor"],
        size_after_computing_1=env_cfg["size_after_computing_1"],
        begin_time=config["general"]["begin_time"],
        end_time=None,
        time_stride=config["general"]["time_stride"],
        tle_filepath=env_cfg["tle_filepath"],
        SOD_file_path=env_cfg["SOD_file_path"],
        mean_interval_time=env_cfg["mean_interval_time"],
        memory=env_cfg["memory"],
        computing_ability=env_cfg["computing_ability"],
        transmission_rate=env_cfg["transmission_rate"],
        downlink_rate=env_cfg["downlink_rate"],
        downstream_delays=env_cfg["downstream_delays"],
        packet_size_range=env_cfg["packet_size_range"],
        state_update_period=env_cfg["state_update_period"],
        print_cycle=config["general"]["print_cycle"],
        del_cycle=env_cfg["del_cycle"],
        visualize=env_cfg["visualize"],
        print_info=env_cfg["print_info"],
        show_detail=env_cfg["show_detail"],
        save_log=env_cfg["save_log"],
        random_edges_del=env_cfg["random_edges_del"],
        random_nodes_del=env_cfg["random_nodes_del"],
        update_cycle=env_cfg["update_cycle"],
        save_training_data=env_cfg.get("save_training_data"),
        elevation_angle=env_cfg["elevation_angle"],
        pole=env_cfg["pole"],
        task_profile=cfg_get(env_cfg, "task_profile", default="legacy"),
        task_num_stages=env_cfg.get("task_num_stages"),
        size_reduction_factor=env_cfg.get("size_reduction_factor"),
        final_result_size_range=env_cfg.get("final_result_size_range"),
        heartbeat_timeout=cfg_get(env_cfg, "heartbeat_timeout", default=0.25),
        target_task_rate_per_sec=env_cfg.get("target_task_rate_per_sec"),
        scheme=experiment_meta.get("scheme"),
    )


def _replay_buffer_size(agent) -> int:
    buffer = getattr(agent, "replay_buffer", None)
    return len(buffer) if buffer is not None else 0


def _save_best_model(agent, config: Dict[str, Any], training_logger: Optional[TrainingLogger]) -> None:
    if training_logger is None or training_logger.best_moving_avg_reward is None:
        return
    if not config.get("general", {}).get("save_best_model", False):
        return
    model_path = normalize_model_path(config["agent"].get("model_path"))
    if not model_path:
        return
    best_path = best_model_path_from_model_path(model_path)
    save_inference_model(agent, best_path)


def _save_periodic_checkpoint(
    agent,
    config: Dict[str, Any],
    *,
    global_step: int,
    epsilon: float,
    episode_reward_list,
    interrupt: bool = False,
) -> Optional[str]:
    general = config.get("general", {})
    checkpoint_interval = general.get("checkpoint_interval")
    checkpoint_dir = general.get("checkpoint_dir")
    if not checkpoint_interval or not checkpoint_dir:
        return None
    if not interrupt and global_step % int(checkpoint_interval) != 0:
        return None

    model_path = normalize_model_path(config["agent"].get("model_path"))
    stem = checkpoint_stem_from_model_path(model_path or "train_model")
    path = build_checkpoint_path(checkpoint_dir, stem, global_step, interrupt=interrupt)
    save_agent_checkpoint(
        agent,
        path,
        epsilon=epsilon,
        global_step=global_step,
        episode_reward_list=episode_reward_list,
    )
    return path


def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one simulation run and return Chapter 3 metrics."""
    seed = config["general"]["random_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase = config["general"]["phase"]
    mode = config["agent"]["mode"]
    general = config.get("general", {})

    agent = None
    if mode in RL_AGENT_MODES:
        agent = build_agent(config, device, phase)

    policy_net = None
    if agent:
        policy_net = getattr(agent, "online_net", None)
        if policy_net is None:
            policy_net = getattr(agent, "actor", None)

    env = _build_environment(config, agent, policy_net, device)

    begin_time = config["general"]["begin_time"]
    time_stride = config["general"]["time_stride"]
    rounds = config["general"]["rounds"]
    skip_time = config["general"]["skip_time"]
    duration = config["general"]["duration"]
    epsilon = cfg_get(general, "epsilon", "epsilon_start", default=0.5)
    min_epsilon = cfg_get(general, "min_epsilon", "epsilon_min", default=0.001)
    epsilon_decay = general["epsilon_decay"]
    target_update_steps = cfg_get(config["agent"], "update_cycle", "target_update_steps", default=30)

    if phase != "train":
        epsilon = 0

    global_step = 0
    episode_reward_list = []
    training_logger = None

    if phase == "train" and agent:
        resume_path = normalize_model_path(general.get("resume_from_checkpoint"))
        active_resume_path = resume_path if resume_path and os.path.exists(resume_path) else None
        if active_resume_path:
            restored = load_agent_checkpoint(agent, active_resume_path)
            global_step = restored["global_step"]
            epsilon = restored["epsilon"]
            episode_reward_list = restored["episode_reward_list"]
            env.episode_reward_list = list(episode_reward_list)
            if not restored["replay_buffer_restored"]:
                print(
                    "WARNING: Replay buffer is not restored; training will continue with empty buffer."
                )
            print(f"Resumed training from checkpoint: {active_resume_path} (global_step={global_step})")

        save_training_data = config.get("environment", {}).get("save_training_data")
        if save_training_data:
            log_path = os.path.join("./training_process_data", save_training_data)
            experiment_meta = config.get("experiment", {})
            training_logger = TrainingLogger(
                log_path,
                window=int(general.get("moving_avg_window", 50)),
                target_task_rate=config.get("environment", {}).get("target_task_rate_per_sec"),
                run_id=general.get("training_run_id"),
                method_name=experiment_meta.get("method_name"),
                resume_from_checkpoint=active_resume_path,
            )
            if episode_reward_list:
                for reward in episode_reward_list[-training_logger.window :]:
                    training_logger._rewards.append(reward)

            def _training_log_hook(record):
                prev_best = training_logger.best_moving_avg_reward
                training_logger.log_interval(record)
                if training_logger.best_moving_avg_reward != prev_best:
                    _save_best_model(agent, config, training_logger)

            env.training_log_hook = _training_log_hook

    total_sim_seconds = 0.0
    interrupt_checkpoint_path = None

    try:
        for _ in range(rounds):
            env.reset(begin_time)
            if phase == "train":
                env.episode_reward_list = list(episode_reward_list)
            steps = int(duration / time_stride)
            total_sim_seconds += steps * time_stride
            for t in range(steps):
                global_step += 1
                if phase == "train" and agent:
                    env.training_context = {
                        "epsilon": epsilon,
                        "replay_buffer_size": _replay_buffer_size(agent),
                        "loss": getattr(agent, "last_loss", None),
                    }
                experiences = env.step(epsilon)
                if phase == "train" and agent:
                    epsilon = max(min_epsilon, epsilon * epsilon_decay)
                    agent.update(experiences)
                    env.training_context["loss"] = getattr(agent, "last_loss", None)
                    env.training_context["replay_buffer_size"] = _replay_buffer_size(agent)
                    episode_reward_list = list(env.episode_reward_list)

                    if (global_step) % int(target_update_steps) == 0:
                        if mode in {"DGR_D3QN", "D3QN", "Pure_DQN", "New_DQN", "Weak_DQN", "DQN"} and "SAC" not in mode:
                            if hasattr(agent, "target_update"):
                                agent.target_update()

                    checkpoint_path = _save_periodic_checkpoint(
                        agent,
                        config,
                        global_step=global_step,
                        epsilon=epsilon,
                        episode_reward_list=episode_reward_list,
                    )
                    if checkpoint_path:
                        print(f"Checkpoint saved: {checkpoint_path}")

            if phase == "test" and config.get("general", {}).get("verbose", False):
                env.show_satellite_computing_time()
            begin_time = env.add_time_to_str(begin_time, skip_time)

        if phase == "train" and agent:
            model_path = normalize_model_path(config["agent"].get("model_path"))
            save_inference_model(agent, model_path)
            if model_path:
                print(f"Final model saved: {model_path}")
            if training_logger and general.get("save_best_model", False):
                best_path = best_model_path_from_model_path(model_path or "")
                if model_path and os.path.exists(best_path):
                    print(f"Best model saved: {best_path}")

    except KeyboardInterrupt:
        if phase == "train" and agent:
            interrupt_checkpoint_path = _save_periodic_checkpoint(
                agent,
                config,
                global_step=global_step,
                epsilon=epsilon,
                episode_reward_list=episode_reward_list,
                interrupt=True,
            )
            if interrupt_checkpoint_path:
                print(f"Training interrupted. Checkpoint saved to {interrupt_checkpoint_path}")
            model_path = normalize_model_path(config["agent"].get("model_path"))
            save_inference_model(agent, model_path)
            if model_path:
                print(f"Latest weights saved: {model_path}")
        raise

    env.simulation_duration_sec = total_sim_seconds
    return env.get_compute_intensive_metrics()


def _experiment_metadata(config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    env_cfg = config.get("environment", {})
    task_rate_target = env_cfg.get("target_task_rate_per_sec")
    computing_ability = env_cfg.get("computing_ability")
    gflops = (computing_ability / 1e9) if computing_ability else None
    random_edges_del = env_cfg.get("random_edges_del")
    experiment_meta = config.get("experiment", {})
    return {
        "load_level": task_rate_target,
        "task_rate_target": task_rate_target,
        "actual_task_rate": metrics.get("actual_task_rate"),
        "computing_ability": computing_ability,
        "computing_ability_gflops": gflops,
        "random_edges_del": random_edges_del,
        "removed_edges_count": metrics.get("removed_edges_count"),
        "sweep_type": experiment_meta.get("sweep_type", "single"),
        "packet_frequency": env_cfg.get("packet_frequency"),
        "simulation_duration_sec": metrics.get("simulation_duration_sec"),
    }


def run_experiment_from_config_path(
    config_path: str,
    output_dir: str = "results/compute_intensive",
    overrides: Optional[Dict[str, Any]] = None,
    save_results: bool = True,
) -> str | Dict[str, Any]:
    config = load_config(config_path)
    if overrides:
        config = copy.deepcopy(config)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

    metrics = run_simulation(config)
    experiment_meta = config.get("experiment", {})
    meta = _experiment_metadata(config, metrics)
    record = build_result_record(
        scenario=config.get("scenario", SCENARIO_NAME),
        method_name=experiment_meta.get("method_name", config["agent"]["mode"]),
        scheme=experiment_meta.get("scheme"),
        algorithm=experiment_meta.get("algorithm"),
        raw_mode_name=experiment_meta.get("raw_mode_name", config["agent"]["mode"]),
        experiment_group=experiment_meta.get("experiment_group", "unknown"),
        load_level=meta["load_level"],
        seed=config["general"]["random_seed"],
        statics_datas=metrics["statics_datas"],
        hop_count_list=metrics["hop_count_list"],
        episode_reward_list=metrics["episode_reward_list"],
        sweep_type=meta["sweep_type"],
        task_rate_target=meta["task_rate_target"],
        actual_task_rate=meta["actual_task_rate"],
        computing_ability=meta["computing_ability"],
        computing_ability_gflops=meta["computing_ability_gflops"],
        random_edges_del=meta["random_edges_del"],
        removed_edges_count=meta["removed_edges_count"],
        packet_frequency=meta["packet_frequency"],
        simulation_duration_sec=meta["simulation_duration_sec"],
    )
    if not save_results:
        return record
    return save_result(record, output_dir)
