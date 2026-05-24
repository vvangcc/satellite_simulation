"""Run Chapter 4 experiments from YAML configs."""
from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from mixed_tasks.metrics import build_result_record, save_result
from PRC import build_agent, cfg_get, normalize_model_path


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def _build_environment(config: Dict[str, Any], agent, policy_net, device):
    from RL_environment_for_computing import SatelliteEnv

    phase = config["general"]["phase"]
    epsilon = cfg_get(config["general"], "epsilon", "epsilon_start", default=0.5)
    if phase != "train":
        epsilon = 0
    env_cfg = config["environment"]
    reward_factors = config.get("mixed_tasks_reward", config.get("reward_factors", {}))

    return SatelliteEnv(
        mode=config["agent"]["mode"],
        select_mode=config["general"]["select_mode"],
        q_net=policy_net,
        epsilon=epsilon,
        reward_factors=reward_factors,
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
        task_profile=env_cfg.get("task_profile", "mixed_tasks"),
        task_num_stages=env_cfg.get("task_num_stages"),
        size_reduction_factor=env_cfg.get("size_reduction_factor"),
        final_result_size_range=env_cfg.get("final_result_size_range"),
        heartbeat_timeout=cfg_get(env_cfg, "heartbeat_timeout", default=0.25),
    )


def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = config["general"]["random_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase = config["general"]["phase"]
    mode = config["agent"]["mode"]

    agent = build_agent(config, device, phase)
    policy_net = getattr(agent, "online_net", None) or getattr(agent, "actor", None)
    env = _build_environment(config, agent, policy_net, device)

    begin_time = config["general"]["begin_time"]
    time_stride = config["general"]["time_stride"]
    rounds = config["general"]["rounds"]
    skip_time = config["general"]["skip_time"]
    duration = config["general"]["duration"]
    epsilon = cfg_get(config["general"], "epsilon", "epsilon_start", default=0.5)
    min_epsilon = cfg_get(config["general"], "min_epsilon", "epsilon_min", default=0.001)
    epsilon_decay = config["general"]["epsilon_decay"]
    target_update_steps = cfg_get(config["agent"], "update_cycle", "target_update_steps", default=500)

    if phase != "train":
        epsilon = 0

    for _ in range(rounds):
        env.reset(begin_time)
        for t in range(int(duration / time_stride)):
            experiences = env.step(epsilon)
            if phase == "train" and agent:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                agent.update(experiences)
                if (t + 1) % int(target_update_steps) == 0:
                    if hasattr(agent, "target_update"):
                        agent.target_update()
                    model_path = normalize_model_path(config["agent"].get("model_path"))
                    if model_path:
                        agent.save_model(model_path)
        begin_time = env.add_time_to_str(begin_time, skip_time)

    return env.get_mixed_tasks_metrics()


def run_experiment_from_config_path(
    config_path: str,
    output_dir: str = "results/mixed_tasks",
    overrides: Optional[Dict[str, Any]] = None,
) -> str:
    config = load_config(config_path)
    if overrides:
        config = copy.deepcopy(config)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

    metrics = run_simulation(config)
    meta = config.get("experiment", {})
    env_cfg = config.get("environment", {})
    record = build_result_record(
        scenario=config.get("scenario", "mixed_tasks"),
        method_name=meta.get("method_name", config["agent"]["mode"]),
        raw_mode_name=meta.get("raw_mode_name", config["agent"]["mode"]),
        seed=config["general"]["random_seed"],
        load_level=env_cfg.get("packet_frequency", 1.0),
        statics=metrics["statics_datas"],
        episode_reward_list=metrics["episode_reward_list"],
    )
    return save_result(record, output_dir)
