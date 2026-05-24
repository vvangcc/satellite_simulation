import argparse
import os
import yaml
import random
import torch
import numpy as np
from Base_Agents import DDQN_Agent, ShuffleEx, cal_agent_dim, PPO_Agent, DQN_Agent, SAC_Agent
from models.d3qn import D3QN_Agent, FlatD3QN_Agent
from agents.mixed_tasks_d3qn_agent import MixedTasksD3QN_Agent
from agents.mixed_tasks_per_d3qn_agent import MixedTasksPERD3QN_Agent
from models.mixed_tasks_state_layout import compute_mixed_tasks_packed_dim, compute_mixed_tasks_ppo_flat_state_dim
from models.state_layout import compute_dgr_packed_dim, compute_dqn_flat_state_dim, ACTION_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def cfg_get(section, key, *fallback_keys, default=None):
    for lookup_key in (key,) + fallback_keys:
        if lookup_key in section:
            return section[lookup_key]
    return default


def normalize_model_path(path):
    if not path:
        return path
    return os.path.normpath(path)


def resolve_state_layout(agent_cfg):
    mode = agent_cfg['mode']
    action_dim = agent_cfg.get('action_dim', ACTION_DIM)
    if mode in ('DGR_D3QN', 'D3QN'):
        return compute_dgr_packed_dim(), action_dim, None
    if mode in ('CH4_D3QN', 'CH4_PER_D3QN'):
        return compute_mixed_tasks_packed_dim(), action_dim, None
    if mode == 'CH4_PPO':
        return compute_mixed_tasks_ppo_flat_state_dim(), action_dim, None
    if mode == 'DQN':
        return compute_dqn_flat_state_dim(), action_dim, None
    if all(k in agent_cfg for k in ('neighbors_dim', 'edges_dim', 'distance_dim', 'mission_dim', 'current_dim')):
        state_dim, action_dim, state_mask = cal_agent_dim(
            neighbors_dim=agent_cfg['neighbors_dim'],
            edges_dim=agent_cfg['edges_dim'],
            distance_dim=agent_cfg['distance_dim'],
            mission_dim=agent_cfg['mission_dim'],
            current_dim=agent_cfg['current_dim'],
            action_dim=action_dim,
        )
        return state_dim, action_dim, state_mask
    raise ValueError(
        f"Agent mode '{mode}' requires either built-in auto layout (DGR_D3QN/D3QN/DQN) "
        "or explicit neighbors_dim/edges_dim/distance_dim/mission_dim/current_dim."
    )


def build_agent(config, device, phase):
    agent_cfg = config['agent']
    mode = agent_cfg['mode']
    state_dim, action_dim, state_mask = resolve_state_layout(agent_cfg)

    if mode == 'CH4_PER_D3QN':
        agent = MixedTasksPERD3QN_Agent(
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            dgr_hidden_dim=cfg_get(agent_cfg, 'dgr_hidden_dim', default=64),
            q_hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            q_hidden_layers=agent_cfg.get('hidden_layers', 2),
            gnn_layers=cfg_get(agent_cfg, 'gnn_layers', default=2),
            top_k=cfg_get(agent_cfg, 'top_k', default=2),
            lambda_mix=cfg_get(agent_cfg, 'lambda_mix', default=0.6),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
            per_alpha=cfg_get(agent_cfg, 'per_alpha', default=0.6),
            per_beta_start=cfg_get(agent_cfg, 'per_beta_start', default=0.4),
            per_beta_end=cfg_get(agent_cfg, 'per_beta_end', default=1.0),
            per_epsilon=cfg_get(agent_cfg, 'per_epsilon', default=1e-6),
        )
    elif mode == 'CH4_D3QN':
        agent = MixedTasksD3QN_Agent(
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            dgr_hidden_dim=cfg_get(agent_cfg, 'dgr_hidden_dim', default=64),
            q_hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            q_hidden_layers=agent_cfg.get('hidden_layers', 2),
            gnn_layers=cfg_get(agent_cfg, 'gnn_layers', default=2),
            top_k=cfg_get(agent_cfg, 'top_k', default=2),
            lambda_mix=cfg_get(agent_cfg, 'lambda_mix', default=0.6),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
        )
    elif mode == 'D3QN':
        agent = FlatD3QN_Agent(
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            q_hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            q_hidden_layers=agent_cfg.get('hidden_layers', 2),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
        )
    elif mode == 'DGR_D3QN':
        agent = D3QN_Agent(
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            dgr_hidden_dim=cfg_get(agent_cfg, 'dgr_hidden_dim', default=64),
            q_hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            q_hidden_layers=agent_cfg.get('hidden_layers', 2),
            gnn_layers=cfg_get(agent_cfg, 'gnn_layers', default=2),
            top_k=cfg_get(agent_cfg, 'top_k', default=2),
            lambda_mix=cfg_get(agent_cfg, 'lambda_mix', default=0.6),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
        )
    elif 'SAC' in mode:
        agent = SAC_Agent(
            state_dim=state_dim,
            hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            q_mask=agent_cfg['q_mask'],
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            hidden_layers=agent_cfg.get('hidden_layers', 2),
            learning_rate=agent_cfg['value_lr'],
            policy_lr=agent_cfg['policy_lr'],
            repeat=agent_cfg.get('repeat', 1),
            tau=agent_cfg['tau'],
            alpha=agent_cfg['alpha'],
            automatic_entropy_tuning=agent_cfg['automatic_entropy_tuning'],
            target_entropy=agent_cfg['target_entropy'],
            shuffle_func=ShuffleEx(state_mask).shuffle if agent_cfg.get('shuffle') else None,
        )
    elif 'PPO' in mode:
        agent = PPO_Agent(
            state_dim=state_dim,
            hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            q_mask=agent_cfg.get('q_mask', -1),
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            hidden_layers=agent_cfg.get('hidden_layers', 2),
            dueling=agent_cfg.get('dueling', False),
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            shuffle_func=ShuffleEx(state_mask).shuffle if agent_cfg.get('shuffle') else None,
        )
    elif mode == 'DQN':
        agent = DQN_Agent(
            state_dim=state_dim,
            hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            q_mask=agent_cfg.get('q_mask', -1),
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            hidden_layers=agent_cfg.get('hidden_layers', 2),
            dueling=False,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
            shuffle_func=None,
        )
    else:
        use_dueling = agent_cfg.get('dueling', False)
        if 'double_dqn' in agent_cfg:
            use_double = bool(agent_cfg['double_dqn'])
        else:
            use_double = 'Weak' not in mode
        Agent = DQN_Agent if ('Weak' in mode or not use_double) else DDQN_Agent
        agent = Agent(
            state_dim=state_dim,
            hidden_dim=agent_cfg['hidden_dim'],
            action_dim=action_dim,
            buffer_length=cfg_get(agent_cfg, 'buffer_length', 'replay_buffer_size'),
            batch_size=agent_cfg['batch_size'],
            gamma=agent_cfg['gamma'],
            device=device,
            q_mask=agent_cfg['q_mask'],
            activation=agent_cfg.get('activation', 'LeakyRelu'),
            hidden_layers=agent_cfg.get('hidden_layers', 2),
            dueling=use_dueling,
            learning_rate=agent_cfg['learning_rate'],
            repeat=agent_cfg.get('repeat', 1),
            negative_slope=cfg_get(agent_cfg, 'negative_slope', default=0.01),
            shuffle_func=ShuffleEx(state_mask).shuffle if agent_cfg.get('shuffle') else None,
        )

    if phase != 'train':
        model_path = normalize_model_path(agent_cfg.get('model_path'))
        if not model_path:
            return agent
        if not os.path.exists(model_path):
            train_hint = "DGR-D3QN" if mode == "DGR_D3QN" else mode.replace("_", "-")
            raise FileNotFoundError(
                f"Required model weights not found at '{model_path}'. "
                f"Train first, e.g.: python scripts/run_compute_intensive_experiments.py --phase train --method {train_hint}"
            )
        agent.load_model(model_path)
    return agent


def run_simulation_from_config(config):
    """Backward-compatible wrapper used by compute_intensive.runner."""
    from compute_intensive.runner import run_simulation
    return run_simulation(config)


def main():
    args = parse_args()
    config = load_config(args.config)

    random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    np.random.seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['general']['random_seed'])

    from compute_intensive.runner import run_simulation
    run_simulation(config)


if __name__ == '__main__':
    main()
