"""Checkpoint save/load helpers for compute_intensive RL training."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch


def agent_has_target_net(agent) -> bool:
    return hasattr(agent, "target_net") and getattr(agent, "target_net") is not None


def checkpoint_stem_from_model_path(model_path: str) -> str:
    base = os.path.basename(model_path)
    if base.endswith(".pth"):
        return base[:-4]
    return base or "train_model"


def build_checkpoint_path(
    checkpoint_dir: str,
    stem: str,
    global_step: int,
    *,
    interrupt: bool = False,
) -> str:
    suffix = f"interrupt_step_{global_step}" if interrupt else f"step_{global_step}"
    return os.path.join(checkpoint_dir, f"{stem}_{suffix}.pth")


def save_agent_checkpoint(
    agent,
    path: str,
    *,
    epsilon: float,
    global_step: int,
    episode_reward_list: List[float],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload: Dict[str, Any] = {
        "online_net": agent.online_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": float(epsilon),
        "global_step": int(global_step),
        "episode_reward_list": list(episode_reward_list),
        "replay_buffer_saved": False,
    }
    if agent_has_target_net(agent):
        payload["target_net"] = agent.target_net.state_dict()
    torch.save(payload, path)


def load_agent_checkpoint(agent, path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location=agent.device)

    if isinstance(payload, dict) and "online_net" in payload:
        agent.online_net.load_state_dict(payload["online_net"])
        if "target_net" in payload and agent_has_target_net(agent):
            agent.target_net.load_state_dict(payload["target_net"])
        elif agent_has_target_net(agent):
            agent.target_net.load_state_dict(payload["online_net"])
        if "optimizer" in payload:
            agent.optimizer.load_state_dict(payload["optimizer"])
        return {
            "epsilon": float(payload.get("epsilon", 0.5)),
            "global_step": int(payload.get("global_step", 0)),
            "episode_reward_list": list(payload.get("episode_reward_list", [])),
            "replay_buffer_restored": bool(payload.get("replay_buffer_saved", False)),
        }

    # Backward compatibility: plain state_dict checkpoints from save_model().
    agent.online_net.load_state_dict(payload)
    if agent_has_target_net(agent):
        agent.target_net.load_state_dict(payload)
    return {
        "epsilon": 0.5,
        "global_step": 0,
        "episode_reward_list": [],
        "replay_buffer_restored": False,
    }


def best_model_path_from_model_path(model_path: str) -> str:
    parent, stem = os.path.split(model_path)
    if stem.endswith(".pth"):
        stem = stem[:-4]
    if stem.startswith("train_"):
        best_stem = f"best_{stem[len('train_'):]}"
    else:
        best_stem = f"best_{stem}"
    return os.path.join(parent, f"{best_stem}.pth")


def save_inference_model(agent, path: Optional[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(agent.online_net.state_dict(), path)
