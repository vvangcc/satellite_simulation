"""Scheme-aware DGR-D3QN policy for Chapter 3 offloading comparison."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from models.action_mask import (
    ACTION_DIM,
    COMPUTE_ACTION,
    build_action_mask,
    has_valid_actions,
    masked_argmax,
    random_valid_action,
)

SCHEMES = frozenset({"GO", "SSO", "MSCO"})


def resolve_decision_target(task, packet, scheme: Optional[str]) -> str:
    """Routing/state target for the next decision (final destination unchanged on packet)."""
    if scheme == "SSO" and task.fixed_compute_node is not None and not task.is_completed:
        return task.fixed_compute_node
    return packet.destination


def _hop_distance(satellite, node_name: str, target_node: str) -> Optional[float]:
    if node_name == target_node:
        return 0.0
    if node_name == satellite.name:
        if hasattr(satellite, "routing_tables") and target_node in satellite.routing_tables:
            return float(satellite.routing_tables[target_node][1])
        return None
    hops_map = getattr(satellite, "neighbor_hops", {})
    if node_name in hops_map and target_node in hops_map[node_name]:
        return float(hops_map[node_name][target_node])
    return None


def build_forward_to_target_mask(
    packet,
    satellite,
    graph,
    target_node: str,
    action_dim: int = ACTION_DIM,
) -> np.ndarray:
    """Keep forward actions that reduce hop distance to target_node; fallback to legal forwards."""
    base = build_action_mask(packet, satellite, graph, action_dim)
    forward_mask = np.zeros(action_dim, dtype=np.float32)
    if target_node is None or target_node == satellite.name:
        forward_mask[: min(4, action_dim - 1)] = base[: min(4, action_dim - 1)]
        return forward_mask

    current_hops = _hop_distance(satellite, satellite.name, target_node)
    closer_found = False
    for action_idx in range(min(4, len(satellite.neighbors), action_dim - 1)):
        if base[action_idx] <= 0.5:
            continue
        neighbor = satellite.neighbors[action_idx]
        neighbor_hops = _hop_distance(satellite, neighbor, target_node)
        if neighbor_hops is not None and current_hops is not None and neighbor_hops < current_hops:
            forward_mask[action_idx] = 1.0
            closer_found = True

    if closer_found:
        return forward_mask

    forward_mask[: min(4, action_dim - 1)] = base[: min(4, action_dim - 1)]
    return forward_mask


def build_scheme_action_mask(
    packet,
    satellite,
    graph,
    scheme: Optional[str],
    action_dim: int = ACTION_DIM,
) -> np.ndarray:
    """Apply offloading-scheme constraints on top of the base DGR-D3QN mask."""
    mask = build_action_mask(packet, satellite, graph, action_dim)
    task = packet.task
    if scheme is None or scheme == "MSCO":
        return mask
    if scheme == "GO":
        mask[COMPUTE_ACTION] = 0.0
        return mask
    if scheme == "SSO":
        if task.is_completed:
            completed_mask = np.zeros(action_dim, dtype=np.float32)
            toward_dest = build_forward_to_target_mask(
                packet, satellite, graph, packet.destination, action_dim
            )
            completed_mask[:4] = toward_dest[:4]
            return completed_mask
        if task.fixed_compute_node is None:
            return mask
        if satellite.name == task.fixed_compute_node:
            fixed_mask = np.zeros(action_dim, dtype=np.float32)
            fixed_mask[COMPUTE_ACTION] = mask[COMPUTE_ACTION]
            return fixed_mask
        off_node_mask = np.zeros(action_dim, dtype=np.float32)
        toward_fixed = build_forward_to_target_mask(
            packet, satellite, graph, task.fixed_compute_node, action_dim
        )
        off_node_mask[:4] = toward_fixed[:4]
        return off_node_mask
    return mask


def bind_sso_compute_node(task, satellite_name: str, scheme: Optional[str]) -> None:
    """Record the first compute satellite for SSO; all stages must stay on it."""
    if scheme == "SSO" and task.fixed_compute_node is None:
        task.fixed_compute_node = satellite_name


def is_go_scheme(scheme: Optional[str], mode: str = "") -> bool:
    return scheme == "GO" or mode == "Ground"


def select_action_with_scheme(
    packet,
    satellite,
    graph,
    scheme: Optional[str],
    agent,
    env,
    *,
    current_state,
    decision_target: Optional[str] = None,
) -> int:
    """Run epsilon-greedy DGR-D3QN with scheme-specific action masking."""
    action_mask = build_scheme_action_mask(packet, satellite, graph, scheme)
    if not has_valid_actions(action_mask):
        return 5

    q_net = agent
    if agent is not None and hasattr(agent, "online_net"):
        q_net = agent.online_net
    elif hasattr(satellite, "q_net"):
        q_net = satellite.q_net

    epsilon = getattr(satellite, "epsilon", 0.0)
    device = getattr(satellite, "device", torch.device("cpu"))
    destination = decision_target if decision_target is not None else resolve_decision_target(
        packet.task, packet, scheme
    )

    if np.random.rand() <= epsilon:
        if np.random.rand() <= 0.5:
            return random_valid_action(action_mask)
        dest_dirs = []
        slots = satellite._neighbor_slots() if hasattr(satellite, "_neighbor_slots") else list(satellite.neighbors)
        max_hop = getattr(satellite, "max_hop", 1) or 1
        for slot_idx, neighbor in enumerate(slots[:4]):
            if neighbor is None:
                dest_dirs.append(2.0)
            elif destination in satellite.neighbor_hops.get(neighbor, {}):
                dest_dirs.append(satellite.neighbor_hops[neighbor][destination] / max_hop)
            elif hasattr(satellite, "routing_tables") and destination in satellite.routing_tables:
                dest_dirs.append(satellite.routing_tables[destination][1] / max_hop)
            else:
                dest_dirs.append(2.0)
        min_value = min(dest_dirs) if dest_dirs else 2.0
        nearest = [
            idx for idx, value in enumerate(dest_dirs)
            if value == min_value and action_mask[idx] > 0.5
        ]
        if nearest:
            return int(np.random.choice(nearest))
        return random_valid_action(action_mask)

    state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(device)
    q_values = q_net(state_tensor, mask_tensor)
    return masked_argmax(q_values[0].detach().cpu().numpy(), action_mask)
