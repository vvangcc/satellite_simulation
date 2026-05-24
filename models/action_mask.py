"""Unified action masking for satellite routing/compute decisions."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from models.state_layout import (
    TASK_IDX_IS_COMPLETED,
    build_action_mask_from_layout,
    compute_dgr_packed_dim,
    task_is_completed_from_state,
    unpack_dgr_state,
)

ACTION_DIM = 5
COMPUTE_ACTION = 4


def _neighbor_satellite(satellite, neighbor, graph):
    if satellite.propagator is not None and neighbor in satellite.propagator.satellites:
        return satellite.propagator.satellites[neighbor]
    return None


def _link_is_active(satellite, neighbor, graph) -> bool:
    if not satellite.active:
        return False
    if neighbor not in satellite.neighbors:
        return False
    if graph is not None and not graph.has_edge(satellite.name, neighbor):
        return False
    if hasattr(satellite, "last_heartbeat") and neighbor in satellite.last_heartbeat:
        elapsed = satellite.env.now - satellite.last_heartbeat[neighbor]
        if elapsed > satellite.heartbeat_timeout:
            return False
    return True


def build_action_mask(packet, satellite, graph, action_dim: int = ACTION_DIM) -> np.ndarray:
    """
    Build legal-action mask for the current satellite decision.

    Actions 0..3: forward to sorted one-hop neighbor (padding slots masked)
    Action 4: local compute (masked when task completed or resources insufficient)
    """
    mask = np.zeros(action_dim, dtype=np.float32)
    task = packet.task
    packet_size = packet.size

    for action_idx in range(min(4, action_dim - 1)):
        if action_idx >= len(satellite.neighbors):
            continue
        neighbor = satellite.neighbors[action_idx]
        if not _link_is_active(satellite, neighbor, graph):
            continue
        neighbor_sat = _neighbor_satellite(satellite, neighbor, graph)
        if neighbor_sat is None or not neighbor_sat.active:
            continue
        if neighbor_sat.current_memory_occupy + packet_size > neighbor_sat.memory:
            continue
        mask[action_idx] = 1.0

    if action_dim > COMPUTE_ACTION and task is not None and not task.is_completed:
        # Packet already occupies local memory after push_forward.
        local_memory_ok = satellite.current_memory_occupy <= satellite.memory
        local_compute_ok = True
        if hasattr(satellite, "computing_remain") and hasattr(satellite, "computing_ability"):
            remaining = (
                satellite.computing_remain / satellite.computing_ability
                - getattr(satellite, "is_computing", False)
                * (satellite.env.now - getattr(satellite, "last_computing_time", 0))
            )
            local_compute_ok = remaining >= 0
        if local_memory_ok and local_compute_ok:
            mask[COMPUTE_ACTION] = 1.0

    return mask


def has_valid_actions(action_mask: np.ndarray) -> bool:
    return bool(np.any(np.asarray(action_mask, dtype=np.float32) > 0.5))


def random_valid_action(action_mask: np.ndarray, rng=None) -> int:
    rng = rng or np.random
    valid = np.flatnonzero(np.asarray(action_mask, dtype=np.float32) > 0.5)
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def masked_argmax(q_values: np.ndarray, action_mask: np.ndarray) -> int:
    q = np.asarray(q_values, dtype=np.float32).reshape(-1)
    mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    valid = np.flatnonzero(mask > 0.5)
    if len(valid) == 0:
        return int(np.argmax(q))
    return int(valid[np.argmax(q[valid])])


def mask_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    valid = action_mask > 0.5
    masked = q_values.masked_fill(~valid, float("-inf"))
    row_valid = valid.any(dim=1, keepdim=True)
    return torch.where(row_valid, masked, torch.zeros_like(masked))


def shuffle_action_mask(mask: np.ndarray, action: int) -> Tuple[np.ndarray, int]:
    """Apply the same neighbor-slot permutation used by ShuffleEx."""
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    forward = mask[:4]
    compute = mask[4] if len(mask) > 4 else 0.0
    indices = np.random.permutation(4)
    new_forward = forward[indices]
    if action < 4:
        new_action = int(np.where(indices == action)[0][0])
    else:
        new_action = action
    new_mask = np.zeros_like(mask)
    new_mask[:4] = new_forward
    if len(new_mask) > 4:
        new_mask[4] = compute
    return new_mask, new_action


def is_legacy_experience(sample) -> bool:
    return len(sample) == 6 and isinstance(sample[1], (int, np.integer, bool))


def parse_experience_batch(batch) -> Tuple[np.ndarray, ...]:
    """Return state, action_mask, action, reward, next_state, next_action_mask, done."""
    if is_legacy_experience(batch[0]):
        state, mark, action, reward, next_state, done = zip(*batch)
        state = np.array(state)
        next_state = np.array(next_state)
        action_mask = _legacy_state_to_mask(state)
        next_action_mask = _legacy_state_to_mask(next_state)
        return state, action_mask, np.array(action), np.array(reward), next_state, next_action_mask, np.array(done)
    if len(batch[0]) == 7:
        state, action_mask, action, reward, next_state, next_action_mask, done = zip(*batch)
        return (
            np.array(state),
            np.array(action_mask, dtype=np.float32),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(next_action_mask, dtype=np.float32),
            np.array(done),
        )
    raise ValueError(f"Unsupported experience tuple length: {len(batch[0])}")


def _legacy_state_to_mask(states):
    """Best-effort fallback for old replay entries that stored integer mark."""
    masks = []
    for state in states:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        if state_arr.shape[0] >= compute_dgr_packed_dim():
            task_state, _, _, _, node_mask = unpack_dgr_state(state_arr)
            mask = build_action_mask_from_layout(node_mask, task_is_completed_from_state(task_state))
        else:
            is_completed = bool(state_arr[TASK_IDX_IS_COMPLETED] > 0.5) if state_arr.shape[0] > TASK_IDX_IS_COMPLETED else bool(state_arr[-1] > 0.5)
            mask[:4] = 1.0
            if not is_completed:
                mask[4] = 1.0
        masks.append(mask)
    return np.asarray(masks, dtype=np.float32)
