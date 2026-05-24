"""mixed_tasks state layout: o_i^n(t) = (S_k, S_q, S_r, S_g)."""
from __future__ import annotations

import numpy as np

from models.state_layout import (
    ACTION_DIM,
    CURRENT_NODE_DIM,
    DEST_STATE_DIM,
    MAX_GRAPH_NODES,
    MAX_NEIGHBOR_SLOTS,
    MAX_TASK_STAGES,
    PER_NEIGHBOR_FLAT_DIM,
    RESOURCE_FEAT_DIM,
    TASK_IDX_IS_COMPLETED,
)

MIXED_TASKS_PROFILE = "mixed_tasks"
TASK_TYPE_COMPUTE_INTENSIVE = 0  # C
TASK_TYPE_DELAY_SENSITIVE = 1  # D
MIXED_TASKS_HARD_DEADLINE_D = 2.0
MIXED_TASKS_SOFT_DEADLINE_C = 4.0

MIXED_TASKS_RL_MODES = frozenset({"CH4_D3QN", "CH4_PER_D3QN", "CH4_PPO"})

MIXED_TASKS_TASK_IDX_DELTA = 15
MIXED_TASKS_TASK_STATE_DIM = 16

MIXED_TASKS_QUEUE_STATE_DIM = 4

MIXED_TASKS_QUEUE_IDX_C_D = 0
MIXED_TASKS_QUEUE_IDX_C_C = 1
MIXED_TASKS_QUEUE_IDX_T_D = 2
MIXED_TASKS_QUEUE_IDX_T_C = 3


def is_mixed_tasks_profile(task_profile: str) -> bool:
    return task_profile == MIXED_TASKS_PROFILE


def is_mixed_tasks_rl_mode(mode: str) -> bool:
    return mode in MIXED_TASKS_RL_MODES


def estimate_remaining_time(task, satellite, destination: str, hops: int) -> float:
    """Rough upper-bound estimate for remaining service time (seconds)."""
    remaining_compute = 0.0
    if not task.is_completed:
        remaining_compute = task.remaining_demand / max(satellite.computing_ability, 1.0)
    hop_delay = 0.0
    if destination in satellite.routing_tables:
        hop_delay = satellite.routing_tables[destination][1] * 0.05
    queue_delay = (
        getattr(satellite, "computing_queue_size_D", 0)
        + getattr(satellite, "computing_queue_size_C", 0)
    ) / max(satellite.computing_ability, 1.0) * 0.5
    tx_delay = (
        sum(getattr(satellite, "transmission_queue_size_D", {}).values())
        + sum(getattr(satellite, "transmission_queue_size_C", {}).values())
    ) / max(satellite.transmission_rate, 1.0) * 8.0
    return remaining_compute + hop_delay + queue_delay + tx_delay


def compute_delta(task, satellite, destination: str, hops: int) -> float:
    elapsed = satellite.env.now - task.birth_time
    if task.task_type == TASK_TYPE_DELAY_SENSITIVE:
        deadline = task.hard_deadline or MIXED_TASKS_HARD_DEADLINE_D
    else:
        deadline = task.soft_deadline or MIXED_TASKS_SOFT_DEADLINE_C
    remaining = estimate_remaining_time(task, satellite, destination, hops)
    return deadline - elapsed - remaining


def build_mixed_tasks_task_state_vector(
    task,
    max_size: float,
    computing_ability: float,
    max_hop: float,
    hops: int,
    satellite,
    destination: str,
) -> np.ndarray:
    demand_pad = np.zeros(MAX_TASK_STAGES, dtype=np.float32)
    output_pad = np.zeros(MAX_TASK_STAGES, dtype=np.float32)
    for idx in range(min(MAX_TASK_STAGES, task.total_stages)):
        demand_pad[idx] = task.demand_seq_flops[idx] / computing_ability
        output_pad[idx] = task.output_size_seq_bytes[idx] / max_size
    if task.task_type == TASK_TYPE_DELAY_SENSITIVE:
        deadline = task.hard_deadline or MIXED_TASKS_HARD_DEADLINE_D
    else:
        deadline = task.soft_deadline or MIXED_TASKS_SOFT_DEADLINE_C
    delta_raw = compute_delta(task, satellite, destination, hops)
    delta_norm = float(np.clip(delta_raw / max(deadline, 1e-6), -2.0, 2.0))
    return np.array(
        [
            float(task.task_type),
            task.current_size_bytes / max_size,
            task.current_stage_demand / computing_ability,
            *demand_pad.tolist(),
            *output_pad.tolist(),
            task.stage_idx / max(MAX_TASK_STAGES, 1),
            task.total_stages / max(MAX_TASK_STAGES, 1),
            float(task.is_completed),
            hops / max_hop,
            delta_norm,
        ],
        dtype=np.float32,
    )


def build_mixed_tasks_queue_state_vector(satellite) -> np.ndarray:
    mem = max(float(satellite.memory), 1.0)
    q_c_d = min(getattr(satellite, "computing_queue_size_D", 0) / mem, 1.0)
    q_c_c = min(getattr(satellite, "computing_queue_size_C", 0) / mem, 1.0)
    tx_d = sum(getattr(satellite, "transmission_queue_size_D", {}).values())
    tx_c = sum(getattr(satellite, "transmission_queue_size_C", {}).values())
    q_t_d = min(tx_d / mem, 1.0)
    q_t_c = min(tx_c / mem, 1.0)
    return np.array([q_c_d, q_c_c, q_t_d, q_t_c], dtype=np.float32)


def compute_mixed_tasks_packed_dim(
    max_nodes: int = MAX_GRAPH_NODES,
    resource_feat_dim: int = RESOURCE_FEAT_DIM,
) -> int:
    return (
        MIXED_TASKS_TASK_STATE_DIM
        + MIXED_TASKS_QUEUE_STATE_DIM
        + DEST_STATE_DIM
        + max_nodes * resource_feat_dim
        + max_nodes * max_nodes
        + max_nodes
    )


def compute_mixed_tasks_ppo_flat_state_dim(max_neighbors: int = MAX_NEIGHBOR_SLOTS) -> int:
    return (
        max_neighbors * PER_NEIGHBOR_FLAT_DIM
        + CURRENT_NODE_DIM
        + MIXED_TASKS_TASK_STATE_DIM
        + MIXED_TASKS_QUEUE_STATE_DIM
    )


def pack_mixed_tasks_state(
    task_state: np.ndarray,
    queue_state: np.ndarray,
    dest_state: np.ndarray,
    node_feats: np.ndarray,
    phys_adj: np.ndarray,
    node_mask: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(task_state, dtype=np.float32).reshape(-1),
            np.asarray(queue_state, dtype=np.float32).reshape(-1),
            np.asarray(dest_state, dtype=np.float32).reshape(-1),
            np.asarray(node_feats, dtype=np.float32).reshape(-1),
            np.asarray(phys_adj, dtype=np.float32).reshape(-1),
            np.asarray(node_mask, dtype=np.float32).reshape(-1),
        ]
    )


def unpack_mixed_tasks_state(
    packed: np.ndarray,
    max_nodes: int = MAX_GRAPH_NODES,
    resource_feat_dim: int = RESOURCE_FEAT_DIM,
):
    packed = np.asarray(packed, dtype=np.float32)
    offset = 0
    task_state = packed[offset: offset + MIXED_TASKS_TASK_STATE_DIM]
    offset += MIXED_TASKS_TASK_STATE_DIM
    queue_state = packed[offset: offset + MIXED_TASKS_QUEUE_STATE_DIM]
    offset += MIXED_TASKS_QUEUE_STATE_DIM
    dest_state = packed[offset: offset + DEST_STATE_DIM]
    offset += DEST_STATE_DIM
    node_feat_size = max_nodes * resource_feat_dim
    node_feats = packed[offset: offset + node_feat_size].reshape(max_nodes, resource_feat_dim)
    offset += node_feat_size
    adj_size = max_nodes * max_nodes
    phys_adj = packed[offset: offset + adj_size].reshape(max_nodes, max_nodes)
    offset += adj_size
    node_mask = packed[offset: offset + max_nodes]
    return task_state, queue_state, dest_state, node_feats, phys_adj, node_mask


def task_is_completed_from_mixed_tasks_state(task_state: np.ndarray) -> bool:
    return bool(np.asarray(task_state, dtype=np.float32).reshape(-1)[TASK_IDX_IS_COMPLETED] > 0.5)


def build_mixed_tasks_action_mask_from_layout(
    node_mask: np.ndarray,
    is_completed: bool,
    action_dim: int = ACTION_DIM,
) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.float32)
    for slot in range(min(MAX_NEIGHBOR_SLOTS, action_dim - 1)):
        if node_mask[slot + 1] > 0.5:
            mask[slot] = 1.0
    if not is_completed and action_dim > MAX_NEIGHBOR_SLOTS:
        mask[MAX_NEIGHBOR_SLOTS] = 1.0
    return mask
