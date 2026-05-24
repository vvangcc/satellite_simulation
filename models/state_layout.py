"""State vector layout and automatic dimension helpers for DGR-D3QN."""
from __future__ import annotations

import numpy as np

MAX_NEIGHBOR_SLOTS = 4
MAX_GRAPH_NODES = MAX_NEIGHBOR_SLOTS + 1
MAX_TASK_STAGES = 4
RESOURCE_FEAT_DIM = 3
ACTION_DIM = 5

# S_t = {s_c, d_c, D, S*, g_s, x_c, ...} with fixed-length padding (paper 3.5.1)
TASK_IDX_TYPE = 0
TASK_IDX_SIZE = 1
TASK_IDX_STAGE_DEMAND = 2
TASK_IDX_DEMAND_SEQ = 3
TASK_IDX_OUTPUT_SEQ = 7
TASK_IDX_STAGE = 11
TASK_IDX_TOTAL_STAGES = 12
TASK_IDX_IS_COMPLETED = 13
TASK_IDX_HOPS = 14
TASK_STATE_DIM = 15

DEST_STATE_DIM = MAX_NEIGHBOR_SLOTS
CURRENT_NODE_DIM = 3
PER_NEIGHBOR_FLAT_DIM = 6


def build_task_state_vector(task, max_size: float, computing_ability: float, max_hop: float, hops: int) -> np.ndarray:
    demand_pad = np.zeros(MAX_TASK_STAGES, dtype=np.float32)
    output_pad = np.zeros(MAX_TASK_STAGES, dtype=np.float32)
    for idx in range(min(MAX_TASK_STAGES, task.total_stages)):
        demand_pad[idx] = task.demand_seq_flops[idx] / computing_ability
        output_pad[idx] = task.output_size_seq_bytes[idx] / max_size
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
        ],
        dtype=np.float32,
    )


def task_is_completed_from_state(task_state: np.ndarray) -> bool:
    return bool(np.asarray(task_state, dtype=np.float32).reshape(-1)[TASK_IDX_IS_COMPLETED] > 0.5)


def compute_dgr_packed_dim(
    max_nodes: int = MAX_GRAPH_NODES,
    resource_feat_dim: int = RESOURCE_FEAT_DIM,
) -> int:
    return (
        TASK_STATE_DIM
        + DEST_STATE_DIM
        + max_nodes * resource_feat_dim
        + max_nodes * max_nodes
        + max_nodes
    )


def compute_dqn_flat_state_dim(max_neighbors: int = MAX_NEIGHBOR_SLOTS) -> int:
    """Flat DQN state: neighbor slots + local node + full task state vector."""
    return max_neighbors * PER_NEIGHBOR_FLAT_DIM + CURRENT_NODE_DIM + TASK_STATE_DIM


def pack_dgr_state(
    task_state: np.ndarray,
    dest_state: np.ndarray,
    node_feats: np.ndarray,
    phys_adj: np.ndarray,
    node_mask: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(task_state, dtype=np.float32).reshape(-1),
            np.asarray(dest_state, dtype=np.float32).reshape(-1),
            np.asarray(node_feats, dtype=np.float32).reshape(-1),
            np.asarray(phys_adj, dtype=np.float32).reshape(-1),
            np.asarray(node_mask, dtype=np.float32).reshape(-1),
        ]
    )


def unpack_dgr_state(
    packed: np.ndarray,
    max_nodes: int = MAX_GRAPH_NODES,
    resource_feat_dim: int = RESOURCE_FEAT_DIM,
):
    packed = np.asarray(packed, dtype=np.float32)
    offset = 0
    task_state = packed[offset: offset + TASK_STATE_DIM]
    offset += TASK_STATE_DIM
    dest_state = packed[offset: offset + DEST_STATE_DIM]
    offset += DEST_STATE_DIM
    node_feat_size = max_nodes * resource_feat_dim
    node_feats = packed[offset: offset + node_feat_size].reshape(max_nodes, resource_feat_dim)
    offset += node_feat_size
    adj_size = max_nodes * max_nodes
    phys_adj = packed[offset: offset + adj_size].reshape(max_nodes, max_nodes)
    offset += adj_size
    node_mask = packed[offset: offset + max_nodes]
    return task_state, dest_state, node_feats, phys_adj, node_mask


def build_action_mask_from_layout(node_mask: np.ndarray, is_completed: bool, action_dim: int = ACTION_DIM) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.float32)
    for slot in range(min(MAX_NEIGHBOR_SLOTS, action_dim - 1)):
        if node_mask[slot + 1] > 0.5:
            mask[slot] = 1.0
    if not is_completed and action_dim > MAX_NEIGHBOR_SLOTS:
        mask[MAX_NEIGHBOR_SLOTS] = 1.0
    return mask


def build_action_mask(node_mask: np.ndarray, is_completed: bool, action_dim: int = ACTION_DIM) -> np.ndarray:
    return build_action_mask_from_layout(node_mask, is_completed, action_dim)


def masked_argmax(q_values: np.ndarray, action_mask: np.ndarray) -> int:
    q = np.asarray(q_values, dtype=np.float32).reshape(-1)
    mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    valid = np.where(mask > 0.5)[0]
    if len(valid) == 0:
        return int(np.argmax(q))
    return int(valid[np.argmax(q[valid])])
