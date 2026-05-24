"""Structured multi-stage computing task model for satellite simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple
import random

import numpy as np


@dataclass
class TaskInfo:
    task_type: int
    current_size_bytes: int
    demand_seq_flops: List[int]
    output_size_seq_bytes: List[int]
    stage_idx: int = 0
    total_stages: int = 1
    destination: str = ""
    is_completed: bool = False
    birth_time: float = 0.0
    last_decision_time: float = 0.0
    last_state: Any = None
    last_action: Any = None
    last_action_mask: Any = None
    fixed_compute_node: Optional[str] = None
    scheme: Optional[str] = None
    deadline: Optional[float] = None
    soft_deadline: Optional[float] = None
    computing_time_sec: float = 0.0
    computing_queue_time_sec: float = 0.0
    transmission_queue_time_sec: float = 0.0
    transmission_time_sec: float = 0.0
    pending_reward: Optional[float] = None
    pending_done: int = 0
    failed: bool = False

    @property
    def hard_deadline(self) -> Optional[float]:
        return self.deadline

    def __post_init__(self):
        if self.total_stages <= 0:
            raise ValueError("total_stages must be positive")
        if len(self.demand_seq_flops) != self.total_stages:
            raise ValueError("demand_seq_flops length must equal total_stages")
        if len(self.output_size_seq_bytes) != self.total_stages:
            raise ValueError("output_size_seq_bytes length must equal total_stages")
        if self.stage_idx < 0 or self.stage_idx > self.total_stages:
            raise ValueError("stage_idx out of range")

    @property
    def current_stage_demand(self) -> int:
        if self.is_completed:
            return 0
        return self.demand_seq_flops[self.stage_idx]

    @property
    def current_stage_output_size(self) -> int:
        if self.is_completed:
            return self.current_size_bytes
        return self.output_size_seq_bytes[self.stage_idx]

    @property
    def remaining_demand(self) -> int:
        if self.is_completed:
            return 0
        return sum(self.demand_seq_flops[self.stage_idx:])

    @property
    def final_output_size(self) -> int:
        return self.output_size_seq_bytes[-1]

    def mission_state_raw(self) -> Tuple[int, int, int, int]:
        """Return unnormalized mission tuple for routing/RL."""
        return (
            self.task_type,
            self.current_size_bytes,
            self.current_stage_demand,
            self.current_stage_output_size,
        )

    def mission_state_normalized(self, max_size: float, computing_ability: float) -> List[float]:
        task_type, size, demand, output_size = self.mission_state_raw()
        return [
            task_type,
            size / max_size,
            demand / computing_ability,
            output_size / max_size,
        ]

    def record_decision(self, now: float, state=None, action=None, action_mask=None):
        self.last_decision_time = now
        if state is not None:
            self.last_state = state
        if action is not None:
            self.last_action = action
        if action_mask is not None:
            self.last_action_mask = np.asarray(action_mask, dtype=np.float32)

    def set_pending_transition(self, reward: float, done: int):
        self.pending_reward = reward
        self.pending_done = int(done)

    def clear_pending_transition(self):
        self.pending_reward = None
        self.pending_done = 0

    def complete_current_stage(self) -> Tuple[int, int, int]:
        """Finish one compute stage and advance pipeline state."""
        if self.is_completed:
            raise RuntimeError("Cannot complete stage on an already completed task")
        demand = self.demand_seq_flops[self.stage_idx]
        old_size = self.current_size_bytes
        new_size = self.output_size_seq_bytes[self.stage_idx]
        self.current_size_bytes = new_size
        self.stage_idx += 1
        if self.stage_idx >= self.total_stages:
            self.is_completed = True
        return old_size, new_size, demand


def build_compute_intensive_task(
    task_type: int,
    destination: str,
    birth_time: float,
    size_range: Sequence[int],
    task_num_stages: Sequence[int],
    computing_demand_factor: Sequence[float],
    size_reduction_factor: Sequence[float],
    final_result_size_range: Sequence[int],
    rng: Optional[random.Random] = None,
    scheme: Optional[str] = None,
) -> TaskInfo:
    rng = rng or random
    num_stages = rng.choice(list(task_num_stages))
    initial_size = int(rng.uniform(size_range[0], size_range[1]))
    current_size = float(initial_size)
    demand_seq: List[int] = []
    output_seq: List[int] = []
    log_min = np.log(size_reduction_factor[0])
    log_max = np.log(size_reduction_factor[1])
    for stage_idx in range(num_stages):
        flops_per_byte = rng.uniform(computing_demand_factor[0], computing_demand_factor[1])
        demand_seq.append(int(flops_per_byte * current_size))
        if stage_idx < num_stages - 1:
            reduction = float(np.exp(rng.uniform(log_min, log_max)))
            current_size /= reduction
            output_seq.append(int(current_size))
        else:
            output_seq.append(int(rng.uniform(final_result_size_range[0], final_result_size_range[1])))
    return TaskInfo(
        task_type=task_type,
        current_size_bytes=initial_size,
        demand_seq_flops=demand_seq,
        output_size_seq_bytes=output_seq,
        stage_idx=0,
        total_stages=num_stages,
        destination=destination,
        is_completed=False,
        birth_time=birth_time,
        last_decision_time=birth_time,
        scheme=scheme,
    )


def build_mixed_tasks_compute_intensive_task(
    destination: str,
    birth_time: float,
    size_range: Sequence[int],
    task_num_stages: Sequence[int],
    computing_demand_factor: Sequence[float],
    size_reduction_factor: Sequence[float],
    final_result_size_range: Sequence[int],
    soft_deadline: float = 4.0,
    rng: Optional[random.Random] = None,
) -> TaskInfo:
    """Compute-intensive task C: multi-stage, soft deadline T_s."""
    task = build_compute_intensive_task(
        task_type=0,
        destination=destination,
        birth_time=birth_time,
        size_range=size_range,
        task_num_stages=task_num_stages,
        computing_demand_factor=computing_demand_factor,
        size_reduction_factor=size_reduction_factor,
        final_result_size_range=final_result_size_range,
        rng=rng,
    )
    task.soft_deadline = soft_deadline
    return task


def build_mixed_tasks_delay_sensitive_task(
    destination: str,
    birth_time: float,
    delay_size_range_mb: Sequence[float] = (15.0, 40.0),
    demand_factor: Sequence[float] = (1200.0, 1800.0),
    hard_deadline: float = 2.0,
    rng: Optional[random.Random] = None,
) -> TaskInfo:
    """Delay-sensitive task D: single stage, hard deadline T_d."""
    rng = rng or random
    size_bytes = int(rng.uniform(delay_size_range_mb[0], delay_size_range_mb[1]) * 1024 * 1024)
    flops_per_byte = rng.uniform(demand_factor[0], demand_factor[1])
    demand = int(flops_per_byte * size_bytes)
    output_size = int(size_bytes * rng.uniform(0.3, 0.6))
    return TaskInfo(
        task_type=1,
        current_size_bytes=size_bytes,
        demand_seq_flops=[demand],
        output_size_seq_bytes=[output_size],
        stage_idx=0,
        total_stages=1,
        destination=destination,
        is_completed=False,
        birth_time=birth_time,
        last_decision_time=birth_time,
        deadline=hard_deadline,
    )


def build_mixed_tasks_task(
      task_type: int,
    destination: str,
    birth_time: float,
    size_range: Sequence[int],
    task_num_stages: Sequence[int],
    computing_demand_factor: Sequence[float],
    size_reduction_factor: Sequence[float],
    final_result_size_range: Sequence[int],
    delay_size_range_mb: Sequence[float] = (15.0, 40.0),
    delay_demand_factor: Sequence[float] = (1200.0, 1800.0),
    soft_deadline_c: float = 4.0,
    hard_deadline_d: float = 2.0,
    rng: Optional[random.Random] = None,
) -> TaskInfo:
    """Build C (type=0) or D (type=1) task for chapter 4."""
    if task_type == 1:
        return build_mixed_tasks_delay_sensitive_task(
            destination=destination,
            birth_time=birth_time,
            delay_size_range_mb=delay_size_range_mb,
            demand_factor=delay_demand_factor,
            hard_deadline=hard_deadline_d,
            rng=rng,
        )
    return build_mixed_tasks_compute_intensive_task(
        destination=destination,
        birth_time=birth_time,
        size_range=size_range,
        task_num_stages=task_num_stages,
        computing_demand_factor=computing_demand_factor,
        size_reduction_factor=size_reduction_factor,
        final_result_size_range=final_result_size_range,
        soft_deadline=soft_deadline_c,
        rng=rng,
    )


def build_legacy_task(
    task_type: int,
    destination: str,
    birth_time: float,
    size_range: Sequence[int],
    computing_demand_factor: Sequence[float],
    computing_demand_factor_2: Sequence[float],
    size_after_computing_factor: Sequence[float],
    size_after_computing_1: int,
    rng: Optional[random.Random] = None,
) -> TaskInfo:
    rng = rng or random
    size = rng.randint(size_range[0], size_range[1])
    if task_type == 0:
        output_size = int(rng.uniform(size_after_computing_factor[0], size_after_computing_factor[1]) * size)
        demand = int(rng.uniform(computing_demand_factor[0], computing_demand_factor[1]) * size)
    else:
        output_size = int(size_after_computing_1)
        demand = int(rng.uniform(computing_demand_factor_2[0], computing_demand_factor_2[1]) * size)
    return TaskInfo(
        task_type=task_type,
        current_size_bytes=size,
        demand_seq_flops=[demand],
        output_size_seq_bytes=[output_size],
        stage_idx=0,
        total_stages=1,
        destination=destination,
        is_completed=False,
        birth_time=birth_time,
        last_decision_time=birth_time,
    )


def attach_task_to_packet(packet, task: TaskInfo):
    packet.task = task
    packet.destination = task.destination
    packet.size = task.current_size_bytes


def sync_packet_size(packet):
    if packet.task is not None:
        packet.size = packet.task.current_size_bytes
