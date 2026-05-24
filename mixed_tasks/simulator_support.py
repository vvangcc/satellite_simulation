"""mixed_tasks simulator helpers."""
from __future__ import annotations

import simpy

from models.mixed_tasks_state_layout import (
    MIXED_TASKS_SOFT_DEADLINE_C,
    TASK_TYPE_COMPUTE_INTENSIVE,
    TASK_TYPE_DELAY_SENSITIVE,
)
from models.mixed_tasks_reward import MixedTasksRewardFunction


def mixed_tasks_statics_defaults() -> dict:
    return {
        "D_total": 0,
        "D_completed": 0,
        "D_on_time": 0,
        "C_total": 0,
        "C_completed": 0,
        "C_soft_deadline": 0,
        "C_total_delay": 0.0,
    }


def build_mixed_tasks_reward(reward_cfg: dict) -> MixedTasksRewardFunction:
    return MixedTasksRewardFunction(
        beta_D_s=reward_cfg.get("beta_D_s", 1.0),
        beta_C_s=reward_cfg.get("beta_C_s", 0.6),
        beta_D_l=reward_cfg.get("beta_D_l", 1.0),
        beta_C_l=reward_cfg.get("beta_C_l", 0.6),
        beta_D_d=reward_cfg.get("beta_D_d", 0.08),
        beta_C_d=reward_cfg.get("beta_C_d", 0.04),
        beta_m=reward_cfg.get("beta_m", 0.25),
        memory_threshold=reward_cfg.get("memory_threshold", 0.1),
    )


def record_task_generated(statics, task):
    if task.task_type == TASK_TYPE_DELAY_SENSITIVE:
        statics["D_total"] += 1
    else:
        statics["C_total"] += 1


def is_d_task(task) -> bool:
    return task.task_type == TASK_TYPE_DELAY_SENSITIVE


def is_c_task(task) -> bool:
    return task.task_type == TASK_TYPE_COMPUTE_INTENSIVE


def d_hard_deadline_exceeded(task, now: float) -> bool:
    if not is_d_task(task) or task.failed:
        return False
    deadline = task.deadline or 2.0
    return (now - task.birth_time) > deadline and not task.is_completed


def within_d_deadline(task, now: float) -> bool:
    deadline = task.deadline or 2.0
    return (now - task.birth_time) <= deadline


def within_c_soft_deadline(task, now: float) -> bool:
    soft = task.soft_deadline or MIXED_TASKS_SOFT_DEADLINE_C
    return (now - task.birth_time) <= soft


def record_success(statics, task, now: float, creation_time: float):
    delay = now - creation_time
    if is_d_task(task):
        statics["D_completed"] += 1
        if within_d_deadline(task, now):
            statics["D_on_time"] += 1
    elif task.is_completed:
        statics["C_completed"] += 1
        statics["C_total_delay"] += delay
        if within_c_soft_deadline(task, now):
            statics["C_soft_deadline"] += 1


def init_mixed_tasks_queues(satellite, env):
    satellite.computing_queue_D = simpy.Store(env)
    satellite.computing_queue_C = simpy.Store(env)
    satellite.computing_queue_size_D = 0
    satellite.computing_queue_size_C = 0
    satellite.transmission_queue_D = {n: simpy.Store(env) for n in satellite.neighbors}
    satellite.transmission_queue_C = {n: simpy.Store(env) for n in satellite.neighbors}
    satellite.transmission_queue_size_D = {n: 0 for n in satellite.neighbors}
    satellite.transmission_queue_size_C = {n: 0 for n in satellite.neighbors}
