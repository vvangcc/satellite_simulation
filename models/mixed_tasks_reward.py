"""mixed_tasks multi-type task reward function."""
from __future__ import annotations

from models.mixed_tasks_state_layout import (
    MIXED_TASKS_SOFT_DEADLINE_C,
    TASK_TYPE_COMPUTE_INTENSIVE,
    TASK_TYPE_DELAY_SENSITIVE,
)


class MixedTasksRewardFunction:
    def __init__(
        self,
        beta_D_s: float = 1.0,
        beta_C_s: float = 0.6,
        beta_D_l: float = 1.0,
        beta_C_l: float = 0.6,
        beta_D_d: float = 0.08,
        beta_C_d: float = 0.04,
        beta_m: float = 0.25,
        memory_threshold: float = 0.1,
    ):
        self.beta_D_s = beta_D_s
        self.beta_C_s = beta_C_s
        self.beta_D_l = beta_D_l
        self.beta_C_l = beta_C_l
        self.beta_D_d = beta_D_d
        self.beta_C_d = beta_C_d
        self.beta_m = beta_m
        self.memory_threshold = memory_threshold

    def _is_d(self, task) -> bool:
        return task.task_type == TASK_TYPE_DELAY_SENSITIVE

    def step_reward(self, task, step_delay: float, memory_remain: float) -> float:
        """Intermediate transition reward during forwarding/compute decisions."""
        reward = 0.0
        if self._is_d(task):
            reward -= self.beta_D_d * step_delay
        else:
            reward -= self.beta_C_d * step_delay
        if memory_remain < self.memory_threshold:
            reward -= self.beta_m * (self.memory_threshold - memory_remain)
        return reward

    def success_reward(self, task, total_delay: float, within_deadline: bool) -> float:
        if self._is_d(task):
            reward = self.beta_D_s - self.beta_D_d * total_delay
            if not within_deadline:
                reward -= self.beta_D_l
            return reward
        reward = self.beta_C_s - self.beta_C_d * total_delay
        if not within_deadline:
            overrun = max(0.0, total_delay - (task.soft_deadline or MIXED_TASKS_SOFT_DEADLINE_C))
            reward -= 0.5 * self.beta_C_l * (
                overrun / max(task.soft_deadline or MIXED_TASKS_SOFT_DEADLINE_C, 1e-6)
            )
        return reward

    def failure_reward(self, task, total_delay: float) -> float:
        if self._is_d(task):
            return -self.beta_D_l - self.beta_D_d * total_delay
        return -self.beta_C_l - self.beta_C_d * total_delay

    def memory_penalty(self, memory_remain: float) -> float:
        if memory_remain >= self.memory_threshold:
            return 0.0
        return self.beta_m * (self.memory_threshold - memory_remain)

    def reach_reward(self, task, delay: float, within_deadline: bool = True) -> float:
        return self.success_reward(task, delay, within_deadline)

    def normal_reward(self, task, delay: float, memory_remain: float) -> float:
        return self.step_reward(task, delay, memory_remain)

    def loss_reward(self, task, delay: float) -> float:
        return self.failure_reward(task, delay)
