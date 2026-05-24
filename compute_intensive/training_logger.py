"""Structured training logs with moving averages for compute_intensive RL runs."""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, Optional


def moving_average(values: Deque[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def new_training_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TrainingLogger:
    def __init__(
        self,
        log_path: str,
        *,
        window: int = 50,
        target_task_rate: Optional[float] = None,
        run_id: Optional[str] = None,
        method_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self.log_path = log_path
        self.window = window
        self.target_task_rate = target_task_rate
        self.run_id = run_id or new_training_run_id()
        self.method_name = method_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.print_fn = print_fn

        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._rewards: Deque[Optional[float]] = deque(maxlen=window)
        self._loss_rates: Deque[Optional[float]] = deque(maxlen=window)
        self._delays: Deque[Optional[float]] = deque(maxlen=window)
        self._hops: Deque[Optional[float]] = deque(maxlen=window)
        self.best_moving_avg_reward: Optional[float] = None

    def log_interval(self, record: Dict[str, object]) -> None:
        episode_reward = record.get("episode_reward")
        packet_loss_rate = record.get("packet_loss_rate")
        average_delay = record.get("average_delay")
        average_hops = record.get("average_hops")

        self._rewards.append(episode_reward if episode_reward is not None else None)
        self._loss_rates.append(packet_loss_rate if packet_loss_rate is not None else None)
        self._delays.append(average_delay if average_delay is not None else None)
        self._hops.append(average_hops if average_hops is not None else None)

        moving_avg_reward = moving_average(self._rewards)
        moving_avg_loss = moving_average(self._loss_rates)
        moving_avg_delay = moving_average(self._delays)
        moving_avg_hops = moving_average(self._hops)

        payload = {
            "run_id": self.run_id,
            "method_name": self.method_name,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            **record,
            f"moving_avg_reward_{self.window}": moving_avg_reward,
            f"moving_avg_packet_loss_{self.window}": moving_avg_loss,
            f"moving_avg_delay_{self.window}": moving_avg_delay,
            f"moving_avg_hops_{self.window}": moving_avg_hops,
            "Average ending reward": episode_reward,
        }

        with open(self.log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._print_summary(payload)
        self._check_task_rate(record.get("actual_task_rate"))

        if moving_avg_reward is not None:
            if self.best_moving_avg_reward is None or moving_avg_reward > self.best_moving_avg_reward:
                self.best_moving_avg_reward = moving_avg_reward

    def _check_task_rate(self, actual_task_rate: object) -> None:
        if self.target_task_rate is None or actual_task_rate is None:
            return
        target = float(self.target_task_rate)
        actual = float(actual_task_rate)
        if target <= 0:
            return
        deviation = abs(actual - target) / target
        if deviation > 0.20:
            self.print_fn(
                "WARNING: actual_task_rate deviates from target_task_rate_per_sec. "
                f"target={target:.3f}, actual={actual:.3f}"
            )

    def _print_summary(self, payload: Dict[str, object]) -> None:
        w = self.window
        parts = [
            f"run_id={payload.get('run_id')}",
            f"current_step={payload.get('current_step')}",
            f"epsilon={payload.get('epsilon')}",
            f"episode_reward={payload.get('episode_reward')}",
            f"moving_avg_reward_{w}={payload.get(f'moving_avg_reward_{w}')}",
            f"packet_loss_rate={payload.get('packet_loss_rate')}",
            f"moving_avg_packet_loss_{w}={payload.get(f'moving_avg_packet_loss_{w}')}",
            f"average_delay={payload.get('average_delay')}",
            f"moving_avg_delay_{w}={payload.get(f'moving_avg_delay_{w}')}",
            f"average_hops={payload.get('average_hops')}",
            f"moving_avg_hops_{w}={payload.get(f'moving_avg_hops_{w}')}",
            f"replay_buffer_size={payload.get('replay_buffer_size')}",
            f"loss={payload.get('loss')}",
            f"actual_task_rate={payload.get('actual_task_rate')}",
        ]
        self.print_fn(" | ".join(str(part) for part in parts))
