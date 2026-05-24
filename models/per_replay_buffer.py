"""Prioritized Experience Replay buffer for PER-D3QN."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class PERReplayBuffer:
    """Proportional PER with importance-sampling weights."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self.buffer: List = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.train_step = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def beta(self) -> float:
        if self.capacity <= 0:
            return self.beta_end
        frac = min(1.0, self.train_step / max(self.capacity, 1))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, experience, priority: Optional[float] = None):
        priority = priority if priority is not None else self.max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max(priority, self.epsilon)
        self.position = (self.position + 1) % self.capacity

    def extend(self, experiences):
        for exp in experiences:
            self.add(exp)

    def _priority_probs(self) -> np.ndarray:
        n = len(self.buffer)
        if n == 0:
            return np.array([], dtype=np.float64)
        prios = self.priorities[:n] ** self.alpha
        total = prios.sum()
        if total <= 0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return prios / total

    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough samples in PER buffer")
        probs = self._priority_probs()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / max(weights.max(), 1e-8)
        batch = [self.buffer[i] for i in indices]
        self.train_step += 1
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(float(td_err)) + self.epsilon) ** self.alpha
            self.priorities[int(idx)] = priority
            self.max_priority = max(self.max_priority, priority)
