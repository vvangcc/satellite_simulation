import unittest

import numpy as np
import torch

from Base_Agents import DDQN_Agent


class DDQNTargetNetTests(unittest.TestCase):
    def test_update_uses_target_net_for_bootstrap(self):
        state_dim = 8
        action_dim = 5
        agent = DDQN_Agent(
            state_dim=state_dim,
            hidden_dim=16,
            action_dim=action_dim,
            buffer_length=100,
            batch_size=4,
            gamma=0.99,
            device=torch.device("cpu"),
            q_mask=4,
        )
        target_calls = {"count": 0}
        online_calls = {"count": 0}
        orig_target_forward = agent.target_net.forward
        orig_online_forward = agent.online_net.forward

        def target_forward(obs):
            target_calls["count"] += 1
            return orig_target_forward(obs)

        def online_forward(obs):
            online_calls["count"] += 1
            return orig_online_forward(obs)

        agent.target_net.forward = target_forward
        agent.online_net.forward = online_forward

        state = np.random.rand(state_dim).astype(np.float32)
        action_mask = np.array([1, 1, 0, 0, 1], dtype=np.float32)
        next_action_mask = np.array([1, 0, 0, 0, 0], dtype=np.float32)
        experiences = [
            [state, action_mask, 0, 1.0, state, next_action_mask, 0]
            for _ in range(8)
        ]
        agent.update(experiences)

        self.assertGreater(online_calls["count"], 0)
        self.assertGreater(target_calls["count"], 0)


if __name__ == "__main__":
    unittest.main()
