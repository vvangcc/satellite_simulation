from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

from models.action_mask import mask_q_values, parse_experience_batch


def get_activation(act_type: str, negative_slope: float = 0.01):
    if act_type == 'LeakyRelu':
        return nn.LeakyReLU(negative_slope=negative_slope)
    elif act_type == 'Relu':
        return nn.ReLU()
    elif act_type == 'PRelu':
        return nn.PReLU()
    else:
        return nn.Identity()


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, activation: str = 'LeakyRelu',
                 hidden_layers: int = 2, dueling=False, scale=1., negative_slope: float = 0.01):
        super(QNetwork, self).__init__()

        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation, negative_slope)
        self.dueling = dueling
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

        self.scale = scale

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.mid_acts = nn.ModuleList([get_activation(activation, negative_slope) for _ in range(hidden_layers)])

    def forward(self, observation):
        x = self.in_layer(observation)
        x = self.act(x)

        for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
            x = mid_layer(x)
            x = mid_act(x)

        if self.dueling:
            value = self.value_stream(x)
            advantages = self.advantage_stream(x)
            x = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            x = self.out_layer(x)

        if self.scale > 1:
            x *= self.scale
        return x


class DDQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None, negative_slope: float = 0.01):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, negative_slope=negative_slope).to(device)
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, negative_slope=negative_slope).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat
        self.last_loss = None

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, action_mask, action, reward, next_state, next_action_mask, done = parse_experience_batch(batch)
            state, action, action_mask = self._shuffle_batch(state, action, action_mask)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action_mask = torch.tensor(action_mask, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state).gather(1, action.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q_online = mask_q_values(self.online_net(next_state), next_action_mask)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_state).gather(1, next_actions).squeeze()
                expected_q = reward + (1 - done) * self.gamma * next_q_target

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.last_loss = float(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        print("Target network updated")

    def save_model(self, file_path):
        if file_path:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))
            self.target_net.load_state_dict(torch.load(file_path))

    def _shuffle_batch(self, states, actions, action_masks=None):
        if not self.shuffle_func:
            return states, actions, action_masks
        shuffled_states = []
        shuffled_actions = []
        shuffled_masks = [] if action_masks is not None else None
        for idx, (state, action) in enumerate(zip(states, actions)):
            mask = None if action_masks is None else action_masks[idx]
            result = self.shuffle_func(state, action, mask)
            if mask is not None:
                state, action, mask = result
                shuffled_masks.append(mask)
            else:
                state, action = result
            shuffled_states.append(state)
            shuffled_actions.append(action)
        if shuffled_masks is not None:
            return np.array(shuffled_states), np.array(shuffled_actions), np.array(shuffled_masks, dtype=np.float32)
        return np.array(shuffled_states), np.array(shuffled_actions), action_masks

    def shuffle(self, experiences):
        return self._shuffle_batch(*experiences)


class DQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None, negative_slope: float = 0.01):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, negative_slope=negative_slope).to(device)
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat
        self.last_loss = None

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, action_mask, action, reward, next_state, next_action_mask, done = parse_experience_batch(batch)
            state, action, action_mask = self._shuffle_batch(state, action, action_mask)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state).gather(1, action.unsqueeze(1)).squeeze()
            next_q = mask_q_values(self.online_net(next_state), next_action_mask).max(dim=1)[0]

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.last_loss = float(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, file_path):
        if file_path:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))

    def target_update(self):
        pass

    def _shuffle_batch(self, states, actions, action_masks=None):
        if not self.shuffle_func:
            return states, actions, action_masks
        shuffled_states = []
        shuffled_actions = []
        shuffled_masks = [] if action_masks is not None else None
        for idx, (state, action) in enumerate(zip(states, actions)):
            mask = None if action_masks is None else action_masks[idx]
            result = self.shuffle_func(state, action, mask)
            if mask is not None:
                state, action, mask = result
                shuffled_masks.append(mask)
            else:
                state, action = result
            shuffled_states.append(state)
            shuffled_actions.append(action)
        if shuffled_masks is not None:
            return np.array(shuffled_states), np.array(shuffled_actions), np.array(shuffled_masks, dtype=np.float32)
        return np.array(shuffled_states), np.array(shuffled_actions), action_masks

    def shuffle(self, experiences):
        return self._shuffle_batch(*experiences)


class PPO_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, scale=1e2).to(
            device)
        self.critic_net = QNetwork(state_dim, hidden_dim, 1, activation, hidden_layers, dueling).to(device)

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer_actor = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

        self.eps_clip = 0.1
        self.max_grad_norm = 0.5

    def update(self, experiences):

        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]
            state, action_mask, action_info, reward, next_state, next_action_mask, done = parse_experience_batch(batch)
            action, old_log_prob = [a[0] for a in action_info], [a[1] for a in action_info]
            state, action, action_mask = self._shuffle_batch(state, action, action_mask)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            action_mask = torch.tensor(action_mask, dtype=torch.float).to(self.device)
            old_log_prob = torch.tensor(old_log_prob, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            with torch.no_grad():
                next_state = self.critic_net(next_state).squeeze()

            action_logits = mask_q_values(self.online_net(state), action_mask)
            action_prob = torch.nn.functional.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action_log_prob = dist.log_prob(action)

            state_value = self.critic_net(state).squeeze()

            advantages = reward + self.gamma * next_state * (1 - done) - state_value.detach()
            ratios = torch.exp(action_log_prob - old_log_prob.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            critic_loss = nn.functional.mse_loss(state_value, reward + self.gamma * next_state * (1 - done))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.online_net.state_dict(), os.path.join(file_path, 'actor.pth'))
        torch.save(self.critic_net.state_dict(), os.path.join(file_path, 'critic.pth'))

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(os.path.join(file_path, 'actor.pth'), map_location=self.device))
            self.critic_net.load_state_dict(torch.load(os.path.join(file_path, 'critic.pth'), map_location=self.device))

    def _shuffle_batch(self, states, actions, action_masks=None):
        if not self.shuffle_func:
            return states, actions, action_masks
        shuffled_states = []
        shuffled_actions = []
        shuffled_masks = [] if action_masks is not None else None
        for idx, (state, action) in enumerate(zip(states, actions)):
            mask = None if action_masks is None else action_masks[idx]
            result = self.shuffle_func(state, action, mask)
            if mask is not None:
                state, action, mask = result
                shuffled_masks.append(mask)
            else:
                state, action = result
            shuffled_states.append(state)
            shuffled_actions.append(action)
        if shuffled_masks is not None:
            return np.array(shuffled_states), np.array(shuffled_actions), np.array(shuffled_masks, dtype=np.float32)
        return np.array(shuffled_states), np.array(shuffled_actions), action_masks

    def shuffle(self, experiences):
        return self._shuffle_batch(*experiences)


class SAC_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 learning_rate: float = 2e-4, policy_lr: float = 2e-4, repeat: int = 1, tau: float = 5e-3,
                 alpha: float = 0.2, automatic_entropy_tuning: bool = True, target_entropy: float = None,
                 shuffle_func=None):
        self.device = device
        self.actor = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling=False).to(device)
        self.critic_1 = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling=False).to(device)
        self.critic_2 = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling=False).to(device)
        self.target_critic_1 = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers,
                                        dueling=False).to(device)
        self.target_critic_2 = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers,
                                        dueling=False).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.repeat = repeat
        self.tau = tau
        self.shuffle_func = shuffle_func

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=policy_lr)
        else:
            self.alpha = alpha

    @property
    def alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = torch.tensor(value, device=self.device)

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(file_path, 'actor.pth'))
        torch.save(self.critic_1.state_dict(), os.path.join(file_path, 'critic1.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(file_path, 'critic2.pth'))
        torch.save(self.target_critic_1.state_dict(), os.path.join(file_path, 'target_critic1.pth'))
        torch.save(self.target_critic_2.state_dict(), os.path.join(file_path, 'target_critic2.pth'))
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha.detach(), os.path.join(file_path, 'log_alpha.pt'))
        else:
            torch.save(self.alpha.detach(), os.path.join(file_path, 'alpha.pt'))

    def load_model(self, file_path):
        if not file_path:
            return
        self.actor.load_state_dict(torch.load(os.path.join(file_path, 'actor.pth'), map_location=self.device))
        self.critic_1.load_state_dict(torch.load(os.path.join(file_path, 'critic1.pth'), map_location=self.device))
        self.critic_2.load_state_dict(torch.load(os.path.join(file_path, 'critic2.pth'), map_location=self.device))
        self.target_critic_1.load_state_dict(
            torch.load(os.path.join(file_path, 'target_critic1.pth'), map_location=self.device))
        self.target_critic_2.load_state_dict(
            torch.load(os.path.join(file_path, 'target_critic2.pth'), map_location=self.device))
        alpha_path = os.path.join(file_path, 'log_alpha.pt' if self.automatic_entropy_tuning else 'alpha.pt')
        if os.path.exists(alpha_path):
            if self.automatic_entropy_tuning:
                self.log_alpha = torch.load(alpha_path, map_location=self.device)
                self.log_alpha.requires_grad_(True)
            else:
                self.alpha = torch.load(alpha_path, map_location=self.device)

    def _mask_logits(self, logits, action_mask):
        return mask_q_values(logits, action_mask)

    def _soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, experiences):
        self.replay_buffer.extend(experiences)
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, action_mask, action_info, reward, next_state, next_action_mask, done = parse_experience_batch(batch)
            if isinstance(action_info[0], (list, tuple)):
                action, old_log_prob = zip(*action_info)
            else:
                raise ValueError("SAC expects (action, log_prob) tuples in experience.")

            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
            action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

            # Critic update
            with torch.no_grad():
                next_logits = self._mask_logits(self.actor(next_state), next_action_mask)
                next_probs = F.softmax(next_logits, dim=-1)
                next_dist = torch.distributions.Categorical(next_probs)
                next_actions = next_dist.sample()
                next_log_probs = next_dist.log_prob(next_actions)

                q1_next = self.target_critic_1(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q2_next = self.target_critic_2(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                min_q_next = torch.min(q1_next, q2_next)
                target_q = reward + (1 - done) * self.gamma * (min_q_next - self.alpha.detach() * next_log_probs)

            current_q1 = self.critic_1(state).gather(1, action.unsqueeze(1)).squeeze(1)
            current_q2 = self.critic_2(state).gather(1, action.unsqueeze(1)).squeeze(1)
            critic_1_loss = F.mse_loss(current_q1, target_q.detach())
            critic_2_loss = F.mse_loss(current_q2, target_q.detach())

            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # Actor update
            logits = self._mask_logits(self.actor(state), action_mask)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled_actions = dist.sample()
            log_probs = dist.log_prob(sampled_actions)

            q1 = self.critic_1(state).gather(1, sampled_actions.unsqueeze(1)).squeeze(1)
            q2 = self.critic_2(state).gather(1, sampled_actions.unsqueeze(1)).squeeze(1)
            min_q = torch.min(q1, q2)
            actor_loss = (self.alpha.detach() * log_probs - min_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Temperature (entropy) update
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # Soft update targets
            self._soft_update(self.target_critic_1, self.critic_1)
            self._soft_update(self.target_critic_2, self.critic_2)

    def target_update(self):
        # SAC 采用软更新，外部无需再调用
        pass

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


def shuffle_neighbors(neighbor_states, other_states, action, action_mask=None):
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    if action < 4:
        new_action = int(np.where(indices == action)[0])
    else:
        new_action = action
    new_state = np.concatenate([new_state, other_states])
    if action_mask is None:
        return new_state, new_action
    forward_mask = np.asarray(action_mask[:4], dtype=np.float32)
    compute_mask = float(action_mask[4]) if len(action_mask) > 4 else 0.0
    new_mask = np.zeros_like(action_mask, dtype=np.float32)
    new_mask[:4] = forward_mask[indices]
    if len(new_mask) > 4:
        new_mask[4] = compute_mask
    return new_state, new_action, new_mask


class ShuffleEx:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, state, action, action_mask=None):
        return shuffle_neighbors(state[:self.shuffle_mask], state[self.shuffle_mask:], action, action_mask)


def cal_agent_dim(neighbors_dim: int, edges_dim: int, distance_dim: int, mission_dim: int, current_dim: int,
                  action_dim: int):
        return neighbors_dim + edges_dim + distance_dim + mission_dim + current_dim, action_dim, -(
                mission_dim + current_dim)