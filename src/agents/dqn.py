import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from src.core.replay_buffer import ReplayBuffer


class DQNNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )


    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, device='cpu', lr=1e-3, gamma=0.99, buffer_size=50000, batch_size=64, target_update=1000, min_buffer=1000):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.min_buffer = min_buffer

        self.policy_net = DQNNet(obs_dim, n_actions).to(device)
        self.target_net = DQNNet(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_size)
        self.step_count = 0

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995


    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.policy_net(state)
        return int(torch.argmax(q, dim=1).item())


    def observe(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
        self.step_count += 1

        # decay epsilon slowly
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.replay) < self.min_buffer:
            return None

        # sample and learn
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            td_target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_net.load_state_dict(self.policy_net.state_dict())