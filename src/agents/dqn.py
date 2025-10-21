import random
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
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQNNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device='cpu', lr=1e-3, gamma=0.99,
                 buffer_size=50000, batch_size=64, target_update=1000,
                 min_buffer=1000, hidden_dim=128, epsilon_start=1.0,
                 epsilon_min=0.1,epsilon_decay=0.999995,):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.min_buffer = min_buffer

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Networks
        self.policy_net = DQNNet(obs_dim, n_actions, hidden_dim).to(device)
        self.target_net = DQNNet(obs_dim, n_actions, hidden_dim).to(device)

        # Sync target network initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Internal counters
        self.learn_step_counter = 0

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Episode counter
        self.episode_count = 0

    # ---------- ACTION SELECTION ----------
    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection"""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    # ---------- TRAINING ----------
    def update(self):
        """Perform a DQN update step"""
        if len(self.memory) < self.min_buffer:
            return None  # Not enough samples yet

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Compute current Q values
        curr_Q = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + self.gamma * max_next_Q * (1 - dones)

        # Compute loss
        loss = self.loss_fn(curr_Q, target_Q)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ---------- ENV INTERACTION ----------
    def observe(self, obs, action, reward, next_obs, done):
        """Store transition and update the network."""
        self.memory.push((obs, action, reward, next_obs, done))
        return self.update()

    # ---------- UTILITIES ----------
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_grad_norm(self):
        """Return total gradient norm of policy network (for convergence)."""
        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return float(total_norm ** 0.5)

    @property
    def exploration(self):
        """Return current epsilon (assume you track it externally)."""
        return getattr(self, "_epsilon", 1.0)  # Set _epsilon during training