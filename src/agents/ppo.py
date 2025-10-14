import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)


class PPOAgent:
    def __init__(self, obs_dim, n_actions, device='cpu', lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.model = PPOActorCritic(obs_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.model(state_t)
        probs = torch.softmax(logits, dim=-1)
        if evaluate:
            action = int(torch.argmax(probs, dim=-1).item())
            return action
        dist = torch.distributions.Categorical(probs)
        action = int(dist.sample().item())
        return action


    def observe(self, batch):
        # Expect batch to be a list of (state, action, reward, next_state, done, log_prob)
        # For this scaffold we simply do a very small policy gradient update using collected batch.
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = [b[2] for b in batch]
        dones = [b[4] for b in batch]

        # compute returns (simple discounted sum)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        logits, values = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions)
        advantages = returns - values.detach()
        # actor loss (surrogate)
        ratio = torch.exp(action_log_probs - action_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # critic loss
        critic_loss = nn.functional.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))