import numpy as np
import random
from collections import defaultdict
import wandb

class QLearningTrainer:
    def __init__(self, env, config, run=None):
        self.env = env
        self.config = config
        self.run = run
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.02)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.99)
        self.total_episodes = config.get('total_episodes', 1000)
        self.n_bins = config.get('n_bins', [18, 14])
        self.obs_space = np.array([[-1.2, 0.6], [-0.07, 0.07]])
        self.bins_list = [np.linspace(self.obs_space[i, 0], self.obs_space[i, 1], self.n_bins[i]-1) 
                          for i in range(len(self.n_bins))]

    def discretize(self, obs):
        return tuple(int(np.digitize(obs[i], self.bins_list[i])) for i in range(len(obs)))

    def train(self):
        for ep in range(self.total_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            s = self.discretize(obs)
            done = False
            ep_reward = 0
            steps = 0
            visited_states = set()
            episode_env_rewards = []
            episode_shaping_rewards = []

            while not done:
                visited_states.add(s)
                # ε-greedy action selection
                if random.random() < self.epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = int(np.argmax(self.q_table[s]))

                step_result = self.env.step(a)
                if len(step_result) == 5:
                    next_obs, r, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, r, done, info = step_result

                # extract shaping_reward safely
                shaping_reward = info.get('shaping_reward', 0) if info else 0
                env_reward = info.get('env_reward', r - shaping_reward) if info else r - shaping_reward
                episode_env_rewards.append(env_reward)
                episode_shaping_rewards.append(shaping_reward)

                ep_reward += r
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]

                s2 = self.discretize(next_obs)
                best_next = np.max(self.q_table[s2])
                self.q_table[s][a] += self.alpha * (r + self.gamma*best_next - self.q_table[s][a])
                s = s2
                steps += 1

            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Compute Q-metrics
            max_q = max([np.max(self.q_table[s]) for s in self.q_table])
            mean_q = np.mean([np.mean(self.q_table[s]) for s in self.q_table])
            visited_best_q = max([np.max(self.q_table[s]) for s in visited_states])

            # Log to WandB
            if self.run:
                wandb.log({
                    'episode': ep,
                    'episode_reward': ep_reward,
                    'episode_length': steps,
                    'epsilon': self.epsilon,
                    'avg_env_reward': np.sum(episode_env_rewards),
                    'avg_shaping_reward': np.sum(episode_shaping_rewards),
                    'avg_total_reward': ep_reward,
                    'max_q': max_q,
                    'mean_q': mean_q,
                    'visited_best_q': visited_best_q
                })

        return self.q_table
