import numpy as np
import random


class QLearningAgent:
    """Tabular Q-learning agent with state discretization.
    API:
    - .select_action(state, evaluate=False)
    - .observe(state, action, reward, next_state, done)
    - .save(path), .load(path)
    """
    def __init__(self, obs_space_low, obs_space_high, n_bins=(20, 20), n_actions=3, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.obs_low = np.array(obs_space_low)
        self.obs_high = np.array(obs_space_high)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # create bins for discretization
        self.bins = [np.linspace(self.obs_low[i], self.obs_high[i], self.n_bins[i]-1) for i in range(len(self.n_bins))]
        shape = tuple(self.n_bins) + (self.n_actions,)
        self.Q = np.zeros(shape, dtype=np.float32)


    def _discretize(self, obs):
        idx = []
        for i, val in enumerate(obs):
            b = np.digitize(val, self.bins[i])
            idx.append(b)
        return tuple(idx)


    def select_action(self, state, evaluate=False):
        s = self._discretize(state)
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return int(np.argmax(self.Q[s]))


    def observe(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        s2 = self._discretize(next_state)
        best_next = np.max(self.Q[s2])
        td_target = reward + (0 if done else self.gamma * best_next)
        td_error = td_target - self.Q[s + (action,)]
        self.Q[s + (action,)] += self.lr * td_error

        if done:
            # decay epsilon at episode end (caller should ensure this is called per-step too)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path + '.npy')