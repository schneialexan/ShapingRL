import numpy as np
import pickle

class QLearningAgent:
    """Tabular Q-learning agent with state discretization for MountainCar-v0."""
    def __init__(
        self,
        obs_low,
        obs_high,
        n_bins=(60, 60),
        n_actions=3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        lr_init=0.5,
        lr_min=0.01,
        lr_decay=0.9995,
    ):
        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)
        self.n_bins = np.array(n_bins)
        self.n_actions = n_actions

        self.gamma = gamma

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Learning rate
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        # Q-table
        self.q_table = np.zeros((*n_bins, n_actions), dtype=np.float32)

        # Precompute bin widths
        self.bin_width = (self.obs_high - self.obs_low) / self.n_bins

        # Episode counter
        self.episode_count = 0

    def _discretize(self, obs):
        ratios = (obs - self.obs_low) / self.bin_width
        indices = np.clip(ratios.astype(int), 0, self.n_bins - 1)
        return tuple(indices)

    def select_action(self, state, evaluate=False):
        s = self._discretize(state)
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[s]))

    def observe(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        s_next = self._discretize(next_state)

        # Decayed learning rate
        lr = max(self.lr_min, self.lr_init * (self.lr_decay ** self.episode_count))

        td_target = reward + (0 if done else self.gamma * np.max(self.q_table[s_next]))
        td_error = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += lr * td_error

        return abs(td_error)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    # ---------- UTILITIES ----------
    def get_max_q_change(self):
        """Returns max absolute change in Q-table since last episode."""
        # Store previous Q-table snapshot at end of episode
        if not hasattr(self, "_prev_q_table"):
            self._prev_q_table = np.copy(self.q_table)
            return 0.0
        max_change = np.max(np.abs(self.q_table - self._prev_q_table))
        self._prev_q_table = np.copy(self.q_table)
        return float(max_change)