from stable_baselines3 import DQN
from core.wandb_callback import WandbCallback

class DQNTrainer:
    def __init__(self, env, config, run=None, device='cpu'):
        self.env = env
        self.config = config
        self.run = run
        self.device = device
        self.model = None

    def train(self):
        self.model = DQN(
            'MlpPolicy',
            self.env,
            device=self.device,
            learning_rate=self.config.get('learning_rate', 1e-3),
            buffer_size=self.config.get('buffer_size', 50_000),
            learning_starts=self.config.get('learning_starts', 1_000),
            batch_size=self.config.get('batch_size', 64),
            gamma=self.config.get('gamma', 0.99),
            exploration_fraction=self.config.get('exploration_fraction', 0.1),
            exploration_final_eps=self.config.get('exploration_final_eps', 0.05),
            target_update_interval=self.config.get('target_update_interval', 500),
            verbose=self.config.get('verbose', 1)
        )

        callback = WandbCallback(self.run)
        self.model.learn(
            total_timesteps=self.config.get('total_timesteps', 200_000),
            callback=callback
        )
        return self.model
