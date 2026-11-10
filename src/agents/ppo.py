from stable_baselines3 import PPO
from core.wandb_callback import WandbCallback

class PPOTrainer:
    def __init__(self, env, config, run=None, shaping="none"):
        self.env = env
        self.config = config
        self.run = run
        self.model = None
        self.shaping = shaping

    def train(self):
        self.model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            gamma=self.config.get('gamma', 0.99),
            clip_range=self.config.get('clip_range', 0.2),
            ent_coef=self.config.get('ent_coef', 0.01),
            verbose=self.config.get('verbose', 1)
        )

        callback = WandbCallback("ppo", run=self.run, shaping=self.shaping)
        self.model.learn(
            total_timesteps=self.config.get('total_timesteps', 100_000),
            callback=callback
        )
        return self.model
