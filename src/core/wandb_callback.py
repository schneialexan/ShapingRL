from stable_baselines3.common.callbacks import BaseCallback
from core.logger import AgentLogger

class WandbCallback(BaseCallback):
    """SB3 callback to log episode stats + shaping influence + SB3 metrics."""
    def __init__(self, run=None, verbose=0):
        super().__init__(verbose)
        self._agent_logger  = AgentLogger(run)
        self.env_rewards = []
        self.shaping_rewards = []
        self.total_rewards = []
        self.nr_episode = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            # Per-step shaping info
            if 'env_reward' in info and 'shaping_reward' in info and 'total_reward' in info:
                self.env_rewards.append(info['env_reward'])
                self.shaping_rewards.append(info['shaping_reward'])
                self.total_rewards.append(info['total_reward'])
                metrics = {
                    "env_reward": self.env_rewards,
                    "shaping_reward": self.shaping_rewards,
                    "total_reward": self.total_rewards
                }
                self._agent_logger.log_metrics(metrics)

            # Episode ended --> log metrics
            if 'episode' in info:
                ep_info = info['episode']
                avg_shaping_reward = (sum(self.shaping_rewards) / len(self.shaping_rewards)) if self.shaping_rewards else 0
                avg_total_reward   = (sum(self.total_rewards) / len(self.total_rewards)) if self.total_rewards else 0
                # SB3 metrics
                sb3_logs = getattr(self.model.logger, 'name_to_value', {})
                metrics = {
                    "episode_reward": ep_info["r"],
                    "episode_length": ep_info["l"],
                    "avg_shaping_reward": avg_shaping_reward,
                    "avg_total_reward": avg_total_reward,
                    "approx_kl": sb3_logs.get("rollout/approx_kl"),
                    "clip_fraction": sb3_logs.get("rollout/clip_fraction"),
                    "entropy_loss": sb3_logs.get("train/entropy_loss"),
                    "policy_gradient_loss": sb3_logs.get("train/policy_gradient_loss"),
                    "value_loss": sb3_logs.get("train/value_loss"),
                    "loss": sb3_logs.get("train/loss"),
                    "explained_variance": sb3_logs.get("train/explained_variance"),
                    "learning_rate": sb3_logs.get("train/learning_rate"),
                    "total_timesteps": sb3_logs.get("time/total_timesteps")
                }

                self._agent_logger.log_episode(self.nr_episode, **metrics)
                self.nr_episode += 1
                # Reset per-episode buffers
                self.env_rewards.clear()
                self.shaping_rewards.clear()
                self.total_rewards.clear()
        return True
