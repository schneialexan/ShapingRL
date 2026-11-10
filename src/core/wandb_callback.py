from stable_baselines3.common.callbacks import BaseCallback
from core.logger import AgentLogger
import os

class WandbCallback(BaseCallback):
    """SB3 callback to log episode stats + shaping influence + SB3 metrics."""
    def __init__(self, agent: str, run=None, verbose=0, save_path=".", shaping="none"):
        super().__init__(verbose)
        self._agent_logger  = AgentLogger(run)
        self.env_rewards = []
        self.shaping_rewards = []
        self.total_rewards = []
        self.nr_episode = 0
        self.agent = agent
        self.shaping = shaping
        self.best_reward = -float('inf')
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            # Per-step shaping info
            if 'env_reward' in info and 'shaping_reward' in info and 'total_reward' in info:
                self.env_rewards.append(info['env_reward'])
                self.shaping_rewards.append(info['shaping_reward'])
                self.total_rewards.append(info['total_reward'])
                metrics = {
                    "step_env_reward": self.env_rewards,
                    "step_haping_reward": self.shaping_rewards,
                    "step_total_reward": self.total_rewards
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
                    "episode_avg_shaping_reward": avg_shaping_reward,
                    "episode_avg_total_reward": avg_total_reward,
                    "episode_approx_kl": sb3_logs.get("rollout/approx_kl"),
                    "episode_clip_fraction": sb3_logs.get("rollout/clip_fraction"),
                    "episode_entropy_loss": sb3_logs.get("train/entropy_loss"),
                    "episode_policy_gradient_loss": sb3_logs.get("train/policy_gradient_loss"),
                    "episode_value_loss": sb3_logs.get("train/value_loss"),
                    "episode_loss": sb3_logs.get("train/loss"),
                    "episode_explained_variance": sb3_logs.get("train/explained_variance"),
                    "episode_learning_rate": sb3_logs.get("train/learning_rate"),
                    "episode_total_timesteps": sb3_logs.get("time/total_timesteps")
                }

                self._agent_logger.log_episode(self.nr_episode, **metrics)
                self.nr_episode += 1
                # Reset per-episode buffers
                if ep_info["r"] > self.best_reward:
                    self.best_reward = ep_info["r"]
                    model_file = os.path.join(self.save_path, f"./models/{self.shaping}/best_{self.agent}_model.zip")
                    self.model.save(model_file)
                    if int(self.verbose) > 0:
                        print(f"New best model saved: Episode {self.nr_episode}, Reward {self.best_reward:.2f}")
                self.env_rewards.clear()
                self.shaping_rewards.clear()
                self.total_rewards.clear()
        return True
