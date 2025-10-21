import os
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggingCallback(BaseCallback):
    """
    Logs episode-level metrics to W&B:
    - Total reward
    - Convergence (approx KL)
    - Exploration (entropy)
    - Model optimality
    """
    def __init__(self, agent, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """
        Track rewards for the current episode.
        Only log when an episode ends.
        """
        reward = self.locals.get("rewards", None)
        done = self.locals.get("dones", None)

        # Track rewards
        if reward is not None:
            if isinstance(reward, np.ndarray):  # multiple envs
                self.episode_rewards.extend(reward.tolist())
            else:
                self.episode_rewards.append(float(reward))

        # Check if episode ended
        if done is not None and np.any(done):
            total_reward = sum(self.episode_rewards)
            self.episode_rewards = []  # reset for next episode

            # Compute convergence and exploration
            convergence = self.agent.get_convergence_metrics().get("approx_kl", None)
            exploration = self.agent.get_policy_entropy()
            model_optimality = total_reward

            # Log once per episode
            wandb.log({
                "episode": self.agent.episode,
                "reward": total_reward,
                "convergence": convergence,
                "exploration": exploration,
                "model_optimality": model_optimality
            })

            if self.agent.episode % 50 == 0:
                print(f"Episode {self.agent.episode} reward: {total_reward:.2f}, exploration: {exploration:.4f}")
            self.agent.episode += 1

        return True

    def _on_rollout_start(self) -> None:
        if self.verbose > 0:
            print("Starting new rollout.")

    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            print("Rollout ended.")


class PPOAgent:
    def __init__(
        self,
        env,
        obs_dim,
        n_actions,
        device="cpu",
        lr=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        n_steps=2048,
        batch_size=64,
        seed_n=42
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.ent_coef = ent_coef
        self._dummy_env = DummyVecEnv([lambda: env])
        self.episode = 0
        self._dummy_env = VecNormalize(self._dummy_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.model = PPO(
            "MlpPolicy",
            self._dummy_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            gae_lambda=0.9,
            gamma=0.99,
            n_epochs=20,
            clip_range=0.2,
            seed=seed_n
        )

    def learn(self, total_timesteps, callback=None):
        """
        Train the PPO agent. Defaults to episode-level logging callback.
        """
        callback = EpisodeLoggingCallback(self) if callback is None else callback
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path, env=self._dummy_env, device=self.device)

    def get_policy_entropy(self, n_samples=5):
        """
        Approximate policy entropy by sampling from the environment.

        Notes:
        - For a discrete action space with n actions, the entropy of a uniform 
        policy (i.e., all actions equally likely) is maximal:
            pi = 1 / n_actions
            H_max = -sum_i(pi * log(pi)) = -n_actions * pi * log(pi) = log(n_actions)
        - Example for MountainCar-v0 with 3 actions:
            pi = 1/3
            H_max = -3*(1/3)*log(1/3) = log(3) ≈ 1.0986
        - Therefore, at the start of training, when the policy is nearly random,
        entropy values close to 1.0986 are expected and indicate high exploration.
        """
        if self._dummy_env is None:
            return 0.0
        entropies = []
        for _ in range(n_samples):
            obs = self._dummy_env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                entropies.append(dist.entropy().mean().item())
        return float(np.mean(entropies))

    def get_convergence_metrics(self):
        """Return PPO training metrics."""
        logs = self.model.logger.name_to_value
        return {
            "policy_loss": logs.get("loss/policy_loss", None),
            "value_loss": logs.get("loss/value_loss", None),
            "approx_kl": logs.get("rollout/approx_kl", None),
            "clip_fraction": logs.get("rollout/clip_fraction", None)
        }

    def evaluate_policy(self, n_eval_episodes=5):
        mean_reward, _ = evaluate_policy(self.model, self._dummy_env, n_eval_episodes=n_eval_episodes)
        return mean_reward
