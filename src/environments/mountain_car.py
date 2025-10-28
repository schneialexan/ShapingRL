import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
import sys
import os

# Add parent folder to sys.path (so imports work)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CONFIG
from shaping.naive_dense import NaiveDenseWrapper
from shaping.potential_based import PotentialBasedWrapper
from agents.ppo import PPOTrainer
from agents.dqn import DQNTrainer
from agents.q_learning import QLearningTrainer

def make_env(shaping='none'):
    env = gym.make('MountainCar-v0')
    # RecordEpisodeStatistics gives ep_rew_mean and ep_len_mean
    env = RecordEpisodeStatistics(env)
    # Stable Baselines3 Monitor wrapper
    env = Monitor(env)
    # Apply shaping wrapper
    if shaping == 'naive':
        env = NaiveDenseWrapper(env)
    elif shaping == 'potential':
        env = PotentialBasedWrapper(env)
    return env

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['ppo', 'dqn', 'q_learning'], default='ppo')
    parser.add_argument('--shaping', choices=['none', 'naive', 'potential'], default='potential')
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--run-name', type=str, default='default_run')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args() if args is None else args

    # Get config for the selected agent
    cfg = CONFIG.get(args.agent, {})
    if args.timesteps and 'total_timesteps' in cfg:
        cfg['total_timesteps'] = args.timesteps
        
    # Automatically set run name
    if args.run_name is None or args.run_name.lower() == 'default_run':
        run_name = f"{args.agent}_{args.shaping}"
    else:
        run_name = args.run_name

    # Initialize Weights & Biases run
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=CONFIG['wandb']['project'],
            name=run_name,
            config={'agent': args.agent, 'shaping': args.shaping, **cfg}
        )

    # Environment setup
    env = make_env(args.shaping)

    # Wrap in VecEnv for SB3
    vec_env = DummyVecEnv([lambda: env])

    if args.agent == 'ppo':
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        trainer = PPOTrainer(vec_env, cfg, run)
    elif args.agent == 'dqn':
        trainer = DQNTrainer(vec_env, cfg, run)
    else:
        trainer = QLearningTrainer(env, CONFIG['q_learning'], run)
    model = trainer.train()
    print('Training finished.')

    if run:
        wandb.finish()


if __name__ == '__main__':
    class Args:
        agent = 'dqn'           # choose 'ppo', 'dqn', or 'q_learning'
        shaping = 'potential'        # choose 'none', 'naive', 'potential'
        timesteps = 200_000
        run_name = None         # e.g. "test_run", default: [model]_[shaping]
        no_wandb = False
    args = Args()
    main(args)
