import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
import sys
import os
import numpy as np

# Add parent folder to sys.path (so imports work)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CONFIG
from core.utils import record_best_q_learning_run, record_best_run
from shaping.naive_dense import NaiveDenseWrapper
from shaping.potential_based import PotentialBasedWrapper
from agents.ppo import PPOTrainer
from agents.dqn import DQNTrainer
from agents.q_learning import QLearningTrainer

def make_env(shaping='none', render_mode="none"):
    env = gym.make('MountainCar-v0', render_mode=render_mode)
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
    parser.add_argument('--record-best', action='store_true', help="Record the best model after training", default=False)
    args: Args = parser.parse_args() if args is None else args

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
        trainer = PPOTrainer(vec_env, cfg, run, args.shaping)
    elif args.agent == 'dqn':
        trainer = DQNTrainer(vec_env, cfg, run, args.shaping)
    else:
        trainer = QLearningTrainer(env, CONFIG['q_learning'], run, args.shaping)
    model = trainer.train()
    print('Training finished.')

    if run:
        wandb.finish()

    if args.record_best:
        if args.agent in ['ppo', 'dqn']:
            eval_env = make_env(args.shaping, render_mode="rgb_array")
            record_best_run(args.agent, eval_env, best_model_dir=f"./models/{args.shaping}", video_folder=f"./videos/{args.shaping}")
        elif args.agent == 'q_learning':
            # Load best Q-table (assumes you saved it during training as 'best_q_table.npy')
            best_q = np.load(f"./models/{args.shaping}/best_q_table.npy", allow_pickle=True).item()
            eval_env = make_env(args.shaping, render_mode="rgb_array")
            record_best_q_learning_run(eval_env, best_q, discretize_fn=QLearningTrainer(eval_env, CONFIG['q_learning']).discretize, video_folder=f"./videos/{args.shaping}")


if __name__ == '__main__':
    class Args:
        agent = 'ppo'           # choose 'ppo', 'dqn', or 'q_learning'
        shaping = 'none'        # choose 'none', 'naive', 'potential'
        timesteps = 200_000
        run_name = None         # e.g. "test_run", default: [model]_[shaping]
        no_wandb = False
        record_best = True
    args = Args()
    main(args)
