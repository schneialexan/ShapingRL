import argparse
import gymnasium as gym
import ale_py
import numpy as np
from datetime import datetime
import wandb
import os
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now the imports will work
from src.agents.ppo import PPOAgent

def make_env(env_name='ALE/MontezumaRevenge-v5', render_mode=None):
    """Create the Montezuma's Revenge environment."""
    try:
        env = gym.make(env_name, render_mode=render_mode)
    except gym.error.Error:
        from gymnasium.envs.registration import register
        
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=20000,
            reward_threshold=3000.0,
        )
        env = gym.make(env_name, render_mode=render_mode)
    return env

def main(cli_args=None, device=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['ppo'], default='ppo')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--project', type=str, default='TestRun-v4')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluate every N timesteps')
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    
    if cli_args is None:
        args = parser.parse_args()
    else:
        # Allow passing a dict or Namespace
        if isinstance(cli_args, dict):
            args = parser.parse_args([])  # create a Namespace
            for k, v in cli_args.items():
                setattr(args, k, v)
        else:
            args = cli_args

    # wandb init
    today = datetime.now().strftime("%Y%m%d")
    run = wandb.init(
        project=args.project, 
        entity=args.entity, 
        config=vars(args), 
        name=f"montezuma_revenge_{today}_{args.agent}"
    )

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env('ALE/MontezumaRevenge-v5', render_mode=render_mode)
    env.reset(seed=args.seed)
    
    print(f"Environment action space: {env.action_space}")
    print(f"Environment actions: {env.unwrapped.get_action_meanings()}")

    # Initialize PPO Agent
    obs_dim = env.observation_space.shape
    n_actions = env.action_space.n
    agent = PPOAgent(env, obs_dim=obs_dim, n_actions=n_actions, device=device, seed_n=args.seed)

    # Train the agent
    print(f"Training PPO agent for {args.timesteps} timesteps...")
    agent.learn(total_timesteps=args.timesteps)
    
    # Final evaluation
    print(f"Final evaluation with {args.n_eval_episodes} episodes...")
    mean_reward = agent.evaluate_policy(n_eval_episodes=args.n_eval_episodes)
    print(f"PPO agent finished with mean reward over {args.n_eval_episodes} episodes: {mean_reward:.2f}")
    
    # Log final evaluation
    wandb.log({
        "final_eval_mean_reward": mean_reward,
    })
    
    # Save the final model
    save_dir = os.environ.get('SHAPINGRL_SAVE_DIR', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'{args.agent}_montezuma_final.pt')
    try:
        agent.save(fname)
        wandb.save(fname)
        print(f"Model saved to {fname}")
    except Exception as e:
        print(f"Failed to save model: {e}")
    
    env.close()
    run.finish()

if __name__ == '__main__':
    # check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main(device=device)
