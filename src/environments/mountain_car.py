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


def parse_args():
    """
    Parse command-line arguments using argparse (Python standard library).
    
    Usage examples:
        # With defaults (all parameters use defaults)
        python mountain_car.py
        
        # Using --key=value format (recommended)
        python mountain_car.py --agent=ppo --shaping=none --timesteps=2000 --run-name='test-run1' --wandb=false
        
        # Using --key value format (alternative)
        python mountain_car.py --agent ppo --shaping none --timesteps 2000 --run-name 'test-run1' --wandb false
        
        # Mixing formats
        python mountain_car.py --agent=ppo --shaping none --timesteps=2000
        
        # Using GPU
        python mountain_car.py --device=cuda --agent=dqn --timesteps=5000
        
        # Get help
        python mountain_car.py --help
    
    Note: argparse automatically handles:
    - Type conversion (e.g., timesteps is converted to int)
    - Validation (choices are enforced)
    - Help messages (--help flag)
    - Default values (shown in help)
    """
    parser = argparse.ArgumentParser(
        description='Train RL agents on MountainCar environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--agent',
        choices=['ppo', 'dqn', 'q-learning'],
        default='ppo',
        help='RL agent to use for training'
    )
    
    parser.add_argument(
        '--shaping',
        choices=['none', 'naive', 'potential'],
        default='none',
        help='Reward shaping method to apply'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000,
        help='Number of training timesteps (small number for quick testing)'
    )
    
    parser.add_argument(
        '--run-name',
        default=None,
        help='Name for the wandb run (default: auto-generated from parameters)'
    )
    
    def str_to_bool(value):
        """Convert string to boolean, accepting true/false/1/0 (case-insensitive)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        value_lower = str(value).lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, got: {value}')
    
    parser.add_argument(
        '--wandb',
        type=str_to_bool,
        default=True,
        help='Enable (true/1) or disable (false/0) wandb logging'
    )
    
    args = parser.parse_args()
    
    return args


def print_run_config(args, run_name):
    """
    Print formatted run configuration to console.
    
    Args:
        args: Parsed command-line arguments
        run_name: Final run name (may differ from args.run_name if default was used)
    """
    # Format wandb status (convert boolean to lowercase string)
    wandb_display = str(args.wandb).lower()
    
    print('\n' + '=' * 50)
    print('Run Configuration')
    print('=' * 50)
    print(f'run-name: {run_name}')
    print(f'device: {args.device}')
    print(f'environment: mountain_car')
    print(f'agent: {args.agent}')
    print(f'shaping: {args.shaping}')
    print(f'timesteps: {args.timesteps}')
    print(f'wandb: {wandb_display}')
    print('=' * 50 + '\n')


def make_env(shaping='none'):
    """Create and configure the MountainCar environment."""
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


def main():
    """Main training function."""
    args = parse_args()
    
    # Get config for the selected agent
    cfg = CONFIG.get(args.agent, {}).copy()
    if 'total_timesteps' in cfg:
        cfg['total_timesteps'] = args.timesteps
    elif args.agent == 'q-learning':
        # q-learning uses total_episodes instead
        cfg['total_episodes'] = args.timesteps
    
    # Set run name: auto-generate from parameters if not provided
    if args.run_name is None:
        run_name = f"{args.device}_{args.agent}_{args.shaping}_{args.timesteps}"
    else:
        run_name = args.run_name
    
    # Print run configuration to console
    print_run_config(args, run_name)
    
    # Prepare tags for wandb (based on parameters)
    tags = ['mountain_car', args.device, args.agent, args.shaping]
    
    # Initialize Weights & Biases run
    # wandb wird hier aktiviert/deaktiviert basierend auf args.wandb:
    # - Wenn args.wandb == True (--wandb=true/1): wandb.init() wird aufgerufen, run wird gesetzt
    # - Wenn args.wandb == False (--wandb=false/0): wandb.init() wird NICHT aufgerufen, run bleibt None
    # Der run-Objekt wird dann an die Trainer übergeben, die es für Logging verwenden
    run = None
    if args.wandb:  # Nur wenn wandb aktiviert ist (True)
        run = wandb.init(
            project=CONFIG['wandb']['project'],
            name=run_name,
            tags=tags,
            config={
                'agent': args.agent,
                'shaping': args.shaping,
                'device': args.device,
                'timesteps': args.timesteps,
                **cfg
            }
        )
    
    # Environment setup
    env = make_env(args.shaping)
    
    # Wrap in VecEnv for SB3 (not needed for q-learning)
    if args.agent == 'q-learning':
        trainer = QLearningTrainer(env, cfg, run)
    else:
        vec_env = DummyVecEnv([lambda: env])
        if args.agent == 'ppo':
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            trainer = PPOTrainer(vec_env, cfg, run, device=args.device)
        elif args.agent == 'dqn':
            trainer = DQNTrainer(vec_env, cfg, run, device=args.device)
    
    model = trainer.train()
    print('Training finished.')
    
    # Nur wenn wandb aktiviert war, wird wandb.finish() aufgerufen
    # Das schließt den wandb-Run ordnungsgemäß ab
    if run:
        wandb.finish()


if __name__ == '__main__':
    main()
