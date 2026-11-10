import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
import sys
import os
import numpy as np

# Add parent folder to sys.path (so imports work)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CONFIG
from core.utils import record_best_q_learning_run, record_best_run, make_env
from agents.ppo import PPOTrainer
from agents.dqn import DQNTrainer
from agents.q_learning import QLearningTrainer

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['ppo', 'dqn', 'q_learning'], default='ppo')
    parser.add_argument('--shaping', choices=['none', 'naive', 'potential'], default='potential')
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--run-name', type=str, default='default_run')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--record-best', action='store_true', help="Record the best model after training", default=False)
    parser.add_argument('--record-video', action='store_true', help="Record the model during training", default=True)
    parser.add_argument('--episodes-per-video', action='store_true', help="How many episodes in between each video gets recorded", default=25)
    parser.add_argument('--potential-shaping-method', action='store_true', help=";Method that is used for the potential shaping ([1]x-pos, [2]height, [3]kinetic function)", default=2)
    args: Args = parser.parse_args() if args is None else args

    # Get config for the selected agent
    cfg = CONFIG.get(args.agent, {})
    if args.timesteps and 'total_timesteps' in cfg:
        cfg['total_timesteps'] = args.timesteps
        
    # Automatically set run name
    if args.run_name is None or args.run_name.lower() == 'default_run':
        run_name = f"{args.agent}_{args.shaping}_{args.shaping_method}"
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
    env = make_env(args.agent, 
                   shaping=args.shaping, 
                   record_video=args.record_video, 
                   episodes_per_video=args.episodes_per_video, 
                   shaping_method=args.shaping_method)

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
        videos = []
        for file in os.listdir(f"./videos/{args.shaping}/{args.agent}"):
            if file.endswith(".mp4"):
                videos.append(wandb.Video(os.path.join(f"./videos/{args.shaping}/{args.agent}", file), format="mp4", caption=f"Episode: {file.split('-')[-1].split('.')[0]}"))
        wandb.log({"training_videos": videos})
        wandb.finish()

    if args.record_best:
        if args.agent in ['ppo', 'dqn']:
            eval_env = make_env(args.agent, 
                                shaping=args.shaping, 
                                record_video=args.record_video, 
                                episodes_per_video=args.episodes_per_video, 
                                shaping_method=args.shaping_method)
            record_best_run(args.agent, eval_env, best_model_dir=f"./models/{args.shaping}", video_folder=f"./videos/{args.shaping}")
        elif args.agent == 'q_learning':
            # Load best Q-table (assumes you saved it during training as 'best_q_table.npy')
            best_q = np.load(f"./models/{args.shaping}/best_q_table.npy", allow_pickle=True).item()
            eval_env = make_env(args.agent, 
                                shaping=args.shaping, 
                                record_video=args.record_video, 
                                episodes_per_video=args.episodes_per_video, 
                                shaping_method=args.shaping_method)
            record_best_q_learning_run(eval_env, best_q, discretize_fn=QLearningTrainer(eval_env, CONFIG['q_learning']).discretize, video_folder=f"./videos/{args.shaping}")


if __name__ == '__main__':
    class Args:
        agent = 'q_learning'           # choose 'ppo', 'dqn', or 'q_learning'
        shaping = 'potential'   # choose 'none', 'naive', 'potential'
        timesteps = 200_000   # for q_learning adjust config.py
        run_name = None         # e.g. "test_run", default: [model]_[shaping]_[method]
        no_wandb = False
        record_best = True
        record_video = True
        episodes_per_video = 100
        shaping_method = 3      # 1, 2, 3
    args = Args()
    main(args)
