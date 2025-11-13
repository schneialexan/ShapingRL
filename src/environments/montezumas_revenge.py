import argparse
import glob
import logging
import re
from contextlib import contextmanager

import gymnasium as gym
import ale_py  # must be there, do not remove !!!!!!!!!!!!!!!!!!!!!!!
from datetime import datetime

import numpy as np
import wandb
import os
import sys
from pathlib import Path

from gymnasium import ObservationWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder

from shaping.montezuma.naive_dense import NaiveDenseWrapper
from shaping.montezuma.montezuma_potential_based import MontezumaPotentialBasedWrapper
from shaping.montezuma.test import TestDenseWrapper
from training.coach import MontezumasKeyGuidelineCoach, MontezumasKeyGuidelineCoachForcingOrder
from util import PROJECT_ROOT_PATH

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo import PPOTrainer
from agents.dqn import DQNTrainer
from agents.q_learning import QLearningTrainer


class ReducedActionSpaceWrapper(gym.ActionWrapper):
    """Wrapper to reduce action space to a meaningful subset.


    Note: Montezuma Revenge has 18 actions in total. The 'fire' action is a jump.
    """

    action_sets = {

        'only_basic_movements': list(range(0, 6)),  # NOOP, FIRE, UP, RIGHT, LEFT, DOWN
        'extended_movements': list(range(0, 10)),  # + [UP/DOWN][LEFT/RIGHT]
        'extended_movements_fire': list(range(0, 14)),  # + [UP/DOWN/LEFT/RIGHT]FIRE
        # the 4 remaining actions are diagonal jumps
    }

    def __init__(self, env, action_subset=None):
        super().__init__(env)
        if action_subset is None:
            action_subset = self.action_sets['extended_movements_fire']
        self.action_subset = action_subset
        self.action_space = gym.spaces.Discrete(len(action_subset))
        self._action_mapping = {i: action_subset[i] for i in range(len(action_subset))}

    def action(self, act):
        return self._action_mapping[act]





def make_env(
    env_name='ALE/MontezumaRevenge-v5',
    render_mode=None,
    reduce_action_space=False,
    shaping='test',
    debugging=False
):
    """Create Montezuma's Revenge environment with optional reduced action space."""
    try:
        env = gym.make(
            env_name,
            render_mode=render_mode,
            #obs_type='ram'
        )
    except gym.error.Error:
        from gymnasium.envs.registration import register
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=4_000,
            reward_threshold=3000.0,
        )
        env = gym.make(
            env_name,
            render_mode=render_mode,
            #obs_type='ram'
        )
    if reduce_action_space:
        env = ReducedActionSpaceWrapper(env)

    if shaping == 'naive':
        env = NaiveDenseWrapper(
            env,
            # guideline_coach=MontezumasKeyGuidelineCoach
            guideline_coach=MontezumasKeyGuidelineCoachForcingOrder
        )
    elif shaping == 'potential':
        env = MontezumaPotentialBasedWrapper(
            env,
            gamma=0.99,
            debugging=debugging
        )
    elif shaping == 'test':
        env = TestDenseWrapper(env)

    return env


def extract_steps(filename):
    """
    Extracts the two step numbers from a filename like:
    'rl-video-step-1600-to-step-1800.mp4'

    Returns:
        tuple of two integers (start_step, end_step)
        or None if no match is found.
    """
    match = re.search(r'step-(\d+)-to-step-(\d+)', filename)
    if match:
        start_step, end_step = map(int, match.groups())
        return start_step, end_step
    return None


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['ppo', 'dqn', 'q_learning'], default='ppo')
    parser.add_argument('--timesteps', type=int, default=30_000)
    parser.add_argument('--project', type=str, default='ShapingRL-Montezuma')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reduce_action_space', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--shaping', type=str, default='test')
    args = parser.parse_args() if args is None else args

    # Automatically set run name
    run_name = args.run_name if args.run_name else f"{args.agent}_montezuma"

    if args.debug:
        os.environ["WANDB_MODE"] = "offline"

    # Initialize Weights & Biases
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        config=vars(args),
        monitor_gym=True
    )

    # Create environment
    render_mode = "human" if args.render else 'rgb_array'
    env = make_env(
        'ALE/MontezumaRevenge-v5',
        render_mode=render_mode,
        reduce_action_space=args.reduce_action_space,
        shaping=wandb.config.shaping,
        debugging=args.debug
    )
    env.reset(seed=42)

    if args.reduce_action_space:
        print(f"Action space reduced to {env.action_space.n} actions")
    else:
        print(f"Full action space: {env.unwrapped.get_action_meanings()}")

    # Wrap for SB3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    if not args.render:
        print('Saving videos!!!!!!!!!!')

        video_recording_length = 1000
        vec_env = VecVideoRecorder(
            vec_env, f"videos/{run.id}",
            record_video_trigger=lambda x: x % video_recording_length == 0,
            video_length=(video_recording_length - 1),
        )

    if args.agent == 'ppo':
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=255,
            clip_reward=500
        )
        trainer = PPOTrainer(
            vec_env,
            config={'total_timesteps': args.timesteps},
            run=run
        )
    elif args.agent == 'dqn':
        trainer = DQNTrainer(
            vec_env,
            config={'total_timesteps': args.timesteps},
            run=run
        )
    else:
        # Approximate number of episodes for Q-learning
        trainer = QLearningTrainer(env, config={'total_episodes': args.timesteps // 200}, run=run)

    # Train agent
    model_or_q = trainer.train()
    print("Training finished.")

    # Save model
    save_dir = os.environ.get('SHAPINGRL_SAVE_DIR', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'{args.agent}_montezuma_final')
    try:
        if args.agent in ['ppo', 'dqn']:
            model_or_q.save(fname)
            wandb.save(fname)
        else:
            # Save Q-table
            import pickle
            with open(fname + '.pkl', 'wb') as f:
                pickle.dump(model_or_q, f)
            wandb.save(fname + '.pkl')
        print(f"Model saved to {fname}")
    except Exception as e:
        print(f"Failed to save model: {e}")

    video_files = sorted(
        glob.glob(os.path.join(PROJECT_ROOT_PATH, 'videos', wandb.run.id, '*.mp4'))
    )

    for path in video_files:
        start_step, end_step = extract_steps(path)
        wandb.log(
            {
                "gym_video": wandb.Video(path, caption='Video', fps=30, format="mp4"),
                "gym_video_start_step": start_step,
                "gym_video_end_step": end_step,
                "env_step": start_step
            }
        )

    # Close environment
    env.close()
    run.finish()


if __name__ == '__main__':
    main()
