import argparse
import glob
import logging
import re
from contextlib import contextmanager
from typing import Literal

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

from shaping.montezuma.grid_based import TestDenseWrapper
from shaping.montezuma.naive import NaiveDenseWrapper
from shaping.montezuma.potential import MontezumaPotentialBasedWrapper
from shaping.montezuma.reward_definition_factory import MontezumasTestShaperRewardDefinition, \
    RewardDefinition
from training.coach import MontezumasKeyGuidelineCoach, MontezumasKeyGuidelineCoachForcingOrder
from util import PROJECT_ROOT_PATH

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo import PPOTrainer
from agents.dqn import DQNTrainer
from agents.q_learning import QLearningTrainer

ReducedActionSpaceConfiguration = Literal['only_basic_movements',
'extended_movements',
'extended_movements_fire'
]

ShapingConfigurations = Literal['naive', 'potential', 'test']


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

    def __init__(self, env, action_subset: ReducedActionSpaceConfiguration | None = None):
        super().__init__(env)
        if action_subset is None:
            action_subset = self.action_sets['extended_movements_fire']
        self.action_subset = action_subset
        self.action_space = gym.spaces.Discrete(len(action_subset))
        self._action_mapping = {i: action_subset[i] for i in range(len(action_subset))}

    def action(self, act):
        return self._action_mapping[act]


def make_env(
        truncate_episode_after_n_steps: int,
        env_name='ALE/MontezumaRevenge-v5',
        render_mode=None,
        reduce_action_space: ReducedActionSpaceConfiguration = None,
        shaping: ShapingConfigurations = 'test',
        debugging: bool = False,
        add_visited_states_in_observations=False
) -> tuple[gym.Env, RewardDefinition | None]:
    reward_definition = None

    """Create Montezuma's Revenge environment with optional reduced action space."""
    try:
        env = gym.make(
            env_name,
            render_mode=render_mode,
            max_episode_steps=truncate_episode_after_n_steps
            # obs_type='ram'
        )
    except gym.error.Error:
        from gymnasium.envs.registration import register
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=truncate_episode_after_n_steps,
            reward_threshold=3000.0,
        )
        env = gym.make(
            env_name,
            render_mode=render_mode,
            max_episode_steps=truncate_episode_after_n_steps
            # obs_type='ram'
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
        reward_definition = MontezumasTestShaperRewardDefinition(
            truncate_episode_after_n_steps=truncate_episode_after_n_steps
        )
        env = TestDenseWrapper(
            env,
            reward_definition=reward_definition,
            add_visited_states_in_observations=add_visited_states_in_observations
        )

    return env, reward_definition


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


def main(
        timesteps: int,
        truncate_episode_after_n_steps: int,
        wandb_project: str,
        wandb_entity: str,
        agent: Literal['ppo', 'dqn', 'q_learning'],
        debug: bool,
        reduced_action_space: ReducedActionSpaceConfiguration,
        shaping: ShapingConfigurations,
        add_visited_states_in_observations: bool,
        ppo_n_steps: int = 2048,
        ppo_learning_rate: float = 3e-4,
        ppo_n_epochs: int = 10
):
    if debug:
        os.environ["WANDB_MODE"] = "offline"

    print(f'{add_visited_states_in_observations = }')

    run_name_parts = []

    if agent == 'ppo':
        run_name_parts.append(f'ppo_montezumas')

        if add_visited_states_in_observations:
            run_name_parts.append(f'add_visited_states_in_observations')

        run_name_parts.append(f'n-steps_{ppo_n_steps}-n-epochs_{ppo_n_epochs}-learning-rate_{ppo_learning_rate:.2e}')

    run_name = '/'.join(run_name_parts)

    # Initialize Weights & Biases
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={
            'timesteps': timesteps,
            'truncate_episode_after_n_steps': truncate_episode_after_n_steps,
            'debug': debug,
            'shaping': shaping,
            'reduced_action_space': reduced_action_space,
            'add_visited_states_in_observations': add_visited_states_in_observations
        },
        monitor_gym=True
    )

    # Create environment
    render_mode = "human" if args.render else 'rgb_array'
    env, reward_definition = make_env(
        truncate_episode_after_n_steps=truncate_episode_after_n_steps,
        env_name='ALE/MontezumaRevenge-v5',
        render_mode=render_mode,
        reduce_action_space=wandb.config['reduced_action_space'],
        shaping=wandb.config.shaping,
        debugging=wandb.config['debug'],
        add_visited_states_in_observations=wandb.config['add_visited_states_in_observations']
    )
    env.reset(seed=42)

    wandb.config.update({'reward_definition': reward_definition.get_reward_definition_log_dict()})

    if reduced_action_space:
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

    if agent == 'ppo':
        wandb.config.update({
            'ppo_config': {
                'n_epochs': ppo_n_epochs,
                'n_steps': ppo_n_steps,
                'learning_rate': ppo_learning_rate,
                'total_timesteps': timesteps
            }})
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=255,
            clip_reward=500,
        )
        trainer = PPOTrainer(
            vec_env,
            config=wandb.config['ppo_config'],
            run=run
        )
    elif agent == 'dqn':
        trainer = DQNTrainer(
            vec_env,
            config={'total_timesteps': timesteps},
            run=run
        )
    elif agent == 'q_learning':
        # Approximate number of episodes for Q-learning
        trainer = QLearningTrainer(env, config={'total_episodes': timesteps // 200}, run=run)
    else:
        raise Exception(f'Invalid parameter value {agent = }')

    # Train agent
    model_or_q = trainer.train()
    print("Training finished.")

    # Save model
    save_dir = os.environ.get('SHAPINGRL_SAVE_DIR', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'{agent}_montezuma_final')
    try:
        if agent in ['ppo', 'dqn']:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['ppo', 'dqn', 'q_learning'], default='ppo')
    parser.add_argument('--timesteps', type=int, default=30_000)
    parser.add_argument('--project', type=str, default='ShapingRL-Montezuma')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--add-visited-states-in-observations', action='store_true')
    parser.add_argument(
        '--reduced-action-space',
        choices=[
            'only_basic_movements',
            'extended_movements',
            'extended_movements_fire'
        ],
        default=None
    )
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--shaping', type=str, default='test')
    args = parser.parse_args()

    main(
        timesteps=args.timesteps,
        truncate_episode_after_n_steps=8_000,
        wandb_entity=args.entity,
        wandb_project=args.project,
        agent=args.agent,
        debug=args.debug,
        reduced_action_space=args.reduced_action_space,
        shaping=args.shaping,
        add_visited_states_in_observations=args.add_visited_states_in_observations,
        ppo_n_steps=4096,
        ppo_n_epochs=20
    )
