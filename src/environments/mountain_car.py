import argparse
import gymnasium as gym
import numpy as np
from datetime import datetime
import wandb
import os
import sys
# local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent

def make_env(env_name='MountainCar-v0'):
    # Gym classic control environment
    return gym.make(env_name)

def run_episode(env, agent, agent_name, evaluate=False, max_steps=200):
    obs, _ = env.reset()
    total_reward = 0.0
    losses = []
    done = False
    steps = 0

    # For PPO API: collect a batch list
    ppo_batch = []

    while not done and steps < max_steps:
        if agent_name == 'ppo':
            action = agent.select_action(obs, evaluate)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            log_prob = None
            ppo_batch.append((obs, action, reward, next_obs, done, log_prob))
            obs = next_obs
            total_reward += reward
            steps += 1
        else:
            action = agent.select_action(obs, evaluate)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            loss = agent.observe(obs, action, reward, next_obs, done)
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total_reward += reward
            steps += 1

    if agent_name == 'ppo' and len(ppo_batch) > 0 and not evaluate:
        loss = agent.observe(ppo_batch)
        return total_reward, loss
    
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else None
    return total_reward, avg_loss

def main(cli_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['q_learning', 'dqn', 'ppo'], default='q_learning')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--project', type=str, default='ShapingRL')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=200)
    
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
    run = wandb.init(project=args.project, entity=args.entity, config=vars(args), name=f"mountain_car_{today}_{args.agent}")

    env = make_env('MountainCar-v0')
    env.reset(seed=args.seed)

    agent = None
    agent_name = args.agent
    if args.agent == 'q_learning':
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        agent = QLearningAgent(obs_low, obs_high, n_bins=(40, 40), n_actions=env.action_space.n)
    elif args.agent == 'dqn':
        obs_dim = env.observation_space.shape[0]
        agent = DQNAgent(obs_dim, env.action_space.n, device='cpu')
    elif args.agent == 'ppo':
        obs_dim = env.observation_space.shape[0]
        agent = PPOAgent(obs_dim, env.action_space.n, device='cpu')

    best_return = -float('inf')
    for ep in range(1, args.episodes + 1):
        ret, loss = run_episode(env, agent, agent_name, evaluate=False, max_steps=args.max_steps)
        wandb.log({'episode': ep, 'return': ret, 'loss': loss})

        if ep % 25 == 0:
            eval_ret, _ = run_episode(env, agent, agent_name, evaluate=True, max_steps=args.max_steps)
            wandb.log({'eval_return': eval_ret, 'episode': ep})
            if eval_ret > best_return:
                best_return = eval_ret
                save_dir = os.environ.get('SHAPINGRL_SAVE_DIR', './checkpoints')
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f'{agent_name}_best.pt')
                try:
                    agent.save(fname)
                    wandb.save(fname)
                except Exception:
                    pass

        if ep % 50 == 0:
            print(f"Episode {ep} return={ret:.2f} eval_best={best_return:.2f}")
    run.finish()

if __name__ == '__main__':
    main({
        'agent': 'dqn',
        'episodes': 100,
        'project': 'TestRun',
        'seed': 42
    })
