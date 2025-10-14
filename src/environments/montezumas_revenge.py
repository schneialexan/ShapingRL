# montezumas_revenge.py (updated)
import gymnasium as gym
import ale_py

# Attempt to manually register the environment if it's missing
try:
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")
except gym.error.Error:
    from gymnasium.envs.registration import register

    register(
        id="ALE/MontezumaRevenge-v5",
        entry_point="ale_py.roms:MontezumaRevenge",
        max_episode_steps=18000,
        reward_threshold=3000.0,
    )
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print("Random agent finished with reward:", total_reward)
