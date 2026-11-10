CONFIG = {
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "total_timesteps": 100_000,
        "verbose": 1,
    },
    "dqn": {
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.02,
        "target_update_interval": 500,
        "total_timesteps": 200_000,
        "verbose": 1,
    },
    "q_learning": {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.995,
        "n_bins": [18,14],
        "total_episodes": 2000,
    },
    "wandb": {
        "project": "mountaincar-reward-shaping-vfinal",
    }
}
