from sklearn.model_selection import ParameterGrid

from environments.montezumas_revenge import main as montezumas_revenge_main


def main():
    parameter_combinations = ParameterGrid({
        "ppo_learning_rate": [3e-4, 3e-5, 3e-6],
        "ppo_n_steps": [8_000, 16_000, 32_000],
        "truncate_episode_after_n_steps": [8_000, 16_000]
    })

    for parameter_combination in parameter_combinations:
        montezumas_revenge_main(
            timesteps=1_000_000,
            wandb_entity=None,
            wandb_project='ShapingRL-Montezuma',
            agent='ppo',
            debug=False,
            reduced_action_space='only_basic_movements',
            shaping='test',
            add_visited_states_in_observations=True,
            ppo_n_epochs=20,
            **parameter_combination,
            wandb_run_group='montezumas_gridsearch_1'
        )


if __name__ == '__main__':
    main()
