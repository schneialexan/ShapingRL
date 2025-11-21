from wandb_settings import MONTEZUMAS_WANDB_PROJECT_CONFIG
from wandb_tables import get_latex_table, WandbDataProvider


def main():
    print(get_latex_table(
        WandbDataProvider(
            wandb_project_config=MONTEZUMAS_WANDB_PROJECT_CONFIG,
            configuration_keys=[
                'ppo_config.learning_rate',
                'shaping'
            ],
            metrics=[
                'episode_length',
                'episode_reward'
            ],
            only_finished_runs=False
        ),
        metric_definitions=[
            ('episode_length', 'high'),
            ('episode_reward', 'high')
        ],
        columns_to_hide=['name']
    ))


if __name__ == '__main__':
    main()
