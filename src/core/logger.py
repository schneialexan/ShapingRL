import wandb

class AgentLogger:
    """Wrapper to log metrics to WandB cleanly."""
    def __init__(self, run=None):
        self.run = run

    def log_episode(self, episode, **kwargs):
        if self.run is not None:
            data = {'episode': episode}
            data.update(kwargs)
            wandb.log(data)

    def log_metrics(self, metrics):
        if self.run is not None:
            wandb.log(metrics)
