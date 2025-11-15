import os
from pathlib import Path

PROJECT_ROOT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")


def create_dir_if_not_exists(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
