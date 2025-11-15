import logging
from util import create_dir_if_not_exists, PROJECT_ROOT_PATH


def configure_logging_for_montezumas(log_name):
    log_root_dir = f"{PROJECT_ROOT_PATH}/log"

    create_dir_if_not_exists(log_root_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=f"{log_root_dir}/{log_name}.log",
        filemode="w",
        force=True,
    )

    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Set a format that is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger("").addHandler(console)
