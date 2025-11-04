import logging
import os

project_root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')


def configure_logging_for_montezumas(log_name):
    log_root_dir = f'{project_root_path}/log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=f'{log_root_dir}/{log_name}.log',
        filemode='w',
        force=True
    )

    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Set a format that is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger('').addHandler(console)
