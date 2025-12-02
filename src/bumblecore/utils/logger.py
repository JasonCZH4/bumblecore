import logging
from pathlib import Path
from typing import Optional

def setup_trainer_logger(
    output_dir: Optional[str] = None,
    logger_name: Optional[str] = None,
    log_file_name: str = "train.log"
) -> logging.Logger:

    if logger_name is None:
        logger_name = Path(__file__).stem

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    handlers = []

    if output_dir is not None:
        log_dir = Path(output_dir).expanduser().resolve() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / log_file_name
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        handlers.append(fh)


    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger