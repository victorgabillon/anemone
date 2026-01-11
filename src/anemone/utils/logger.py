# logger_module.py

from contextlib import contextmanager
from logging import DEBUG, ERROR, WARNING, Logger, getLogger
from typing import Generator

from rich.logging import RichHandler

anemone_logger = getLogger("anemone_app")
anemone_logger.setLevel(DEBUG)

if not anemone_logger.handlers:
    console_handler = RichHandler()
    console_handler.setLevel(DEBUG)
    anemone_logger.addHandler(console_handler)
    anemone_logger.propagate = False


def set_chipiron_logger_level(level: int) -> None:
    """
    Set the logging level for the chipiron logger and all its handlers.

    This ensures that both the logger and its handlers are set to the same level,
    so log messages at the specified level will actually be displayed.

    Args:
        level (int): The logging level to set (e.g., logging.DEBUG, logging.INFO,
                    logging.WARNING, logging.ERROR, logging.CRITICAL)
    """
    anemone_logger.setLevel(level)
    for handler in anemone_logger.handlers:
        handler.setLevel(level)


@contextmanager
def suppress_logging(
    logger: Logger, level: int = WARNING
) -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress logging for a specific logger to a given level.

    Sets the logger's level to the specified value for the duration of the context, then restores
    its original level afterwards. Useful for silencing output from a particular logger during
    benchmarking or other operations.

    Args:
        logger (logging.Logger): The logger to suppress.
        level (int): The logging level to set (e.g., logging.ERROR, logging.WARNING).

    Yields:
        None
    """
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Suppress all logging from all loggers (global)
@contextmanager
def suppress_all_logging(level: int = ERROR) -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress logging from all loggers to a specified level.

    This sets the level of all loggers (including the root logger) to the given level for the duration
    of the context, then restores their original levels afterwards. Useful for benchmarking or
    situations where you want to silence all logging output temporarily.

    Args:
        level (int): The logging level to set (e.g., logging.ERROR, logging.WARNING).

    Yields:
        None
    """
    logger_dict = getLogger().manager.loggerDict
    original_levels: dict[str, int] = {}

    for name in logger_dict:
        logger = getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(level)

    root_logger = getLogger()
    original_root_level = root_logger.level
    root_logger.setLevel(level)

    try:
        yield
    finally:
        for name, original_level in original_levels.items():
            getLogger(name).setLevel(original_level)
        root_logger.setLevel(original_root_level)
