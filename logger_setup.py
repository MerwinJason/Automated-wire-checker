"""
Initializes and configures the global logger for the application.
"""
import logging
import sys

def setup_logger():
    """
    Sets up and returns a configured logger instance.
    """
    logger_instance = logging.getLogger("BreadboardChecker")
    logger_instance.setLevel(logging.DEBUG)  # Set to INFO for less verbosity in production
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger_instance.addHandler(handler)
    return logger_instance

logger = setup_logger()