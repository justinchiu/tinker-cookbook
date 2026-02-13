"""
Configure tau2's loguru logger BEFORE tau2 is imported.
This module must be imported at the very beginning of train.py.
"""

import logging
import os
import sys

from loguru import logger as loguru_logger

# Remove ALL default handlers immediately
loguru_logger.remove()


def setup_tau2_logging():
    """Setup tau2 logging after log directory has been validated."""
    tau2_log_level = os.environ.get("TAU2_LOG_LEVEL", "WARNING").upper()
    tau2_log_file = os.environ.get("TAU2_LOG_FILE")

    if tau2_log_file:
        # Add file handler for ALL loguru logs (including tau2)
        loguru_logger.add(
            tau2_log_file,
            level=tau2_log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )
        print(
            f"Redirecting tau2 logs to: {tau2_log_file} (level: {tau2_log_level})"
        )

        # Also setup LiteLLM logging to the same file
        litellm_logger = logging.getLogger("LiteLLM")
        handler = logging.FileHandler(tau2_log_file)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        litellm_logger.addHandler(handler)
        litellm_logger.propagate = False
    else:
        # Only show warnings and above to stderr if no file specified
        loguru_logger.add(
            sys.stderr,
            level=tau2_log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        )
        print(f"tau2 logs to stderr (level: {tau2_log_level})")
