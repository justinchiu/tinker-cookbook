"""
Configure tau2's loguru logger BEFORE tau2 is imported.
This module must be imported at the very beginning of train.py.
"""

import os
import sys

# Configure loguru immediately
try:
    from loguru import logger as loguru_logger

    # Remove ALL default handlers
    loguru_logger.remove()

    # Get configuration from environment
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
        print(f"Redirecting tau2 logs to: {tau2_log_file} (level: {tau2_log_level})")
    else:
        # Only show warnings and above to stderr if no file specified
        loguru_logger.add(
            sys.stderr,
            level=tau2_log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        )
        print(f"tau2 logs to stderr (level: {tau2_log_level})")

except ImportError:
    # Loguru not installed
    print("Warning: loguru not installed, cannot configure tau2 logging")