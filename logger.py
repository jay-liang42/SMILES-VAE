import logging
import sys

def get_logger():
    # Retrieve logger by name.
    # Using a named logger allows shared configuration across files.
    logger = logging.getLogger("smiles_vae")

    # If handlers already exist, the logger has been configured before.
    # Avoid adding duplicate handlers (which causes repeated log lines).
    if logger.handlers:
        return logger

    # Set minimum severity level to display.
    # INFO will show training progress, warnings, etc.
    logger.setLevel(logging.INFO)

    # Define log message structure.
    # Includes timestamp, severity level, and message.
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # StreamHandler sends logs to terminal/stdout.
    # stdout is preferred for containers and logging systems.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Attach handler to logger
    logger.addHandler(handler)

    return logger
