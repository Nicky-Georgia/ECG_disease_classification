import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger("MLServiceLogger")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        "logs/server.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()
