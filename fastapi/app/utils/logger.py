import logging
from logging.handlers import RotatingFileHandler
import os

log_dir = "./logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(f"{log_dir}/app.log", maxBytes=5_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)