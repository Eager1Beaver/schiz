"""App logger."""

import logging

# configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)
# Get the root logger
logger = logging.getLogger()
