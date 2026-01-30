import logging
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

loge_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(loge_path, LOG_FILE, exist_ok=True)

LOG_FILE_PATH = os.path.join(loge_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)