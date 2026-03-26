import os
import logging
from datetime import datetime


def setup_logger(log_dir='logs', log_filename_prefix='training_log_results'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{log_filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger