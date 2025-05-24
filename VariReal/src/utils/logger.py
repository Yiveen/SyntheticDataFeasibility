import logging
import os
from datetime import datetime

from utils.utils import get_generated_key

class Logger:
    def __init__(self, exp_name, log_level=logging.INFO):
        self.total_steps = 0
        
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        key = get_generated_key(exp_name)
        log_dir = f"logs/{key}"
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{current_time}.log')
        
        self.logger = logging.getLogger(f"{key}")
        self.logger.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # output to file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.propagate = False

        self.logger.info(f"Logger initialized for experiment: {exp_name.dataset}_{exp_name.gen_class}")

    def log(self, message, level=logging.INFO):
        """log info"""
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        else:
            self.logger.info(message)

