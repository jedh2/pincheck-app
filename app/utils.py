import os
import logging
from datetime import datetime

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_timestamp():
    """Return current timestamp for filenames/logging."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logging(log_file='app.log'):
    """Configure basic logging."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )