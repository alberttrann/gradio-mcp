import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create handlers
    # File handler with daily rotation
    log_file = os.path.join(log_dir, f"mcp_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    loggers = {
        'server': logging.getLogger('server'),
        'research': logging.getLogger('research'),
        'synthesis': logging.getLogger('synthesis'),
        'client': logging.getLogger('client')
    }
    
    for logger in loggers.values():
        logger.setLevel(logging.DEBUG)
    
    return loggers
