from loguru import logger as default_logger

user_logger = default_logger

user_logger.add("file.log", rotation="1 MB", retention="10 days", compression="zip")  

__all__ = ['user_logger', 'default_logger']

