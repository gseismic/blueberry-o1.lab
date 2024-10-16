import sys
from loguru import logger as user_logger

user_logger.remove()
user_logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <cyan>{message}</cyan>", level="DEBUG")
# user_logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
# user_logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
__all__ = ['user_logger']

# user_logger.error('error')
# user_logger.info('info')
# user_logger.warning('warning')
# user_logger.debug('debug')
