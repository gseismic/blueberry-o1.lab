from llm_data.logger import default_logger
from llm_data.utils.spider import FreeSpider

class BaseAPI:

    def __init__(self, spider=None, logger=None):
        self.spider = spider or FreeSpider()
        self.logger = logger or default_logger
