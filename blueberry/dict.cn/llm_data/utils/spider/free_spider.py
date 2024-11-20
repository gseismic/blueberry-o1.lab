
import time
import requests
from llm_data.logger import default_logger

DEFAULT_TIMEOUT     = 10
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_SLEEP = 3

# 个人早期代码，质量不高，带更新，请勿使用
class FreeSpider(object):

    def __init__(self, headers=None, proxies=None,
                 timeout=None, retry_sleep=None, 
                 retry_count=None, logger=None, 
                 verbosity=0, **kwargs):
        self.init(headers, proxies, timeout, retry_sleep, retry_count, logger, verbosity)

    def init(self, headers=None, proxies=None, timeout=None, 
             retry_sleep=None, retry_count=None, logger=None, 
             verbosity=0, **kwargs):
        default_headers = {
            'User-Agent': 
                ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' 
                 '(KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36') }
        self.headers = headers if headers else default_headers
        self.proxies = proxies
        self.timeout = timeout if timeout else DEFAULT_TIMEOUT
        self.retry_sleep = retry_sleep if retry_sleep else DEFAULT_RETRY_SLEEP
        if retry_count is None:
            self.retry_count = DEFAULT_RETRY_COUNT
        else:
            self.retry_count = retry_count
        self.logger = logger or default_logger
        # self.logger.debug('retry_count: %d, timeout: %f' % (self.retry_count, self.timeout))
        self.verbosity = verbosity
        self._num_errors = 0

    @property
    def num_errors(self):
        return self._num_errors

    def normalize_proxies(self, proxies):
        if isinstance(proxies, dict) or proxies is None:
            return proxies
        else:
            return {'http': proxies, 'https': proxies}

    def rawget(self, url, charset=None, params={}, headers=None,
                      proxies=None,timeout=None, **kwargs):
        """
        不保留本次访问记录
        """
        assert(isinstance(proxies, dict) or proxies is None)
        if proxies == None: proxies = self.proxies
        if headers == None: headers = self.headers
        if timeout == None: timeout = self.timeout
        proxies = self.normalize_proxies(proxies)
        response = requests.get(url, params=params, headers=headers, 
                                proxies=proxies, timeout=timeout,
                                **kwargs)
        # 原始正在使用的编码：encoding, 重新encode
        if charset:
            response.encoding = charset
        response.text.encode('utf8')
        return response

    def rawpost(self, url, data, charset=None, headers=None, 
                      proxies=None, timeout=None, **kwargs):
        assert(isinstance(proxies, dict) or proxies is None)
        if proxies == None: proxies = self.proxies
        if headers == None: headers = self.headers
        if timeout == None: timeout = self.timeout
        proxies = self.normalize_proxies(proxies)
        response = requests.post(url, data=data, headers=headers, 
                                proxies=proxies, timeout=timeout,
                                **kwargs)
        if charset:
            response.encoding = charset
        response.text.encode('utf8')
        return response

    def get(self, url, charset=None, params={}, headers=None,
            proxies=None, timeout=None, 
            retry_count=None, retry_sleep=None,
            allowed_code=[200],
            **kwargs):
        if retry_count == None: retry_count = self.retry_count
        if retry_sleep == None: retry_sleep = self.retry_sleep
        if allowed_code == None: allowed_code = [200]
        cnt = 0
        response = None
        while True:
            cnt += 1
            succ = False
            try:
                response = self.rawget(url=url, charset=charset, params=params, 
                                       headers=headers, proxies=proxies, 
                                       timeout=timeout, **kwargs)
                # XXX new
                if response.status_code in allowed_code:
                    succ = True
                else:
                    succ, msg = False, 'Bad code: %d' % response.status_code
            except requests.exceptions.ConnectTimeout as e:
                succ, msg = False, 'ConnectTimeout: %s' % str(e)
            except requests.exceptions.ConnectionError as e:
                succ, msg = False, 'ConnectionError: %s' % str(e)
            except Exception as e:
                succ, msg = False, str(e)
            # except: 2020-03-01 20:01:05
            #    succ, msg = False, 'UnknownException'

            if succ:
                break
            else:
                if self.verbosity >= -1:
                    self.logger.warning(msg)
                if cnt <= retry_count:
                    if self.verbosity >= 0:
                        self.logger.info('Retry [%d]: %s ...' % (cnt, url))
                    time.sleep(retry_sleep)
                else:
                    if self.verbosity >= -2:
                        self.logger.error('Failure after %d retries: %s' % ( \
                                retry_count, url))
                    self._num_errors += 1
                    raise Exception('Connection Timeout/Error or Bad status_code')
                    # break
        return response

    def post(self, url, data, charset=None, headers=None,
                      proxies=None, timeout=None, 
                      retry_count=None, retry_sleep=None,
                      **kwargs):
        if retry_count == None: retry_count = self.retry_count
        if retry_sleep == None: retry_sleep = self.retry_sleep
        cnt = 0
        response = None
        while True:
            cnt += 1
            succ = False
            try:
                response = self.rawpost(url=url, data=data, charset=charset, 
                                       headers=headers, proxies=proxies, 
                                       timeout=timeout, **kwargs)
                succ = True
            except requests.exceptions.ConnectTimeout as e:
                succ, msg = False, 'ConnectTimeout: %s' % str(e)
            except requests.exceptions.ConnectionError as e:
                succ, msg = False, 'ConnectionError: %s' % str(e)
            except Exception as e:
                succ, msg = False, str(e)
            if succ:
                break
            else:
                if self.verbosity >= -1:
                    self.logger.warning(msg)
                if cnt <= retry_count:
                    if self.verbosity >= -2:
                        self.logger.info('Retry [%d]: %s ...' % (cnt, url))
                    time.sleep(retry_sleep)
                else:
                    if self.verbosity >= -2:
                        self.logger.error('Failure after %d retries: %s' % (retry_count, url))
                    self._num_errors += 1
                    raise Exception('Connection Timeout/Error')
        return response

    def get_source(self, url, charset=None, params={}, headers=None,
                         proxies=None, timeout=None, retry_count=None,
                         retry_sleep=None, verbose=1, **kwargs):
        if params == None: params={}
        response = self.get(url, charset=charset, params=params,
                                 headers=headers, 
                                 proxies=proxies, timeout=timeout, 
                                 retry_count=retry_count, 
                                 retry_sleep=retry_sleep, **kwargs)
        return response.text if response!= None else None

    def post_source(self, url, data={}, charset=None, headers=None,
                         proxies=None, timeout=None, retry_count=None,
                         retry_sleep=None, verbose=1, **kwargs):
        if data == None: data ={}
        response = self.post(url, data=data, charset=charset,
                                 headers=headers, 
                                 proxies=proxies, timeout=timeout, 
                                 retry_count=retry_count, 
                                 retry_sleep=retry_sleep, **kwargs)
        return response.text if response!= None else None