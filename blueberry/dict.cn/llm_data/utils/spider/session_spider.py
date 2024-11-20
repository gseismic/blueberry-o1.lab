import os
import time
import requests
import traceback
from llm_data.logger import default_logger

DEFAULT_TIMEOUT = 10
DEFAULT_RETRY_COUNT = 5
DEFAULT_RETRY_SLEEP = 3

# 个人早期代码，质量不高，带更新，请勿使用
class SessionSpider(object):

    cookies = None # response cookies
    def __init__(self, 
                 session=None,
                 headers=None,
                 logger=None, 
                 timeout=DEFAULT_TIMEOUT, 
                 retry_sleep=DEFAULT_RETRY_SLEEP, 
                 retry_count=DEFAULT_RETRY_COUNT, **kwargs):
        """
        init:
        """
        self.init(session, headers, logger, timeout, retry_sleep, retry_count, **kwargs)
        self._num_errors = 0

    @property
    def num_errors(self):
        return self._num_errors

    def init(self, session, headers, logger, timeout, retry_sleep, retry_count, **kwargs):
        self.logger = logger or default_logger
        if session != None: 
            self.s = session
            if headers != None:
                self.logger.info("header ignored, for current session")
        else:
            if headers is None:
                headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' '(KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36'}
            self.s = requests.Session()
            self.s.headers.update(headers)
        self.timeout        = timeout if timeout != None else DEFAULT_TIMEOUT
        self.retry_count    = retry_count if retry_count != None else DEFAULT_RETRY_COUNT
        self.retry_sleep    = retry_sleep if retry_sleep != None else DEFAULT_RETRY_SLEEP

    def before_request(self, url, type):
        pass

    def after_request(self, url, type, response):
        pass

    def _get(self, url, charset=None, timeout=None, params={}, verbose=1, **kwargs):
        self.before_request(url, type="get")
        if timeout == None: timeout = self.timeout
        try:
            response = self.s.get(url, params=params, timeout=timeout, **kwargs)
            if charset:
                response.encoding = charset
            response.text.encode('utf8')
            # added
            self.status_code = response.status_code
            self.cookies = response.cookies
        except Exception as e:
            if verbose >=2:
                traceback.print_exc()
            if verbose >=1:
                self.logger.info(str(e))
            response = None
        self.after_request(url, type, response)
        return response

    def _post(self, url, data, charset=None, timeout=None, verbose=1, **kwargs):
        # bug?
        self.before_request(url, type="post")
        if timeout == None: 
            timeout = self.timeout
        try:
            response = self.s.post(url, data=data, timeout=timeout, **kwargs)
            if charset:
                response.encoding = charset
            response.text.encode('utf8')
            self.status_code = response.status_code
            self.cookies = response.cookies
        except Exception as e:
            if verbose >=2:
                traceback.print_exc()
            if verbose >= 1:
                self.logger.warn("{}".format(e))
            response = None
        self.after_request(url, type, response)
        return response

    def retry_loop(self, type, url, cnt, timeout, retry_count, retry_sleep, verbose, **kwargs):
        self.logger.info("failure with itry=%d, sleep %f sec, then retry" % (cnt, retry_sleep) )
        pass

    def retry_failure(self, type, url, cnt, timeout, retry_count, 
                           retry_sleep, verbose, **kwargs):
        self.logger.error('Error: %s: failure after %d retries' % (url, retry_count) )
        self._num_errors += 1

    # fixed retry_sucess
    def retry_success(self, type, url, cnt, timeout, retry_count, 
                           retry_sleep, verbose, **kwargs):
        if verbose>=1:
            self.logger.info("retry successful!")

    def __loop_get_post(self, func, type, url, charset=None, timeout=None, 
                              retry_count=None, retry_sleep=None, 
                              verbose=1, **kwargs):
        """
        if success: 
                return data
        else: 
                return None
        retry_count: = call retry_count + 1 get/post
        """
        if timeout == None: timeout = self.timeout
        if retry_count == None: retry_count = self.retry_count
        if retry_sleep == None: retry_sleep = self.retry_sleep
        cnt = 0
        while True:
            cnt += 1
            # call self._get / self._post
            # kwargs include params / data
            response = func(url=url, charset=charset, timeout=timeout, verbose=verbose, **kwargs)
            # 2017-12-09 00:38:40 self.status_code not in [500, 501, 502]:
            if response != None and self.status_code == 200:
                if cnt != 1:
                    self.retry_success(type=type, url=url, cnt=cnt, timeout=timeout, 
                                       retry_count=retry_count, retry_sleep=retry_sleep,
                                       verbose=verbose, **kwargs)
                break;
            if cnt <= retry_count: # as its name
                self.retry_loop(type=type, url=url, cnt=cnt, timeout=timeout, 
                                retry_count=retry_count, retry_sleep=retry_sleep,
                                verbose=verbose, **kwargs)
                time.sleep(retry_sleep)
            else: 
                self.retry_failure(type=type, url=url, cnt=cnt, timeout=timeout, 
                                retry_count=retry_count, retry_sleep=retry_sleep,
                                verbose=verbose, **kwargs)
                break
        return response

    def get(self, url, charset=None, timeout=None, retry_count=None,
                  retry_sleep=None, params={}, verbose=1, **kwargs):
        return self.__loop_get_post(self._get, type="get", url=url, charset=charset, 
                # 2018-02-24 14:30:51 BUG FIXED
                                   retry_count=retry_count, retry_sleep=retry_sleep,
                                   timeout=timeout, params=params, 
                                   verbose=verbose, **kwargs)

    def post(self, url, data, charset=None, timeout=None, retry_count=None,
                  retry_sleep=None, verbose=1, **kwargs):
        return self.__loop_get_post(self._post, type="post", url=url, charset=charset, 
                                   retry_count=retry_count, retry_sleep=retry_sleep,
                                   timeout=timeout, data=data, 
                                   verbose=verbose, **kwargs)

    def get_source(self, url, charset=None, timeout=None, retry_count=None,
                         retry_sleep=None, params={}, verbose=1, **kwargs):
        if params == None: params={}
        response = self.get(url, charset=charset, timeout=timeout, 
                                 retry_count=retry_count, retry_sleep=retry_sleep, 
                                 params=params, **kwargs)
        return response.text if response!= None else None

    def post_source(self, url, data, charset=None, timeout=None, retry_count=None,
                          retry_sleep=None, verbose=1, **kwargs):
        response = self.post(url, data=data, charset=charset, timeout=timeout, 
                                 retry_count=retry_count, retry_sleep=retry_sleep, 
                                 **kwargs)
        return response.text if response!= None else None

    def down_stream(self, url, path, name=None, chunk_size=1024):
        """ 
        e.g:
            image format unknown
            http://www.xxx.com/demo.png
            path: /data
            out: /data/demo.png
        """
        basename = url.split('/')[-1]  
        if name != None: 
            pos = basename.rfind('.')
            if pos == -1:
                basename = name
            else:
                basename = name + '.' + basename[pos+1:]
        filename = os.path.join(path, basename)
        r = self.s.get(url, stream=True) 
        with open(filename, 'wb') as f:  
            for chunk in r.iter_content(chunk_size=chunk_size):  
                if chunk: 
                    # filter out keep-alive new chunks  
                    f.write(chunk);
                    f.flush()  
            return filename  

    def down_picture(self, url, path, name=None, chunk_size=1024):
        return self.down_stream(url, path, name, chunk_size)

    def down_video(self, url, path, name=None, chunk_size=1024):
        return self.down_stream(url, path, name, chunk_size)

    def ping_proxies(self, url, http, https=None, timeout=20, return_response=False):
        old_proxies = self.s.proxies
        self.update_proxies(http, https)
        result = self.ping(url, timeout, return_response)
        self.update_proxies(old_proxies['http'], old_proxies['https'])
        return result

    def ping(self, url, timeout=20, return_response=False):
        try:
            t1 = time.time()
            response = self.s.get(url, timeout=timeout)
            _t = time.time() - t1
        except Exception as e:
            _t = None
            response = None
        if return_response:
            return (_t, response)
        else:
            return _t

    def save_cookie(self, filename):
        """ todo """
        pass

    def load_cookie(self, filename):
        """ todo """
        pass

    def from_session(self, session):
        self.s = session

    def update_headers(self, headers):
        """更新session headers，可以只更新其中的一个关键字"""
        self.s.headers.update(headers)
    
    def update_proxies(self, http, https=None):
        if https == None: https = http
        proxies = {'http': http, 'https': https}
        self.s.proxies = proxies

    def update_useragent(self, useragent):
        self.headers.update({'User-Agent': useragent})

    def update_cookie(self, cookie):
        self.headers.update({'Cookie': cookie})

    def update_cookies(self, cookies):
        self.headers.update({'Cookie': cookies})

    @property
    def cookies_dict(self):
        return {c.name: c.value for c in self.cookies}

    @property
    def headers(self):
        return self.s.headers
