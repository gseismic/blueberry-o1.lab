from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    
    def __init__(self, padding=1, unknown=0) -> None:
        super().__init__()
        assert unknown != padding
        self.padding = padding
        self.unknown = unknown
    
    @abstractmethod
    def encode(self, text):
        pass
    
    @abstractmethod
    def decode(self, tokens):
        pass
    
    # @abstractmethod
    # @property
    # def word2token_dict(self):
    #     pass
    
    # @abstractmethod
    # @property
    # def token2word_dict(self):
    #     pass
    

class DictTokenizer(BaseTokenizer):
    
    def __init__(self,
                 word2token_dict,
                 fn_text2words=None,
                 padding=1, unknown=0) -> None:
        super().__init__(padding=padding, unknown=unknown)
        self._word2token_dict = word2token_dict
        self._token2word2_dict = {t: w for w, t in self._word2token_dict.items()}
        self.fn_text2words = fn_text2words
    
    def encode(self, text):
        if self.fun_text2words is None:
            words = iter(text)
        else:
            words = self.fn_text2words(text)
        return [self.word2token_dict[word] for word in words]
    
    def decode(self, tokens):
        return [self.token2word2_dict[token] for token in tokens]
    
    @property
    def word2token_dict(self):
        return self._word2token_dict
    
    @property
    def token2word_dict(self):
        return self._token2word2_dict
