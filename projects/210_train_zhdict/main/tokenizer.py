import re
from collections import Counter


class Tokenizer:
    default_begin_text_token = '<|text/|>'
    default_end_text_token = '<|/text|>'
    default_padding_token = '<|padding|>'
    default_unknown_token = '<|unknown|>'
    default_pretrain_text_sep = '|||\n\n'
    def __init__(self, 
                 non_special_tokens: list,
                 extra_special_tokens=None,
                #  num_reserved_special_tokens=100,
                 begin_text_token=None,
                 end_text_token=None,
                 padding_token=None, 
                 unknown_token=None):
        assert isinstance(non_special_tokens, list)
        assert extra_special_tokens is None or isinstance(extra_special_tokens, list)
        extra_special_tokens = extra_special_tokens or []
        for token in non_special_tokens:
            assert not token.startswith('<|') and not token.endswith('|>'), f'non-special token `{token}` is not a valid non-special token'
        # special-token: 不会被打印的token，类似于<shift><ctrl>
        self.extra_special_tokens = extra_special_tokens
        self.padding_token = padding_token or self.default_padding_token
        self.unknown_token = unknown_token or self.default_unknown_token
        self.begin_text_token = begin_text_token or self.default_begin_text_token
        self.end_text_token = end_text_token or self.default_end_text_token
        self.special_tokens = [
            self.begin_text_token,
            self.end_text_token,
            self.padding_token,
            self.unknown_token,
        ] + extra_special_tokens
        for token in self.special_tokens:
            assert self.is_special_token(token), f'special token `{token}` is not a valid special token'
        self.non_special_tokens = non_special_tokens
        self.tokens = self.non_special_tokens + self.special_tokens
        self.char_to_idx = {token: id for id, token in enumerate(self.tokens)}
        self.idx_to_char = {id: token for token, id in self.char_to_idx.items()}
        self.begin_text_id = self.char_to_idx[self.begin_text_token]
        self.end_text_id = self.char_to_idx[self.end_text_token]
        self.padding_id = self.char_to_idx[self.padding_token]
        self.unknown_id = self.char_to_idx[self.unknown_token]
        self.special_tokens_ids = set([self.char_to_idx[token] for token in self.special_tokens])
        self._vocab_size = len(self.tokens)
    
    @classmethod
    def from_files(cls, 
                  filenames: list[str],
                  extra_non_special_tokens=None,
                  extra_special_tokens=None,
                  begin_text_token=None,
                  end_text_token=None,
                  padding_token=None,
                  unknown_token=None,
                  pretrain_text_sep=None
                 ):
        assert isinstance(filenames, list)
        extra_special_tokens = extra_special_tokens or []
        extra_non_special_tokens = extra_non_special_tokens or []
        assert len(set(extra_non_special_tokens)) == len(extra_non_special_tokens)
        assert len(set(extra_special_tokens)) == len(extra_special_tokens)
        pretrain_text_sep = pretrain_text_sep or cls.default_pretrain_text_sep
        token_counter = Counter()
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
                texts = text.split(pretrain_text_sep)
                for text in texts:
                    tokens = cls._parse_tokens(text)
                    token_counter.update(tokens)
        
        # print(f'{len(token_counter)} unique tokens found.')
        full_non_special_tokens = extra_non_special_tokens
        full_extra_special_tokens = extra_special_tokens
        for token, _ in token_counter.items():
            if cls.is_special_token(token) and token not in extra_special_tokens:
                full_extra_special_tokens.append(token)
            else:
                if token not in extra_non_special_tokens:
                    full_non_special_tokens.append(token)
 
        return cls(non_special_tokens=full_non_special_tokens, 
                   extra_special_tokens=full_extra_special_tokens, 
                   begin_text_token=begin_text_token, 
                   end_text_token=end_text_token, 
                   padding_token=padding_token, 
                   unknown_token=unknown_token)
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @staticmethod
    def _parse_tokens(text):
        return re.findall(r'<\|[^>]*\|>|[\s]|[^\s]', text, re.A)
    
    @staticmethod
    def is_special_token(token):
        return token.startswith('<|') and token.endswith('|>')

    def encode(self, text,
               bos: bool, eos: bool, 
               allowed_special_tokens = set(), 
               disallowed_special_tokens = set(), 
               if_unknown='encode'):
        assert isinstance(text, str)
        assert if_unknown in ['ignore', 'encode', 'raise']
        if allowed_special_tokens and disallowed_special_tokens:
            assert len(allowed_special_tokens & disallowed_special_tokens) == 0
        # special: startswith `<|` and endswith `|>`
        tokens = self._parse_tokens(text)
        for token in tokens:
            if self.is_special_token(token):
                if allowed_special_tokens is not None and token not in allowed_special_tokens:
                    raise ValueError(f'special token `{token}` not in the allowed list')
                if disallowed_special_tokens is not None and token in disallowed_special_tokens:
                    raise ValueError(f'special token `{token}` in the disallowed list')
                assert token in self.special_tokens, f'special token `{token}` not in dictionary'
            else:
                if if_unknown == 'raise' and token not in self.non_special_tokens:
                    raise ValueError(f'non-special token `{token}` not in dictionary')
        
        if if_unknown == 'ignore':
            encoded_tokens = [self.char_to_idx[token]
                              for token in tokens if token in self.char_to_idx]
        elif if_unknown == 'encode':
            encoded_tokens = [self.char_to_idx.get(token, self.unknown_id) for token in tokens]
        else:
            encoded_tokens = [self.char_to_idx.get[token] for token in tokens]
            
        if bos:
            encoded_tokens.insert(0, self.char_to_idx[self.begin_text_token])
        if eos:
            encoded_tokens.append(self.char_to_idx[self.end_text_token])
        return encoded_tokens

    def decode(self, tokens, 
               skip_bos=False, skip_eos=False,
               skip_padding=False, skip_unknown=False, 
               skip_all_special=False):
        def should_decode(token):
            if (skip_bos or skip_all_special) and token == self.begin_text_id:
                return False
            if (skip_eos or skip_all_special) and token == self.end_text_id:
                return False
            if (skip_padding or skip_all_special) and token == self.padding_id:
                return False
            if (skip_unknown or skip_all_special) and token == self.unknown_id:
                return False
            if skip_all_special and token in self.special_tokens_ids:
                return False
            return True
        return ''.join([self.idx_to_char[token] 
                        for token in tokens 
                        if should_decode(token)])
