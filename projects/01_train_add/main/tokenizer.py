import re


class Tokenizer:
    def __init__(self, 
                 begin_text_token='<|text/|>',
                 end_text_token='<|/text|>',
                 padding_token='<|padding|>', 
                 unknown_token='<|unknown|>'):
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.begin_text_token = begin_text_token
        self.end_text_token = end_text_token
        self.special_tokens = [
            self.begin_text_token,
            self.end_text_token,
            self.padding_token,
            self.unknown_token,
        ]
        self.non_special_tokens = [
            str(i) for i in range(10)
        ] + ['+', '-', '='] + [' ', '✅', '❎']
        self.tokens = self.non_special_tokens + self.special_tokens
        self.char_to_idx = {char: idx for idx, char in enumerate(self.tokens)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.begin_text_id = self.char_to_idx[self.begin_text_token]
        self.end_text_id = self.char_to_idx[self.end_text_token]
        self.padding_id = self.char_to_idx[self.padding_token]
        self.unknown_id = self.char_to_idx[self.unknown_token]
        
        self._vocab_size = len(self.tokens)
    
    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, text,
               bos: bool, eos: bool, 
               allowed_special_tokens = set(), 
               disallowed_special_tokens = set()):
        assert isinstance(text, str)
        if allowed_special_tokens and disallowed_special_tokens:
            assert len(allowed_special_tokens & disallowed_special_tokens) == 0
        # special: startswith `<|` and endswith `|>`
        tokens = re.findall(r'<\|[^>]*\|>|[\s]|[^\s]', text, re.A)
        for token in tokens:
            if token.startswith('<|') and token.endswith('|>'):
                if allowed_special_tokens is not None and token not in allowed_special_tokens:
                    raise ValueError(f'special token `{token}` not in the allowed list')
                if disallowed_special_tokens is not None and token in disallowed_special_tokens:
                    raise ValueError(f'special token `{token}` in the disallowed list')
                assert token in self.special_tokens, f'special token `{token}` not in dictionary'
            else:
                assert token in self.non_special_tokens, f'non-special token `{token}` not in dictionary'
        encoded_tokens = [self.char_to_idx.get(char, self.unknown_id)
                          for char in tokens if char in self.char_to_idx]
        if bos:
            encoded_tokens.insert(0, self.char_to_idx[self.begin_text_token])
        if eos:
            encoded_tokens.append(self.char_to_idx[self.end_text_token])
        return encoded_tokens

    def decode(self, tokens):
        return ''.join([self.idx_to_char[token] for token in tokens])
