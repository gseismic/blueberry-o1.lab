import string
from blueberry.tokenizer import DictTokenizer


def make_tokenizer(words_list):
    word2token_dict = {i: word for i, word in enumerate(words_list)} 
    base_idx = len(word2token_dict)
    word2token_dict |= {
        word: base_idx + idx
        for idx, word in enumerate(
            [
                '<padding>', '<unknown>', '<train_data/>', '</train_data>'
            ])
    }
    return DictTokenizer(word2token_dict, fn_text2words=None)


g_tokenizer = make_tokenizer(list(string.digits) + list('=+-*/%'))

__all__ = ['g_tokenizer']