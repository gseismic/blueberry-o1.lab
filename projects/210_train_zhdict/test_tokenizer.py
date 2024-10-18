from main.tokenizer import Tokenizer
import settings

# ** init dataset **
tokenizer = Tokenizer.from_files([settings.pretrain_data_file], 
                                  pretrain_text_sep=settings.pretrain_text_sep)

token_ids = tokenizer.encode('你是谁？', bos=True, eos=True)
print(token_ids)
print(tokenizer.decode(token_ids))

token_ids = tokenizer.encode('α', bos=True, eos=True)
print(token_ids)
print(tokenizer.decode(token_ids, skip_all_special=False))