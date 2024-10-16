from main.tokenizer import Tokenizer

tokenizer = Tokenizer()
token_ids = tokenizer.encode('1+2= ', bos=True, eos=True)
print(token_ids)
