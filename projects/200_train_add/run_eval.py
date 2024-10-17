from blueberry.model.gpt.naive import GPT
from main.tokenizer import Tokenizer
from main import settings


tokenizer = Tokenizer()
gpt = GPT.load(settings.final_model_file)

# ** generate **
start_tokens = tokenizer.encode("12 =", bos=True, eos=False)
print(f'{start_tokens=}')

max_generate_len = 20 # settings.gpt_config['seq_len']

# 温度调整生成
def test_temperature(num_test, temperature):
    print(f"Generated sequence with **temperature**: {temperature=}")
    for i in range(num_test):
        generated_sequence = gpt.generate(start_tokens, max_generate_len,
                                          temperature=temperature)
        decoded_sequence = tokenizer.decode(generated_sequence)
        decoded_sequence = tokenizer.decode(generated_sequence, do_format=True)
        print(f'\t{decoded_sequence}')

# Top-k 采样生成
def test_top_k(num_test, temperature, top_k):
    print(f"Generated sequence with **top-k** sampling: {temperature=}, {top_k=}")
    for i in range(num_test):
        generated_sequence = gpt.generate(start_tokens, max_generate_len,
                                          temperature=temperature, top_k=top_k)
        decoded_sequence = tokenizer.decode(generated_sequence, do_format=True)
        print(f'\t{decoded_sequence}')


def test_top_p(num_test, temperature, top_p):
    print(f"Generated sequence with **top-p** sampling: {temperature=}, {top_p=}")
    for i in range(num_test):
        generated_sequence = gpt.generate(start_tokens, max_generate_len,
                                          temperature=temperature, top_p=top_p)
        decoded_sequence = tokenizer.decode(generated_sequence, do_format=True)
        print(f'\t{decoded_sequence}')


test_temperature(num_test=10, temperature=0.1)
test_temperature(num_test=10, temperature=0.9)
test_temperature(num_test=10, temperature=2.0)
test_temperature(num_test=10, temperature=5.0)

test_top_k(num_test=10, temperature=0.9, top_k=2)
test_top_k(num_test=10, temperature=0.9, top_k=8)

test_top_p(num_test=10, temperature=0.9, top_p=0.2)
test_top_p(num_test=10, temperature=0.9, top_p=0.8)
