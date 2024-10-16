import torch
from torch.utils.data import DataLoader
from blueberry.gpt.naive import GPT
from blueberry.utils.seed import seed_all
from main.dataset import MemDataset
from main.tokenizer import Tokenizer
from main.pre_trainer import PreTrainer

seed_all(36)

def get_pretrain_data():
    import settings
    with open(settings.pretrain_data_file) as f:
        text = f.read()
        lines = text.strip().split(settings.text_sep)
    return lines

data = get_pretrain_data()
print(data)

tokenizer = Tokenizer()
batch_size = 10
dataset = MemDataset(data, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gpt_config = {
    "num_layers": 2,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 512,
    "vocab_size": tokenizer.vocab_size,
    "seq_len": 20,
    "dropout": 0.1
}
# gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout=dropout)
gpt = GPT.from_config(gpt_config)
max_seq_len = gpt_config['seq_len']

# ** train **
trainer = PreTrainer(gpt, data_loader=dataloader, device='cuda')
trainer.train(max_epochs=300,
              target_loss_ratio=0.01,
              target_loss=0.01,
              verbose_freq=10)

# ** generate **
start_tokens = tokenizer.encode("3", bos=True, eos=False)
print(f'{start_tokens=}')

max_generate_len = 20

# 温度调整生成
generated_sequence = gpt.generate(start_tokens, max_generate_len, temperature=0.7)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **temperature**:", decoded_sequence)

# Top-k 采样生成
generated_sequence = gpt.generate(start_tokens, max_generate_len, temperature=0.7, top_k=3)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-k** sampling:", decoded_sequence)

# Top-p 采样生成
generated_sequence = gpt.generate(start_tokens, max_generate_len, temperature=0.7, top_p=0.9)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-p** sampling:", decoded_sequence)

