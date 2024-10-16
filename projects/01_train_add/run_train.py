from torch.utils.data import DataLoader
from blueberry.gpt.naive import GPT
from blueberry.utils.seed import seed_all
from main.dataset import MemDataset
from main.tokenizer import Tokenizer
from main.pre_trainer import PreTrainer
import settings

seed_all(36)

def get_pretrain_data():
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

gpt_config = settings.gpt_config | {'vocab_size': tokenizer.vocab_size}
# gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout=dropout)
gpt = GPT.from_config(gpt_config)
max_seq_len = gpt_config['seq_len']

# ** train **
trainer = PreTrainer('demo', gpt, data_loader=dataloader, device='cuda')
trainer.train(max_epochs=100,
              target_loss_ratio=0.01,
              target_loss=0.01,
              final_model_file=settings.final_model_file,
              verbose_freq=10)

