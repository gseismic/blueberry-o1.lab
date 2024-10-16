from torch.utils.data import DataLoader
from blueberry.model.gpt.naive import GPT
from blueberry.utils.seed import seed_all
from blueberry.trainer.naive import PreTrainer
from blueberry.data.pretrain import MemDataset
from main.tokenizer import Tokenizer
import settings

seed_all(36)
# ** hyper parameters **
batch_size = 10
max_epochs = 10
target_loss_ratio = 0.001
target_loss = 0.01

tokenizer = Tokenizer()
gpt_config = settings.gpt_config | {'vocab_size': tokenizer.vocab_size}
dataset = MemDataset.from_file(settings.pretrain_data_file, tokenizer=tokenizer,
                               seq_len=gpt_config['seq_len'], 
                               pretrain_text_sep=settings.pretrain_text_sep)
# dataset = MemDataset(data, tokenizer=tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout=dropout)
gpt = GPT.from_config(gpt_config)

# ** train **
trainer = PreTrainer(gpt, data_loader=data_loader)
trainer.train(max_epochs=max_epochs,
              target_loss_ratio=target_loss_ratio,
              target_loss=target_loss,
              final_model_file=settings.final_model_file,
              checkpoint_freq=10,
              overwrite_if_exists=True,
              verbose_freq=10)
