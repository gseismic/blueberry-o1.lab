from torch.utils.data import DataLoader
from blueberry.model.gpt.naive import GPT
from blueberry.trainer.naive import PreTrainer
from blueberry.data.pretrain import MemDataset
from blueberry.utils.seed import seed_all
from blueberry.logger import user_logger
from main.tokenizer import Tokenizer
import settings

pretrain_config = settings.pretrain_config

batch_size = pretrain_config['batch_size']
max_epochs = pretrain_config['max_epochs']
target_loss_ratio = pretrain_config['target_loss_ratio']
target_loss = pretrain_config['target_loss']
device = pretrain_config.get('device', 'cuda')
verbose_freq = pretrain_config.get('verbose_freq', 10)

seed_all(pretrain_config.get('seed', 36))

# ** init dataset **
tokenizer = Tokenizer.from_files([settings.pretrain_data_file], 
                                pretrain_text_sep=settings.pretrain_text_sep)
gpt_config = settings.gpt_config | {'vocab_size': tokenizer.vocab_size}
dataset = MemDataset.from_file(settings.pretrain_data_file, 
                               tokenizer=tokenizer,
                               seq_len=gpt_config['seq_len'], 
                               pretrain_text_sep=settings.pretrain_text_sep)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gpt = GPT.from_config(gpt_config)

param_count = gpt.count_parameters()
print(f'number of parameters: {param_count/1e6} M')

# ** train **
trainer = PreTrainer(gpt, data_loader=data_loader, logger=user_logger, device=device)
exit_info = trainer.train(max_epochs=max_epochs,
              target_loss_ratio=target_loss_ratio,
              target_loss=target_loss,
              final_model_file=settings.final_model_file,
              checkpoint_freq=None,
              overwrite_if_exists=True,
              verbose_freq=verbose_freq)

import matplotlib.pyplot as plt
plt.semilogy(exit_info['history']['epoch'], exit_info['history']['loss'], label='loss')
plt.title('loss')
plt.legend(loc='upper right')

plt.show()
