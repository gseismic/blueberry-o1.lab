from torch.utils.data import DataLoader
from blueberry.trainer.naive import PreTrainer
from blueberry.data.pretrain import MemDataset
from blueberry.utils.seed import seed_all
from blueberry.logger import user_logger
from blueberry.tokenizer.naive import Tokenizer
from blueberry.model.gpt.rope import GPT
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
                               seq_len=settings.seq_len, 
                               pretrain_text_sep=settings.pretrain_text_sep)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
if settings.pretrain_initial_model is not None:
    user_logger.info(f'Initial model for pretraining: {settings.pretrain_initial_model}')
    gpt = GPT.from_pretrained(settings.pretrain_initial_model)
else:
    user_logger.info(f'No initial model for pretraining')
    gpt = GPT.from_config(gpt_config)

param_count = gpt.count_parameters()
user_logger.info(f'Number of parameters: {param_count/1e6} M')
user_logger.info(f'Number training batches: {len(data_loader)}')

# ** train **
trainer = PreTrainer(gpt, data_loader=data_loader, logger=user_logger, device=device)
exit_info = trainer.train(max_epochs=max_epochs,
              lr=settings.pretrain_config['lr'],
              warmup_epochs=settings.pretrain_config['warmup_epochs'],
              grad_clip=settings.pretrain_config['grad_clip'],
              target_loss_ratio=target_loss_ratio,
              target_loss=target_loss,
              final_model_file=settings.final_model_file,
              checkpoint_freq=settings.checkpoint_freq,
              overwrite_if_exists=True,
              verbose_freq=verbose_freq)

import matplotlib.pyplot as plt
plt.semilogy(exit_info['history']['epoch'], exit_info['history']['loss'], label='loss')
plt.title('loss')
plt.legend(loc='upper right')

plt.show()
