from blueberry.trainer.adap import DPOTrainer
from blueberry.data.difftune.naive.mem import DifftuneDataset
from blueberry.tokenizer.naive import Tokenizer
from blueberry.utils.seed import seed_all
from blueberry.utils.xz import ensure_xzfile_decompressed
from blueberry.logger import user_logger
from blueberry.model.gpt.naive import GPT
from torch.utils.data import DataLoader
import settings

seed_all(settings.dpo_config['seed'])

ensure_xzfile_decompressed(settings.dpo_data_file)
user_logger.info(f'** init tokenizer **')
tokenizer = Tokenizer.from_files([settings.pretrain_data_file], 
                                pretrain_text_sep=settings.pretrain_text_sep)
gpt_config = settings.gpt_config | {'vocab_size': tokenizer.vocab_size}

# ** init dataset **
user_logger.info(f'** init dataset **')
dataset = DifftuneDataset.from_file(settings.dpo_data_file, 
                               tokenizer=tokenizer,
                               max_seq_len=settings.seq_len, 
                               data_sep=settings.dpo_text_sep)
data_loader = DataLoader(dataset, batch_size=settings.dpo_config['batch_size'], shuffle=True)
print(f'datasetlen: {len(dataset)}')

user_logger.info(f'Initial model for finetuning: {settings.dpo_initial_model}')
gpt = GPT.from_pretrained(settings.dpo_initial_model)
gpt_ref = GPT.from_pretrained(settings.dpo_ref_model)

trainer = DPOTrainer(model=gpt, ref_model=gpt_ref, data_loader=data_loader, logger=user_logger)
exit_info = trainer.train(max_epochs=settings.dpo_config['max_epochs'],
              lr=settings.dpo_config['lr'],
              warmup_epochs=settings.dpo_config['warmup_epochs'],
              grad_clip=settings.dpo_config['grad_clip'],
              beta=settings.dpo_config['beta'],
              target_loss_ratio=settings.dpo_config['target_loss_ratio'],
              target_loss=settings.dpo_config['target_loss'],
              final_model_file=settings.dpo_final_model_file,
              checkpoint_freq=settings.dpo_config['checkpoint_freq'],
              checkpoint_dir=settings.dpo_checkpoint_dir,
              overwrite_if_exists=True,
              verbose_freq=settings.dpo_config['verbose_freq'],
              batch_verbose_freq=settings.dpo_config['batch_verbose_freq'])

import matplotlib.pyplot as plt
plt.semilogy(exit_info['history']['epoch'], exit_info['history']['loss'], label='loss')
plt.title('loss')
plt.legend(loc='upper right')

plt.show()
