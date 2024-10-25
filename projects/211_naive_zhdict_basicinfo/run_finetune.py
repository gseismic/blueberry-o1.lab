from blueberry.trainer.adap import FinetuneTrainer
from blueberry.data.finetune.naive.mem import FinetuneDataset
from blueberry.tokenizer.naive import Tokenizer
from blueberry.utils.seed import seed_all
from blueberry.utils.xz import ensure_xzfile_decompressed
from blueberry.logger import user_logger
from blueberry.model.gpt.naive import GPT
from torch.utils.data import DataLoader
import settings

seed_all(settings.finetune_config['seed'])

ensure_xzfile_decompressed(settings.finetune_data_file)
user_logger.info(f'** init tokenizer **')
tokenizer = Tokenizer.from_files([settings.pretrain_data_file], 
                                pretrain_text_sep=settings.pretrain_text_sep)
gpt_config = settings.gpt_config | {'vocab_size': tokenizer.vocab_size}

# ** init dataset **
user_logger.info(f'** init dataset **')
dataset = FinetuneDataset.from_file(settings.finetune_data_file, 
                               tokenizer=tokenizer,
                               max_seq_len=settings.seq_len, 
                               data_sep=settings.finetune_text_sep)
data_loader = DataLoader(dataset, batch_size=settings.finetune_config['batch_size'], shuffle=True)

print('len', len(dataset))

user_logger.info(f'Initial model for finetuning: {settings.finetune_initial_model}')
gpt = GPT.from_pretrained(settings.finetune_initial_model)

trainer = FinetuneTrainer(model=gpt, data_loader=data_loader, logger=user_logger)
exit_info = trainer.train(max_epochs=settings.finetune_config['max_epochs'],
              lr=settings.finetune_config['lr'],
              warmup_epochs=settings.finetune_config['warmup_epochs'],
              grad_clip=settings.finetune_config['grad_clip'],
              target_loss_ratio=settings.finetune_config['target_loss_ratio'],
              target_loss=settings.finetune_config['target_loss'],
              final_model_file=settings.finetune_final_model_file,
              checkpoint_freq=settings.finetune_config['checkpoint_freq'],
              checkpoint_dir=settings.finetune_checkpoint_dir,
              overwrite_if_exists=True,
              verbose_freq=settings.finetune_config['verbose_freq'],
              batch_verbose_freq=settings.finetune_config['batch_verbose_freq'])

# raise
import matplotlib.pyplot as plt
plt.semilogy(exit_info['history']['epoch'], exit_info['history']['loss'], label='loss')
plt.title('loss')
plt.legend(loc='upper right')

plt.show()
