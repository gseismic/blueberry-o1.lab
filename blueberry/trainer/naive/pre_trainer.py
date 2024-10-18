import uuid
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from ...logger import user_logger

class PreTrainer:
    
    def __init__(self, model, data_loader=None, device='cuda', logger=None):
        self.model = model
        self.data_loader = data_loader
        self.logger = logger or user_logger
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def train_batch(self, gpt, criterion, optimizer, input_seq, target_seq):
        # input_seq, target_seq: 移位已经在dataset中处理了
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)

        output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)
        # 计算损失并使用 .reshape()
        loss = criterion(output.reshape(-1, gpt.vocab_size), target_seq.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item()
    
    def train(self,
              max_epochs, 
              lr=0.0001,
              target_loss_ratio=None,
              target_loss=None,
              verbose_freq=10,
              checkpoint_freq=None,
              final_model_file='models/final_model.pth',
              checkpoint_dir='models/checkpoint',
              overwrite_if_exists=False):
        # 损失函数和优化器
        gpt = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(gpt.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=(29 + max_epochs)//30, gamma=0.9)
        first_loss = None

        epoch_history = defaultdict(list)
        gpt.train()

        training_start_time = time.time()
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            for input_seq, target_seq in self.data_loader:
                this_loss = self.train_batch(gpt, criterion, optimizer, input_seq, target_seq)
                epoch_loss += this_loss
            
            lr_scheduler.step() 
            current_lr = lr_scheduler.get_last_lr()[0]
            if checkpoint_freq is not None and (epoch+1) % checkpoint_freq == 0:
                Path(checkpoint_dir).parent.mkdir(exist_ok=True)
                checkpoint_file = Path(checkpoint_dir) / f'chkpt_{epoch+1}.pth'
                checkpoint_file = self._ensure_good_filename(checkpoint_file, overwrite_if_exists)
                self.model.save(checkpoint_file)
                self.logger.info(f'Checkpoint model saved: {checkpoint_file}')
                
            if first_loss is None:
                first_loss = epoch_loss
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_history['epoch'].append(epoch)
            epoch_history['loss'].append(epoch_loss / len(self.data_loader))
            epoch_history['epoch_time'].append(epoch_time)
            epoch_history['lr'].append(current_lr)
            recent_avg_time = np.mean(epoch_history['epoch_time'][-20:])
            epoch_history['estimated_remaining_time'].append(
                (max_epochs - epoch -1 )* recent_avg_time
            )
            
            if (
                (target_loss_ratio is not None and first_loss > 0)
                 and epoch_loss < first_loss * target_loss_ratio
            ):
                self.logger.info(f'Early stop: reach target_loss_ratio {target_loss_ratio}')
                break

            if target_loss is not None and epoch_loss < target_loss:
                self.logger.info(f'Early stop: reach target_loss {target_loss}')
                break

            if epoch == 0 or (epoch+1) % verbose_freq == 0:
                _time = epoch_history['estimated_remaining_time'][-1]
                _hours, _minutes, _secs, _msecs =self._get_estimated_hms(_time)
                self.logger.info((
                    f'Epoch {epoch+1:>3}, Loss: {epoch_loss/len(self.data_loader):.6f}, '
                    f'lr: {current_lr:10.3e} '
                    f'Rem-Time: {_hours:02d}:{_minutes:02d}:{_secs:02d}.{_msecs:03d}'
                ))

        if not final_model_file:
            final_model_file = self._make_default_final_model_name()

        final_model_file = self._ensure_good_filename(final_model_file, overwrite_if_exists)
        Path(final_model_file).parent.mkdir(exist_ok=True)
        self.model.save(final_model_file)
        self.logger.info(f'Final model saved: {final_model_file}')

        exit_info = {
            'history': epoch_history,
            'total_training_time': time.time() - training_start_time,
            'final_loss': epoch_history['loss'][-1]
        }
        return exit_info
    
    def _get_estimated_hms(self, _time):
        _hours = int(_time/3600)
        _remaining = _time - _hours*3600
        _minutes = int(_remaining/60)
        _remaining -= _minutes*60
        _secs = int(_remaining)
        _msecs = int((_remaining - _secs)*1000)
        return _hours, _minutes, _secs, _msecs

    def _make_default_final_model_name(self):
        h = datetime.datetime.now().strftime("%y%m%d%H%M")
        # final_model_file = f'models/final_model_{str(uuid.uuid4())[-8:]}.pth'
        final_model_file = f'models/final_model_{h}.pth'
        return final_model_file
    
    def _ensure_good_filename(self, filename, overwrites_if_exists):
        old_filename = filename
        while True:
            if Path(filename).exists():
                if overwrites_if_exists:
                    self.logger.warning(f'`{old_filename}` already exists, overwriten')
                    return filename
                filename = Path(str(filename) + '.' + str(uuid.uuid4())[-8:])
            else:
                break
        if str(old_filename) != str(filename):
            self.logger.warning(f'`{old_filename}` already exists, using new filename: `{filename}`')
        return filename
            
