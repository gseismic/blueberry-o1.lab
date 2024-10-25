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
from ...utils.lr_scheduler import AdapLR

class FinetuneTrainer:
    
    def __init__(self, model, data_loader=None, device='cuda', logger=None):
        self.model = model
        self.data_loader = data_loader
        self.logger = logger or user_logger
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def train_batch(self, gpt, criterion, optimizer, input_seq, target_seq, mask_seq, grad_clip):
        # input_seq, target_seq: 移位已经在dataset中处理了
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)

        output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)
        # 计算损失并使用 .reshape()
        loss = criterion(output.reshape(-1, gpt.vocab_size), target_seq.reshape(-1))
        if mask_seq is None:
            loss = loss.mean() 
        else:
            # 微调损失和预训练损失的mask
            assert len(mask_seq) == 2
            finetune_mask, pretrain_mask = mask_seq
            alpha = 0.6
            mask_seq = alpha * finetune_mask + (1 - alpha) * pretrain_mask
            mask_seq = mask_seq.to(self.device)
            mask_seq = mask_seq.view(-1)
            # XXX 可考虑梯度累加
            # print(f'{mask_seq=}, len(mask_seq)={len(mask_seq)}')
            # print(f'{loss=}, len(loss)={len(loss)}')
            loss = torch.sum(loss * mask_seq) / torch.sum(mask_seq)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), grad_clip)
        optimizer.step()

        return loss.detach().item()
    
    def train(self,
              max_epochs, 
              lr=0.0001,
              warmup_epochs=20,
              grad_clip=None,
              target_loss_ratio=None,
              target_loss=None,
              verbose_freq=10,
              batch_verbose_freq=100,
              checkpoint_freq=None,
              final_model_file='models/final_model.pth',
              checkpoint_dir='models/checkpoint',
              overwrite_if_exists=False):
        # 检查 final_model_file 是否存在
        final_model_path = Path(final_model_file)
        if final_model_path.exists() and not overwrite_if_exists:
            self.logger.warning(f"最终模型文件 {final_model_file} 已存在且未设置覆盖。训练将不会进行。")
            return
        # 检查 checkpoint_dir 是否存在
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 如果设置了 checkpoint_freq，确保 checkpoint_dir 存在
        if checkpoint_freq is not None:
            if not checkpoint_path.exists():
                checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"检查点将每 {checkpoint_freq} 个 epoch 保存到 {checkpoint_dir}")
        
        # 记录训练开始时间
        training_start_time = time.time()
        
        # 初始化最佳损失和对应的 epoch
        best_loss = float('inf')
        best_epoch = -1
        # 确保 final_model_file 的父目录存在
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 损失函数和优化器
        gpt = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none') # do NOT use reduction='mean'
        optimizer = optim.Adam(gpt.parameters(), lr=lr) #, weight_decay=0.01)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=(29 + max_epochs)//30, gamma=0.9)
        lr_scheduler = AdapLR(optimizer, lr) #, enlarge_ratio=1.01, shrink_ratio=0.99, loss_eps=1e-8)
        first_loss = None

        epoch_history = defaultdict(list)
        gpt.train()

        for epoch in range(max_epochs):
            # warm up
            new_lr = None
            warming_up = epoch < warmup_epochs
            if warming_up:
                new_lr = lr * ((epoch + 1) / warmup_epochs)**2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            epoch_start_time = time.time()
            epoch_loss = 0

            inner_batch_losses = []
            for ii, tuple_ in enumerate(self.data_loader):
                if len(tuple_) == 2:
                    input_seq, target_seq = tuple_
                    mask_seq = None
                elif len(tuple_) == 3:
                    input_seq, target_seq, mask_seq = tuple_

                this_loss = self.train_batch(gpt, criterion, optimizer,
                                             input_seq, target_seq, mask_seq, grad_clip)
                epoch_loss += this_loss
                inner_batch_losses.append(this_loss)
                if not warming_up:
                    lr_scheduler.step_batch(this_loss)
                if batch_verbose_freq is not None and (ii+1 == 1 or (ii+1) % batch_verbose_freq == 0):
                    _avg_inner_loss = np.mean(inner_batch_losses)
                    if warming_up:
                        self.logger.info(f'\tEpoch {epoch+1}, Batch {ii+1}/{len(self.data_loader)}, lr: {new_lr:10.3e}, Batch-Loss: {_avg_inner_loss:.6f}')            
                    else:
                        self.logger.info(f'\tEpoch {epoch+1}, Batch {ii+1}/{len(self.data_loader)}, lr: {lr_scheduler.get_last_lr()[0]:10.3e}, Batch-Loss: {_avg_inner_loss:.6f}')            
            if not warming_up:
                lr_scheduler.step()

            if new_lr is None:
                current_lr = lr_scheduler.get_last_lr()[0]
            else:
                current_lr = new_lr
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
            
