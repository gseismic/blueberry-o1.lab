import torch
import torch.nn as nn
import torch.optim as optim
from .logger import user_logger

class PreTrainer:
    
    def __init__(self, model, data_loader=None, device='cuda', logger=None):
        self.model = model
        self.data_loader = data_loader
        self.logger = logger or user_logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def train(self, max_epochs, 
              target_loss_ratio=None, 
              target_loss=None,
              verbose_freq=10):
        # 损失函数和优化器
        gpt = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(gpt.parameters(), lr=0.0001)
        first_loss = None

        # 训练步骤
        gpt.train()
        for epoch in range(max_epochs):
            epoch_loss = 0
            for input_seq, target_seq in self.data_loader:
                # input_seq, target_seq: 移位已经在dataset中处理了
                # print(f'{tokenizer.decode([input_seq])=}')
                # print(f'{tokenizer.decode([target_seq])=}')
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # 生成目标序列，预测下一个词
                output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)
                # 目标右移1位
                # 计算损失并使用 .reshape()
                loss = criterion(output.reshape(-1, gpt.vocab_size), target_seq.reshape(-1))

                optimizer.zero_grad()   # 梯度重置为0
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
            if first_loss is None:
                first_loss = epoch_loss
            
            if target_loss_ratio is not None and first_loss > 0 and epoch_loss < first_loss * target_loss_ratio:
                self.logger.info(f'Early stop: reach target_loss_ratio {target_loss_ratio}')
                break
            if target_loss is not None and epoch_loss < target_loss:
                self.logger.info(f'Early stop: reach target_loss {target_loss}')
                break

            if epoch == 0 or (epoch+1) % verbose_freq == 0:
                self.logger.info(f'Epoch {epoch+1}, Loss: {epoch_loss/len(self.data_loader):.6f}')
