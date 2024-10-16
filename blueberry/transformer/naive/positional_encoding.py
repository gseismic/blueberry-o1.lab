from ...config import torch, nn 
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, seq_len, pe_base=10000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(pe_base) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        # print(f'{self.encoding.shape=}')

    def forward(self, x):
        device = x.device
        if self.encoding.device != device:
            self.encoding.to(device)
        # x: [batch_size, seq_len]
        # print(f'PositionalEncoding: {x.shape=}')
        y = self.encoding[:, :x.size(1)]
        # print(f'PositionalEncoding-y: {y.shape=}')
        # y: [1, seq_len, embed_dim]
        return y
