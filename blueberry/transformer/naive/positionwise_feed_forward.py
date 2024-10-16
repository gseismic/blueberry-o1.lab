from ...config import nn, F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个位置进行MLP，因为已经通过Atttion
        # 这个 MLP 是位置无关的，即对每个位置的特征进行相同的变换。
        # 具体来说，对于输入张量中的每个位置（即每个时间步），MLP 层都会应用相同的线性变换、激活函数和第二个线性变换。
        # 因此，MLP 层处理每个位置的特征完全独立于其他位置的特征。
        # 这种设计允许模型有效地处理序列数据中的每个位置的特征，同时保留位置之间的关系信息。
        # input-x: [batch_size, seq_len, embed_dim]
        # linear1: [batch_size, seq_len, ff_dim]
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        # linear2: [batch_size, seq_len, embed_dim]
        x = self.linear2(x)
        return x
