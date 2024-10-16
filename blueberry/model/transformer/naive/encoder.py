from ....config import nn
from .positionwise_feed_forward import PositionwiseFeedForward
from .multi_head_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    # 这样的Block可以叠加多层，一般6层
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # print(f'TransformerBlock: {x.shape=}')
        # x: 是还没有乘以W_q, W_k, W_v的原始embeding输入
        # XXX 似乎推荐先layer_norm?
        # attn_output: [batch_size, seq_len, embed_dim=num_heads*d_model]
        attn_output, _ = self.attention(x, x, x, mask)
        # x: [batch_size, seq_len, embed_dim=num_heads*d_model]
        x = self.layer_norm1(x + self.dropout(attn_output))
        # print(f'TransformerBlock-x: {x.shape=}')
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        # print(f'TransformerBlock-out: {x.shape=}')
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_dim]
        # print(f'TransformerEncoder: {x.shape=}')
        for layer in self.layers:
            x = layer(x, mask)
        # y: [batch_size, seq_len, embed_dim]
        y = self.layer_norm(x)
        # print(f'TransformerEncoder: {y.shape=}')
        return y
