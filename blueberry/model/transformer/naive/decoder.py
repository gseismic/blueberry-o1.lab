from ....config import nn
from .positionwise_feed_forward import PositionwiseFeedForward
from .multi_head_attention import MultiHeadAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Self-attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout(self_attn_output))
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout(cross_attn_output))
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ffn_output))
        return x

class TransformerDecoder(nn.Module):
    '''
    Note:
    src_mask（源掩码）用于在计算交叉注意力时，屏蔽掉某些编码器的输入位置。具体作用包括：
        避免处理填充标记（Padding Tokens）：
        在实际应用中，输入序列可能被填充到相同的长度。在计算注意力时，填充部分应被屏蔽掉，以防其对注意力计算产生影响。src_mask 可以用于遮蔽这些填充标记。

        实现掩蔽（Masking）：
        在某些任务中，例如在序列到序列模型中，可能需要对编码器的某些部分进行掩蔽，以避免在生成预测时关注到不相关的部分。
    '''
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.layer_norm(x)
