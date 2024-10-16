from ....config import torch, nn, F


class MultiHeadAttention(nn.Module):
    '''
    Note:
        seq_len是动态确定的，允许跟max_seq_len不一致，应比max_seq_len小
    '''

    def __init__(self, embed_dim, num_heads, qkv_bias=True):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # d_model

        # 这里可以选择添加bias项
        # embed_dim =  d_model * num_heads
        # 这里实际是多个头并在一起，并且不会导致QKV增加head的维度
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        # query, key, value: 是原始的X, 还没有乘以W_q, W_k, W_v
        # query: [batch_size, seq_len, embed_dim]
        # key: [batch_size, seq_len, embed_dim]
        # value: [batch_size, seq_len, embed_dim]

        # print(f'MultiHeadAttention: {query.shape=}')
        # print(f'MultiHeadAttention: {key.shape=}')
        # print(f'MultiHeadAttention: {value.shape=}')
        batch_size = query.size(0)

        # 1. 线性变换并分成多个头
        # [batch_size, seq_len, embed_dim=d_model*num_heads]
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)    # [batch_size, seq_len, embed_dim]
        V = self.v_linear(value)  # [batch_size, seq_len, embed_dim]

        #print(f'MultiHeadAttention: {Q.shape=}')
        #print(f'MultiHeadAttention: {K.shape=}')
        #print(f'MultiHeadAttention: {V.shape=}')

        # 拆分为多个头并调整维度
        # [batch_size, seq_len, num_heads, d_model]
        #print(f'{Q.view(batch_size, -1, self.num_heads, self.head_dim).shape=}')
        # 不需要变动内存布局: Q.view(batch_size, -1, self.num_heads, self.head_dim): [batch_size, seq_len, num_heads, d_model]
        # 可能需要变动内存布局  .transpose(1, 2): [batch_size, num_heads, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Q/K/V: [batch_size, num_heads, seq_len, d_model]

        #print(f'MultiHeadAttention-view: {Q.shape=}')
        #print(f'MultiHeadAttention: {K.shape=}')
        #print(f'MultiHeadAttention: {V.shape=}')

        # 2. 计算注意力得分, 因为head处于额外维度，这样处理没有问题
        # Q:            [batch_size, num_heads, seq_len, d_model]
        # K.transpose(-2, -1): [batch_size, num_heads, d_model, seq_len]
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # print(f'**{scores.shape=}')
        # print(f'**{scores=}')
        if mask is not None:
            # print(f'**{mask.shape=}')
            # mask: [batch_size, 1, self.seq_len, self.seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            # print(f'After-mask:**{scores=}')
        # print(f'**{scores.shape=}')
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        # print(f'MultiHeadAttention: {attn_weights.shape=}')

        # 3. 加权求和
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        # V: [batch_size, num_heads, seq_len, d_model]
        # context: [batch_size, num_heads, seq_len, d_model]
        context = torch.matmul(attn_weights, V)
        # print(f'MultiHeadAttention: {context.shape=}')

        # 4. 合并头
        # context.transpose(1, 2): [batch_size, seq_len, num_heads, d_model]
        # context.transpose(1, 2).contiguous().view: 
        #       [batch_size, seq_len, embed_dim=num_heads*d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 5. 最后线性变换
        # out_linear: embed_dim, embed_dim
        # output:      [batch_size, seq_len, embed_dim=num_heads*d_model]
        output = self.out_linear(context)
        # print(f'{output.shape=}')

        return output, attn_weights
