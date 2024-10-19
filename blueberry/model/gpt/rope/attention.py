import torch.nn as nn
import torch.nn.functional as F
from .rotary import Rotary, apply_rotary_emb
from .utils import RMSNorm

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0
        self.c_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.c_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.c_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.ln_q = RMSNorm(self.head_dim, eps=1e-8)
        self.ln_k = RMSNorm(self.head_dim, eps=1e-8)

    def forward(self, x):
        # shape: (batch_size, seq_len, embed_dim)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        # q shape: (batch_size, seq_len, num_heads, head_dim)
        cos, sin = self.rotary(q)
        # XXX 
        # q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        assert q.size(-1) == k.size(-1) == self.head_dim
        q, k = self.ln_q(q), self.ln_k(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
