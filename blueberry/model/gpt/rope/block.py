import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalSelfAttention
from .utils import RMSNorm

# from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
# with modifications
class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron)
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.embed_dim, config.ff_dim, bias=False)
        self.c_proj  = nn.Linear(config.ff_dim, config.embed_dim, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        eps = 1e-8
        self.ln_1 = RMSNorm(config.embed_dim, eps=eps)
        self.ln_2 = RMSNorm(config.embed_dim, eps=eps)
        self.embed_dim = config.embed_dim

    def forward(self, x):
        # shape: (batch_size, seq_len, embed_dim)
        # x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        # x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
