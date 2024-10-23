import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from .block import Block
from ..base import BaseGPT
from .utils import RMSNorm

# from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
@dataclass
class GPTConfig:
    vocab_size : int = None # 50304
    num_layers : int = 12
    num_heads : int = 2*2 # head dim 128 suggested by @Grad62304977
    embed_dim : int = 128 # 512
    ff_dim : int = 128 * 4


class GPT(BaseGPT):

    def __init__(self, config):
        super().__init__(GPTConfig(**config))
        self.vocab_size = self.config.vocab_size # for compatibility with naive model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.embed_dim),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)]),
        ))
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight
        self.ln_f = RMSNorm(self.config.embed_dim, eps=1e-8)

    # def forward(self, idx, targets=None, return_logits=True):
    def forward(self, idx):
        # forward the GPT model itself
        # shape: (batch_size, seq_len)
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, embed_dim)
        for block in self.transformer.h:
            x = block(x)
        # x = F.rms_norm(x, (x.size(-1),))
        # x shape: (batch_size, seq_len, embed_dim)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = logits.float() # use tf32/fp32 for logits
        return logits

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     logits = logits.float() # use tf32/fp32 for logits
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     logits = logits.float() # use tf32/fp32 for logits
        #     loss = None

        # # there are performance reasons why not returning logits is prudent, if not needed
        # if not return_logits:
        #     logits = None

        # return logits, loss
    

    @torch.no_grad()
    def generate(self, start_tokens, max_len, temperature=1.0,
                 top_k=None, top_p=None, callback=None):
        from ..naive.generate import generate_sequence
        device = next(self.parameters()).device
        samples = generate_sequence(self, start_tokens, max_len, self.vocab_size, 
                                    max_seq_len=None, # no max
                                    device=device,
                                    temperature=temperature, 
                                    top_k=top_k, top_p=top_p,
                                    callback=callback)
        return samples
