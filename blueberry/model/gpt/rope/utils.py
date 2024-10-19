import torch
import torch.nn as nn

# https://github.com/meta-llama/llama3/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # ** 对最后一个维度求均值
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
# def rms_norm(x, dim, eps=1e-6):
#     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
#     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
