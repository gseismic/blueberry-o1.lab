from abc import abstractmethod
import torch
import torch.nn as nn

class BaseGPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        gpt = torch.load(path)
        return gpt

    @classmethod
    def from_pretrained(cls, path):
        return cls.load(path)
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
    @abstractmethod
    def forward(self, idx, targets=None, return_logits=True):
        raise NotImplementedError
